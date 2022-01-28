
use std::error::Error;

#[derive(Debug, Clone)]
pub struct Array {
    pub data: Vec<f64>,
    dimensions: Vec<usize>,
}

impl Array {
    pub fn new(data: Vec<f64>, dimensions: Vec<usize>) -> Self {
        assert_eq!(data.len(), dimensions.iter().product());
        Array {data, dimensions}
    }

    pub fn empty() -> Self {
        Array::new(vec![], vec![0])
    }
}

impl Array {
    pub fn n_dim(&self) -> usize {
        self.dimensions.len()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn copy_array(&mut self, other: &Array) {
        if self.len() == other.len() {
            self.data.copy_from_slice(other.data.as_slice());
        } else {
            self.data = other.data.clone();
        }
        self.dimensions.clear();
        self.dimensions.extend_from_slice(&other.dimensions);
    }

    pub fn reshape(&mut self, dimensions: &[usize]) {
        assert_eq!(self.data.len(), dimensions.iter().product());
        self.dimensions.clear();
        self.dimensions.extend_from_slice(dimensions);
    }
}

pub struct Network {
    operators: Vec<Box<dyn Operator>>,
    input: Array,
}

impl Network {
    pub fn new() -> Self {
        let operators = Vec::new();
        Network {operators, input: Array::empty()}
    }
}

impl Network {
    pub fn add_operator(&mut self, operator: Box<dyn Operator>) -> Result<(), Box<dyn Error>> {
       self.operators.push(operator);
       Ok(())
    }

    pub fn count_parameters(&self) -> usize {
        let mut total: usize = 0;
        for operator in self.operators.iter() {
            total += operator.count_parameters();
        }  // DEV Note: Replace by a call to fold()

        total
    }

    pub fn execute_inference<'a>(&mut self, input: &'a Array) -> &Array {
        self.input = input.to_owned();  // DEV Note: MUST BE OPTIMIZED
        self.operators
            .iter_mut()
            .fold(&self.input, |acc, op| op.execute_operation(acc)
            .unwrap())
    }
}

pub trait Operator {
    fn execute_operation<'a>(&mut self, input: &'a Array) -> Result<&Array, &'static str>;
    fn count_parameters(&self) -> usize;
    fn initialize_array(&mut self, size: usize, dimensions: Vec<usize>);
}

pub struct Flatten {
    pub output: Array,
}

impl Flatten {
    pub fn new() -> Self {
        Flatten {output: Array::empty()}
    }
}

impl Operator for Flatten {
    fn execute_operation<'a>(&mut self, input: &'a Array) -> Result<&Array, &'static str>{
        self.output.copy_array(input);
        self.output.reshape(&[input.len()]);
        Ok(&self.output)
    }

    fn count_parameters(&self) -> usize { 0 }
    fn initialize_array(&mut self, _size: usize, _dimensions: Vec<usize>) {}
}

pub struct ReLu {
    pub output: Array,
}

impl ReLu {
    pub fn new() -> Self {
        ReLu {output: Array::empty()}
    }
}

impl Operator for ReLu {
    fn execute_operation<'a>(&mut self, input: &'a Array) -> Result<&Array, &'static str>{
        self.initialize_array(input.len(), vec![input.len()]);

        for (n, item) in input.data.iter().enumerate() {
            self.output.data[n] = item.max(0.0);
        }
        Ok(&self.output)
    }

    fn count_parameters(&self) -> usize { 0 }

    fn initialize_array(&mut self, size: usize, dimensions: Vec<usize>) {
        if self.output.len() == 0 {
            self.output = Array::new(vec![0.; size], dimensions);
        }
    }
}

pub struct LinearCombination {
    pub output: Array,
    weights: Array,
    bias: Array,
}

impl LinearCombination {
    pub fn new(weights: Vec<f64>, weights_dimensions: Vec<usize>, bias: Vec<f64>) -> Self {
        LinearCombination {output: Array::empty(),
                           weights: Array::new(weights.clone(), weights_dimensions),
                           bias: Array::new(bias.clone(), vec![bias.len()])}
    }
}

impl Operator for LinearCombination {
    fn execute_operation<'a>(&mut self, input: &'a Array) -> Result<&Array, &'static str>{
        if input.len() * self.bias.len() != self.weights.len() {
            eprintln!("[Linear Combination] Argument `input` or bias dimension mismatch regarding to weights matrix");
            return Err("Arrays dimension mismatch");
        }

        self.initialize_array(self.bias.len(), self.bias.dimensions.clone());

        let column_size = self.bias.len();
        let mut inbound = 0;
        let mut outbound = column_size;

        for (n, item) in input.data.iter().enumerate() {
            // DEV Note: use iteration by chunks via chunks()
            if n > 0 {
                inbound = n * column_size;
                outbound = inbound + column_size;
            }
            let weights_column = &self.weights.data[inbound..outbound];
            for m in 0..column_size {
                self.output.data[m] += weights_column[m] * item;
            }
        }

        for (n, item) in self.bias.data.iter().enumerate() {
            self.output.data[n] += item;
        }

        Ok(&self.output)
    }

    fn count_parameters(&self) -> usize {
        // DEV Note:  Do I need to add the output vector length to the count ?
        self.weights.data.len() + self.bias.len()
    }

    fn initialize_array(&mut self, size: usize, dimensions: Vec<usize>) {
        self.output.copy_array(&Array::new(vec![0.0; size], dimensions.clone()));
    }

}

pub struct SoftMax {
    pub output: Array,
}

impl SoftMax {
    pub fn new() -> Self {
        SoftMax {output: Array::empty()}
    }
}

impl Operator for SoftMax {
    fn execute_operation<'a>(&mut self, input: &'a Array) -> Result<&Array, &'static str>{
        self.initialize_array(input.len(), vec![input.len()]);
        let normalized_input: Vec<f64> = input.data.iter().map(|x| x.exp()).collect();
        let normalized_sum: f64 = normalized_input.iter().sum();
        // DEV Note: perform a for_each() to apply modifications instead of reallocating a new array
        self.output = Array::new(normalized_input.iter().map(|x| x / normalized_sum).collect(), vec![normalized_input.len()]);
        Ok(&self.output)
    }

    fn count_parameters(&self) -> usize { 0 }

    fn initialize_array(&mut self, size: usize, dimensions: Vec<usize>) {
        if self.output.len() == 0 {
            self.output = Array::new(vec![0.; size], dimensions);
        }
    }

}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flatten_2_dimensions_array() {
        let input_array = Array::new(
            vec![1., 2., 3.,
                 4., 5., 6.,],
            vec![2, 3]
            );
        let mut network = Network::new();
        network.add_operator(Box::new(Flatten::new())).unwrap();

        let result_array = network.execute_inference(&input_array);
        assert_eq!(result_array.n_dim(), 1);
        assert_eq!(result_array.dimensions, vec![6]);

    }

    #[test]
    fn flatten_already_flat_array() {
        let input_array = Array::new(vec![1., 2., 3., 4.], vec![4]);
        let mut network = Network::new();
        network.add_operator(Box::new(Flatten::new())).unwrap();

        let result_array = network.execute_inference(&input_array);
        assert_eq!(result_array.n_dim(), 1);
        assert_eq!(result_array.dimensions, vec![4]);

    }

    #[test]
    fn basic_chained_relu_1_dim() {
        let input_array = Array::new(vec![4., -1., 5., 0., -2.], vec![5]);
        let mut network = Network::new();
        network.add_operator(Box::new(ReLu::new())).unwrap();
        network.add_operator(Box::new(ReLu::new())).unwrap();

        assert_eq!(*network.execute_inference(&input_array).data, vec![4., 0., 5., 0., 0.]);
    }

    #[test]
    fn basic_linear_combination_1_dim() {
        let input_array = Array::new(vec![1., 2., 3., 4.], vec![4]);
        let mut network = Network::new();
        let weights = vec![
            1., 5.,
            2., 6.,
            3., 7.,
            4., 8.,
        ];
        let weights_dim = vec![4, 2];
        let lin_comb = LinearCombination::new(weights, weights_dim, vec![2., 2.]);
        network.add_operator(Box::new(lin_comb)).unwrap();

        assert_eq!(*network.execute_inference(&input_array).data, vec![32., 72.]);
    }
}
