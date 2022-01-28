
use std::error::Error;

#[derive(Debug, Clone)]
pub struct Array {
    pub data: Vec<f64>,
}

impl Array {
    pub fn new(data: Vec<f64>) -> Self {
        Array {data}
    }
}

pub struct Network {
    operators: Vec<Box<dyn Operator>>,
    input: Array,
}

impl Network {
    pub fn new() -> Self {
        let operators = Vec::new();
        Network {operators, input: Array::new(vec![])}
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
        }

        total
    }

    pub fn execute_inference<'a>(&mut self, input: &'a Array) -> &Array {
        self.input = input.to_owned();  // Note: à optimiser
        self.operators.iter_mut().fold(&self.input, |acc, op| op.execute_operation(acc).unwrap())
    }
}

pub trait Operator {
    fn execute_operation<'a>(&mut self, input: &'a Array) -> Result<&Array, &'static str>;
    fn count_parameters(&self) -> usize;
}

pub struct Flatten {
    pub output: Option<Array>,
}

impl Flatten {
    pub fn new() -> Self {
        Flatten {output: None}
    }
}

impl Operator for Flatten {
    fn execute_operation<'a>(&mut self, input: &'a Array) -> Result<&Array, &'static str>{
        let result = Array::new(input.data.iter().flatten().collect::<Vec<f64>>());
        println!("Result: {:?}", result);  // DEBUG
        self.output = Some(result);
        Ok(self.output.as_ref().unwrap())
    }

    fn count_parameters(&self) -> usize { 0 }
}

pub struct ReLu {
    pub output: Option<Array>,
}

impl ReLu {
    pub fn new() -> Self {
        ReLu {output: None}
    }
}

impl Operator for ReLu {
    fn execute_operation<'a>(&mut self, input: &'a Array) -> Result<&Array, &'static str>{
        let mut results = Array::new(vec![0.0; input.data.len()]);

        for (n, item) in input.data.iter().enumerate() {
            results.data[n] = item.max(0.0);
        }

        println!("Result: {:?}", results);  // DEBUG
        self.output = Some(results);
        Ok(self.output.as_ref().unwrap())
    }

    fn count_parameters(&self) -> usize { 0 }
}

pub struct LinearCombination {
    pub output: Array,
    weights: Array,
    bias: Vec<f64>,
}

impl LinearCombination {
    pub fn new(weights: Array, bias: Vec<f64>) -> Result<Self,  &'static str>{
        let remainder = weights.data.len() % bias.len();
        match remainder {
            0 => Ok(LinearCombination {output: Array::new(bias.clone()),
                                       weights,
                                       bias}),
            _ => {
                eprintln!("[Linear Combination] Bias vector (len: {}) is not a mutliple of weigths matrix (len: {})",
                          bias.len(), weights.data.len());
                Err("Arrays dimension mismatch")
            }
        }
    }
}

impl Operator for LinearCombination {
    fn execute_operation<'a>(&mut self, input: &'a Array) -> Result<&Array, &'static str>{
        if input.data.len() * self.bias.len() != self.weights.data.len() {
            eprintln!("[Linear Combination] Argument `input` dimension mismatch");
            return Err("Arrays dimension mismatch");
        }

        let column_size = self.bias.len();
        let mut inbound = 0;
        let mut outbound = column_size;

        for (n, item) in input.data.iter().enumerate() {
            if n > 0 {
                inbound = n * column_size;
                outbound = inbound + column_size;
            }
            let weights_column = &self.weights.data[inbound..outbound];
            println!("in: {} | out: {} | weights_col: {:?}", inbound, outbound, weights_column);
            for m in 0..column_size {
                self.output.data[m] += weights_column[m] * item;
                println!("item :  {} | weight: {} | comput: {}", item, self.weights.data[m * (n + 1)], self.output.data[m]);
            }
        }

        println!("Output: {:?}", self.output);  // DEBUG
        Ok(&self.output)
    }

    fn count_parameters(&self) -> usize {
        // Faut-il que je compte les paramètres de sortie de la combinaison ?
        self.weights.data.len() + self.bias.len()
    }
}

pub struct SoftMax {
    pub output: Option<Array>,
}

impl SoftMax {
    pub fn new() -> Self {
        SoftMax {output: None}
    }
}

impl Operator for SoftMax {
    fn execute_operation<'a>(&mut self, input: &'a Array) -> Result<&Array, &'static str>{
        //let mut results = Array::new(vec![0.0; input.data.len()]);
        let normalized_input: Vec<f64> = input.data.iter().map(|x| x.exp()).collect();
        let normalized_sum: f64 = normalized_input.iter().sum();
        let results = Array::new(normalized_input.iter().map(|x| x / normalized_sum).collect());
        println!("Result: {:?}", results);  // DEBUG
        self.output = Some(results);
        Ok(self.output.as_ref().unwrap())
    }

    fn count_parameters(&self) -> usize { 0 }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flatten_2_dimensions_array() {
        let input_array = Array::new(
            vec![
                vec![1., 2., 3.],
                vec![4., 5., 6.,]
                ]
            );
        let mut network = Network::new();
        network.add_operator(Box::new(Flatten::new())).unwrap();

        let expected_array = Array::new(vec![1., 2., 3., 4., 5., 6.]);
        assert_eq!(*network.execute_inference(&input_array).data, expected_array.data)
    }

    #[test]
    fn flatten_already_flat_array() {
        let input_array = Array::new(vec![1., 2., 3., 4.]);
        let mut network = Network::new();
        network.add_operator(Box::new(Flatten::new())).unwrap();

        assert_eq!(*network.execute_inference(&input_array).data, input_array.data)
    }

    #[test]
    fn basic_chained_relu_1_dim() {
        let input_array = Array::new(vec![4., -1., 5., 0., -2.]);
        let mut network = Network::new();
        network.add_operator(Box::new(ReLu::new())).unwrap();
        network.add_operator(Box::new(ReLu::new())).unwrap();

        let expected_array = Array::new(vec![4., 0., 5., 0., 0.]);
        assert_eq!(*network.execute_inference(&input_array).data, expected_array.data)
    }

    #[test]
    fn basic_linear_combination_1_dim() {
        let input_array = Array::new(vec![1., 2., 3., 4.]);
        let mut network = Network::new();
        let weights = Array::new(vec![
            1., 5.,
            2., 6.,
            3., 7.,
            4., 8.,
        ]);
        let lin_comb = LinearCombination::new(weights, vec![2., 2.]).unwrap();
        network.add_operator(Box::new(lin_comb)).unwrap();

        let expected_array = Array::new(vec![32., 72.]);
        assert_eq!(*network.execute_inference(&input_array).data, expected_array.data)
    }

    #[test]
    fn basic_soft_max_1_dim() {
        let input_array = Array::new(vec![1., 2., 5., -1.]);
        let mut network = Network::new();
        network.add_operator(Box::new(SoftMax::new())).unwrap();

        // Pas d'assertion ici, utilisé en débug pour voir si la sortie est bien celle imaginée
        network.execute_inference(&input_array);
    }

}
