
mod parameters_nn;
mod dataset;

use dataset::DATASET;
use parameters_nn::*;
use trial;


const IMAGE_SIZE: [usize; 2] = [28, 28];
const GRAY_SCALE_CHARS: [&str; 10] = [" ", ".", ":", "-", "=", "+", "*", "#", "%", "@"];
const SEPARATOR: &str = "----------------------------------------------";


fn display_ascii_art(number: &trial::Array, height: usize, width: usize) {
    let mut inbound = 0;
    let mut outbound = width;

    for n in 0..height {
        if n > 0 {
            inbound = n * width;
            outbound = inbound + width;
        }

        let line: String = number.data[inbound..outbound]
            .iter()
            .map(|x| GRAY_SCALE_CHARS[(x * 10.).min(9.) as usize])
            .collect();
        println!("{:?}", line);
    }
}

fn main() {
    println!("Dataset contains {} items", DATASET.len());
    print!("Network initialization... ");

    let mut network = trial::Network::new();

    network.add_operator(Box::new(trial::Flatten::new())).unwrap();
    network.add_operator(Box::new(trial::LinearCombination::new(WEIGHTS_1.to_vec(),
                                  vec![WEIGHTS_1.len()],
                                  BIAS_1.to_vec()))).unwrap();
    network.add_operator(Box::new(trial::ReLu::new())).unwrap();
    network.add_operator(Box::new(trial::LinearCombination::new(WEIGHTS_2.to_vec(),
                                  vec![WEIGHTS_2.len()],
                                  BIAS_2.to_vec()))).unwrap();
    network.add_operator(Box::new(trial::ReLu::new())).unwrap();
    network.add_operator(Box::new(trial::LinearCombination::new(WEIGHTS_3.to_vec(),
                                  vec![WEIGHTS_3.len()],
                                  BIAS_3.to_vec()))).unwrap();
    network.add_operator(Box::new(trial::SoftMax::new())).unwrap();

    println!("Done");
    println!("Inference engine has {} parameters", network.count_parameters());
    println!("{}", SEPARATOR);
    println!("{}", SEPARATOR);

    for image in DATASET[..3].iter() {
        let image_array = trial::Array::new(image.to_vec(), IMAGE_SIZE.to_vec());
        display_ascii_art(&image_array, IMAGE_SIZE[0], IMAGE_SIZE[1]);
        let results_array = network.execute_inference(&image_array);
        let max_value: f64 = results_array.data
            .clone()
            .into_iter()
            .reduce(f64::max)
            .unwrap();
        let max_index: usize = results_array.data
            .iter()
            .position(|n| n.eq(&max_value)).unwrap();

        println!("Above image represents number '{}' (score: {})", max_index, max_value);
        println!("results: {:?}", results_array);  // DEBUG

        println!("{}", SEPARATOR);
        println!("{}", SEPARATOR);
    }
    println!("Dataset exhausted");
}
