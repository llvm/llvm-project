// RUN: mlir-opt %s -split-input-file -verify-diagnostics

func.func @ins_1_index_outs_none_results_1_index(%arg0 : index) -> index {
  %0 = test.destination_style_op ins(%arg0 : index) -> index
  func.return %0 : index
}

// -----

func.func @ins_1_index_outs_1_tensor_results_1_index(%arg0 : index, %arg1 : tensor<2x2xf32>) -> index {
  // expected-error @+1 {{op expected the number of tensor results (0) to be equal to the number of output tensors (1)}}
  %0 = test.destination_style_op ins(%arg0 : index) outs(%arg1 : tensor<2x2xf32>) -> index
  func.return %0 : index
}

// -----

func.func @ins_1_tensor_outs_none_results_1_index(%arg0 :tensor<2x2xf32>) -> index {
  %0 = test.destination_style_op ins(%arg0 : tensor<2x2xf32>) -> index
  func.return %0 : index
}

// -----

func.func @ins_1_tensor_outs_1_tensor_results_1_index(%arg0 :tensor<2x2xf32>, %arg1 : tensor<2x2xf32>) -> index {
  // expected-error @+1 {{op expected the number of tensor results (0) to be equal to the number of output tensors (1)}}
  %0 = test.destination_style_op ins(%arg0 : tensor<2x2xf32>) outs(%arg1 : tensor<2x2xf32>) -> index
  func.return %0 : index
}

// -----

func.func @ins_1_index_outs_none_results_1_tensor(%arg0 : index) -> tensor<2x2xf32> {
  // expected-error @+1 {{op expected the number of tensor results (1) to be equal to the number of output tensors (0)}}
  %0 = test.destination_style_op ins(%arg0 : index) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

func.func @ins_1_index_outs_1_tensor_results_1_tensor(%arg0 : index, %arg1 : tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = test.destination_style_op ins(%arg0 : index) outs(%arg1 : tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

func.func @ins_1_tensor_outs_none_results_1_tensor(%arg0 :tensor<2x2xf32>) -> tensor<2x2xf32> {
  // expected-error @+1 {{op expected the number of tensor results (1) to be equal to the number of output tensors (0)}}
  %0 = test.destination_style_op ins(%arg0 : tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

func.func @ins_1_tensor_outs_1_tensor_results_1_tensor(%arg0 :tensor<2x2xf32>, %arg1 : tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = test.destination_style_op ins(%arg0 : tensor<2x2xf32>) outs(%arg1 : tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}
