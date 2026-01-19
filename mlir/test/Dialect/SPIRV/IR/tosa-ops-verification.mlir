// RUN: mlir-opt --verify-diagnostics %s

//===----------------------------------------------------------------------===//
// spirv.TOSA.ArgMax
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @argmax_non_i32_result_element_type(%arg0: !spirv.arm.tensor<3x28x17x17xi8>) -> (!spirv.arm.tensor<3x28x17xi16>) {
  // expected-error @+1 {{op result #0 must be 1D/2D/3D/4D/5D tensorArm of Int32 values}}
  %2 = spirv.Tosa.ArgMax axis = 3, nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<3x28x17x17xi8> -> !spirv.arm.tensor<3x28x17xi16>
  spirv.ARM.GraphOutputs %2 : !spirv.arm.tensor<3x28x17xi16>
}

spirv.ARM.Graph @argmax_incorrect_output_rank(%arg0: !spirv.arm.tensor<3x28x17x17xi8>) -> (!spirv.arm.tensor<3x28xi32>) {
  // expected-error @+1 {{op result rank must be max of 1 and (input rank - 1), got 2}}
  %2 = spirv.Tosa.ArgMax axis = 2, nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<3x28x17x17xi8> -> !spirv.arm.tensor<3x28xi32>
  spirv.ARM.GraphOutputs %2 : !spirv.arm.tensor<3x28xi32>
}

spirv.ARM.Graph @argmax_axis_value_not_in_input_rank_range(%arg0: !spirv.arm.tensor<3x28x17x17xi8>) -> (!spirv.arm.tensor<3x28x17xi32>) {
  // expected-error @+1 {{op specified axis is greater than the rank of input, got axis = 4 and input rank = 4}}
  %2 = spirv.Tosa.ArgMax axis = 4, nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<3x28x17x17xi8> -> !spirv.arm.tensor<3x28x17xi32>
  spirv.ARM.GraphOutputs %2 : !spirv.arm.tensor<3x28x17xi32>
}
