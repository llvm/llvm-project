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
  // expected-error @+1 {{op failed to verify that output rank must be equal to max(1, rank(input))}}
  %2 = spirv.Tosa.ArgMax axis = 2, nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<3x28x17x17xi8> -> !spirv.arm.tensor<3x28xi32>
  spirv.ARM.GraphOutputs %2 : !spirv.arm.tensor<3x28xi32>
}

spirv.ARM.Graph @argmax_axis_value_not_in_input_rank_range(%arg0: !spirv.arm.tensor<3x28x17x17xi8>) -> (!spirv.arm.tensor<3x28x17xi32>) {
  // expected-error @+1 {{op failed to verify that axis attribute value should be lower than rank(input)}}
  %2 = spirv.Tosa.ArgMax axis = 4, nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<3x28x17x17xi8> -> !spirv.arm.tensor<3x28x17xi32>
  spirv.ARM.GraphOutputs %2 : !spirv.arm.tensor<3x28x17xi32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Conv2D
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @conv2d_wrong_input_integer_element_type(%arg0: !spirv.arm.tensor<1x65535x3x1xi32>, %arg1: !spirv.arm.tensor<7x1x1x1xi32>, %arg2: !spirv.arm.tensor<1xi64>) -> (!spirv.arm.tensor<1x65536x2x7xi64>) {
  %5 = spirv.Constant dense<35> : !spirv.arm.tensor<1xi32>
  %6 = spirv.Constant dense<57> : !spirv.arm.tensor<1xi32>
  // expected-error @+1 {{op failed to verify that if input has type integer then input must have a type in [8-bit signless integer,16-bit signless integer]}}
  %7 = spirv.Tosa.Conv2D pad = [1, 0, 0, 0], stride = [1, 2], dilation = [7, 1], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x65535x3x1xi32>, !spirv.arm.tensor<7x1x1x1xi32>, !spirv.arm.tensor<1xi64>, !spirv.arm.tensor<1xi32>, !spirv.arm.tensor<1xi32> -> !spirv.arm.tensor<1x65536x2x7xi64>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x65536x2x7xi64>
}

spirv.ARM.Graph @conv2d_mismatch_result_element_type_i8_input(%arg0: !spirv.arm.tensor<1x65535x3x1xi8>, %arg1: !spirv.arm.tensor<7x1x1x1xi8>, %arg2: !spirv.arm.tensor<1xi16>) -> (!spirv.arm.tensor<1x65536x2x7xi16>) {
  %5 = spirv.Constant dense<35> : !spirv.arm.tensor<1xi8>
  %6 = spirv.Constant dense<57> : !spirv.arm.tensor<1xi8>
  // expected-error @+1 {{op failed to verify that if input has type 8-bit signless integer then output must have a type in [32-bit signless integer]}}
  %7 = spirv.Tosa.Conv2D pad = [1, 0, 0, 0], stride = [1, 2], dilation = [7, 1], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x65535x3x1xi8>, !spirv.arm.tensor<7x1x1x1xi8>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x65536x2x7xi16>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x65536x2x7xi16>
}

spirv.ARM.Graph @conv2d_mismatch_result_element_type_i16_input(%arg0: !spirv.arm.tensor<1x65535x3x1xi16>, %arg1: !spirv.arm.tensor<7x1x1x1xi16>, %arg2: !spirv.arm.tensor<1xi32>) -> (!spirv.arm.tensor<1x65536x2x7xi32>) {
  %5 = spirv.Constant dense<35> : !spirv.arm.tensor<1xi16>
  %6 = spirv.Constant dense<57> : !spirv.arm.tensor<1xi16>
  // expected-error @+1 {{op failed to verify that if input has type 16-bit signless integer then output must have a type in [64-bit signless integer]}}
  %7 = spirv.Tosa.Conv2D pad = [1, 0, 0, 0], stride = [1, 2], dilation = [7, 1], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x65535x3x1xi16>, !spirv.arm.tensor<7x1x1x1xi16>, !spirv.arm.tensor<1xi32>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi16> -> !spirv.arm.tensor<1x65536x2x7xi32>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x65536x2x7xi32>
}

spirv.ARM.Graph @conv2d_mismatch_result_element_type_f16_input(%arg0: !spirv.arm.tensor<1x34x18x27xf16>, %arg1: !spirv.arm.tensor<11x1x1x27xf16>, %arg2: !spirv.arm.tensor<11xf32>) -> (!spirv.arm.tensor<1x34x18x11xf32>) {
  %5 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf16>
  %6 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf16>
  // expected-error @+1 {{op failed to verify that if input has type 16-bit float then output must have a type in [16-bit float]}}
  %7 = spirv.Tosa.Conv2D pad = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1], acc_type = <FP16>, local_bound = true, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x34x18x27xf16>, !spirv.arm.tensor<11x1x1x27xf16>, !spirv.arm.tensor<11xf32>, !spirv.arm.tensor<1xf16>, !spirv.arm.tensor<1xf16> -> !spirv.arm.tensor<1x34x18x11xf32>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x34x18x11xf32>
}

spirv.ARM.Graph @conv2d_mismatch_result_element_type_f32_input(%arg0: !spirv.arm.tensor<1x34x18x27xf32>, %arg1: !spirv.arm.tensor<11x1x1x27xf32>, %arg2: !spirv.arm.tensor<11xf16>) -> (!spirv.arm.tensor<1x34x18x11xf16>) {
  %5 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf32>
  %6 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf32>
  // expected-error @+1 {{op failed to verify that if input has type 32-bit float then output must have a type in [32-bit float]}}
  %7 = spirv.Tosa.Conv2D pad = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1], acc_type = <FP16>, local_bound = true, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x34x18x27xf32>, !spirv.arm.tensor<11x1x1x27xf32>, !spirv.arm.tensor<11xf16>, !spirv.arm.tensor<1xf32>, !spirv.arm.tensor<1xf32> -> !spirv.arm.tensor<1x34x18x11xf16>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x34x18x11xf16>
}

spirv.ARM.Graph @conv2d_bias_element_type_must_be_same_as_result_element_type(%arg0: !spirv.arm.tensor<1x65535x3x1xi8>, %arg1: !spirv.arm.tensor<7x1x1x1xi8>, %arg2: !spirv.arm.tensor<1xi16>) -> (!spirv.arm.tensor<1x65536x2x7xi32>) {
  %5 = spirv.Constant dense<35> : !spirv.arm.tensor<1xi8>
  %6 = spirv.Constant dense<57> : !spirv.arm.tensor<1xi8>
  // expected-error @+1 {{op failed to verify that all of {bias, output} have same element type}}
  %7 = spirv.Tosa.Conv2D pad = [1, 0, 0, 0], stride = [1, 2], dilation = [7, 1], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x65535x3x1xi8>, !spirv.arm.tensor<7x1x1x1xi8>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x65536x2x7xi32>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x65536x2x7xi32>
}

spirv.ARM.Graph @conv2d_accumulator_must_be_INT32_for_i8_input_element_type(%arg0: !spirv.arm.tensor<1x65535x3x1xi8>, %arg1: !spirv.arm.tensor<7x1x1x1xi8>, %arg2: !spirv.arm.tensor<1xi32>) -> (!spirv.arm.tensor<1x65536x2x7xi32>) {
  %5 = spirv.Constant dense<35> : !spirv.arm.tensor<1xi8>
  %6 = spirv.Constant dense<57> : !spirv.arm.tensor<1xi8>
  // expected-error @+1 {{op failed to verify that acc_type must be one in [INT32] when type has value 8-bit signless integer}}
  %7 = spirv.Tosa.Conv2D pad = [1, 0, 0, 0], stride = [1, 2], dilation = [7, 1], acc_type = <INT48>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x65535x3x1xi8>, !spirv.arm.tensor<7x1x1x1xi8>, !spirv.arm.tensor<1xi32>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x65536x2x7xi32>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x65536x2x7xi32>
}

spirv.ARM.Graph @conv2d_accumulator_must_be_INT48_for_i16_input_element_type(%arg0: !spirv.arm.tensor<1x65535x3x1xi16>, %arg1: !spirv.arm.tensor<7x1x1x1xi8>, %arg2: !spirv.arm.tensor<1xi64>) -> (!spirv.arm.tensor<1x65536x2x7xi64>) {
  %5 = spirv.Constant dense<35> : !spirv.arm.tensor<1xi16>
  %6 = spirv.Constant dense<57> : !spirv.arm.tensor<1xi8>
  // expected-error @+1 {{op failed to verify that acc_type must be one in [INT48] when type has value 16-bit signless integer}}
  %7 = spirv.Tosa.Conv2D pad = [1, 0, 0, 0], stride = [1, 2], dilation = [7, 1], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x65535x3x1xi16>, !spirv.arm.tensor<7x1x1x1xi8>, !spirv.arm.tensor<1xi64>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x65536x2x7xi64>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x65536x2x7xi64>
}

spirv.ARM.Graph @conv2d_accumulator_must_be_either_FP16_or_FP32_for_f16_input_element_type(%arg0: !spirv.arm.tensor<1x34x18x27xf16>, %arg1: !spirv.arm.tensor<11x1x1x27xf16>, %arg2: !spirv.arm.tensor<11xf16>) -> (!spirv.arm.tensor<1x34x18x11xf16>) {
  %5 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf16>
  %6 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf16>
  // expected-error @+1 {{op failed to verify that acc_type must be one in [FP16,FP32] when type has value 16-bit float}}
  %7 = spirv.Tosa.Conv2D pad = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1], acc_type = <INT32>, local_bound = true, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x34x18x27xf16>, !spirv.arm.tensor<11x1x1x27xf16>, !spirv.arm.tensor<11xf16>, !spirv.arm.tensor<1xf16>, !spirv.arm.tensor<1xf16> -> !spirv.arm.tensor<1x34x18x11xf16>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x34x18x11xf16>
}

spirv.ARM.Graph @conv2d_accumulator_must_be_either_FP32_for_f32_input_element_type(%arg0: !spirv.arm.tensor<1x34x18x27xf32>, %arg1: !spirv.arm.tensor<11x1x1x27xf32>, %arg2: !spirv.arm.tensor<11xf32>) -> (!spirv.arm.tensor<1x34x18x11xf32>) {
  %5 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf32>
  %6 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf32>
  // expected-error @+1 {{op failed to verify that acc_type must be one in [FP32] when type has value 32-bit float}}
  %7 = spirv.Tosa.Conv2D pad = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1], acc_type = <FP16>, local_bound = true, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x34x18x27xf32>, !spirv.arm.tensor<11x1x1x27xf32>, !spirv.arm.tensor<11xf32>, !spirv.arm.tensor<1xf32>, !spirv.arm.tensor<1xf32> -> !spirv.arm.tensor<1x34x18x11xf32>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x34x18x11xf32>
}


//===----------------------------------------------------------------------===//
// spirv.TOSA.Conv3D
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @conv3d_wrong_input_integer_element_type(%arg0: !spirv.arm.tensor<1x65535x3x1x1xi32>, %arg1: !spirv.arm.tensor<7x1x1x1x1xi32>, %arg2: !spirv.arm.tensor<1xi64>) -> (!spirv.arm.tensor<1x65536x2x7x1xi64>) {
  %5 = spirv.Constant dense<35> : !spirv.arm.tensor<1xi32>
  %6 = spirv.Constant dense<57> : !spirv.arm.tensor<1xi32>
  // expected-error @+1 {{ op failed to verify that if input has type integer then input must have a type in [8-bit signless integer,16-bit signless integer]}}
  %7 = spirv.Tosa.Conv3D pad = [1, 0, 0, 0, 0, 0], stride = [1, 2, 3], dilation = [7, 1, 1], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x65535x3x1x1xi32>, !spirv.arm.tensor<7x1x1x1x1xi32>, !spirv.arm.tensor<1xi64>, !spirv.arm.tensor<1xi32>, !spirv.arm.tensor<1xi32> -> !spirv.arm.tensor<1x65536x2x7x1xi64>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x65536x2x7x1xi64>
}

spirv.ARM.Graph @conv3d_mismatch_result_element_type_i8_input(%arg0: !spirv.arm.tensor<1x65535x3x1x1xi8>, %arg1: !spirv.arm.tensor<7x1x1x1x1xi8>, %arg2: !spirv.arm.tensor<1xi16>) -> (!spirv.arm.tensor<1x65536x2x7x1xi16>) {
  %5 = spirv.Constant dense<35> : !spirv.arm.tensor<1xi8>
  %6 = spirv.Constant dense<57> : !spirv.arm.tensor<1xi8>
  // expected-error @+1 {{op failed to verify that if input has type 8-bit signless integer then output must have a type in [32-bit signless integer]}}
  %7 = spirv.Tosa.Conv3D pad = [1, 0, 0, 0, 0, 0], stride = [1, 2, 3], dilation = [7, 1, 1], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x65535x3x1x1xi8>, !spirv.arm.tensor<7x1x1x1x1xi8>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x65536x2x7x1xi16>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x65536x2x7x1xi16>
}

spirv.ARM.Graph @conv3d_mismatch_result_element_type_i16_input(%arg0: !spirv.arm.tensor<1x65535x3x1x1xi16>, %arg1: !spirv.arm.tensor<7x1x1x1x1xi8>, %arg2: !spirv.arm.tensor<1xi32>) -> (!spirv.arm.tensor<1x65536x2x7x1xi32>) {
  %5 = spirv.Constant dense<35> : !spirv.arm.tensor<1xi16>
  %6 = spirv.Constant dense<57> : !spirv.arm.tensor<1xi8>
  // expected-error @+1 {{op failed to verify that if input has type 16-bit signless integer then output must have a type in [64-bit signless integer]}}
  %7 = spirv.Tosa.Conv3D pad = [1, 0, 0, 0, 0, 0], stride = [1, 2, 3], dilation = [7, 1, 1], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x65535x3x1x1xi16>, !spirv.arm.tensor<7x1x1x1x1xi8>, !spirv.arm.tensor<1xi32>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x65536x2x7x1xi32>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x65536x2x7x1xi32>
}

spirv.ARM.Graph @conv3d_mismatch_result_element_type_f16_input(%arg0: !spirv.arm.tensor<1x34x18x27x1xf16>, %arg1: !spirv.arm.tensor<11x1x1x27x1xf16>, %arg2: !spirv.arm.tensor<11xf32>) -> (!spirv.arm.tensor<1x34x18x11x1xf32>) {
  %5 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf16>
  %6 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf16>
  // expected-error @+1 {{op failed to verify that if input has type 16-bit float then output must have a type in [16-bit float]}}
  %7 = spirv.Tosa.Conv3D pad = [0, 0, 0, 0, 0, 0], stride = [1, 1, 1], dilation = [1, 1, 1], acc_type = <FP16>, local_bound = true, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x34x18x27x1xf16>, !spirv.arm.tensor<11x1x1x27x1xf16>, !spirv.arm.tensor<11xf32>, !spirv.arm.tensor<1xf16>, !spirv.arm.tensor<1xf16> -> !spirv.arm.tensor<1x34x18x11x1xf32>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x34x18x11x1xf32>
}

spirv.ARM.Graph @conv3d_mismatch_result_element_type_f32_input(%arg0: !spirv.arm.tensor<1x34x18x27x1xf32>, %arg1: !spirv.arm.tensor<11x1x1x27x1xf32>, %arg2: !spirv.arm.tensor<11xf16>) -> (!spirv.arm.tensor<1x34x18x11x1xf16>) {
  %5 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf32>
  %6 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf32>
  // expected-error @+1 {{op failed to verify that if input has type 32-bit float then output must have a type in [32-bit float]}}
  %7 = spirv.Tosa.Conv3D pad = [0, 0, 0, 0, 0, 0], stride = [1, 1, 1], dilation = [1, 1, 1], acc_type = <FP16>, local_bound = true, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x34x18x27x1xf32>, !spirv.arm.tensor<11x1x1x27x1xf32>, !spirv.arm.tensor<11xf16>, !spirv.arm.tensor<1xf32>, !spirv.arm.tensor<1xf32> -> !spirv.arm.tensor<1x34x18x11x1xf16>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x34x18x11x1xf16>
}

spirv.ARM.Graph @conv3d_bias_element_type_must_be_same_as_result_element_type(%arg0: !spirv.arm.tensor<1x65535x3x1x1xi8>, %arg1: !spirv.arm.tensor<7x1x1x1x1xi8>, %arg2: !spirv.arm.tensor<1xi16>) -> (!spirv.arm.tensor<1x65536x2x7x1xi32>) {
  %5 = spirv.Constant dense<35> : !spirv.arm.tensor<1xi8>
  %6 = spirv.Constant dense<57> : !spirv.arm.tensor<1xi8>
  // expected-error @+1 {{op failed to verify that all of {bias, output} have same element type}}
  %7 = spirv.Tosa.Conv3D pad = [1, 0, 0, 0, 0, 0], stride = [1, 2, 3], dilation = [7, 1, 1], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x65535x3x1x1xi8>, !spirv.arm.tensor<7x1x1x1x1xi8>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x65536x2x7x1xi32>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x65536x2x7x1xi32>
}

spirv.ARM.Graph @conv3d_accumulator_must_be_INT32_for_i8_input_element_type(%arg0: !spirv.arm.tensor<1x65535x3x1x1xi8>, %arg1: !spirv.arm.tensor<7x1x1x1x1xi8>, %arg2: !spirv.arm.tensor<1xi32>) -> (!spirv.arm.tensor<1x65536x2x7x1xi32>) {
  %5 = spirv.Constant dense<35> : !spirv.arm.tensor<1xi8>
  %6 = spirv.Constant dense<57> : !spirv.arm.tensor<1xi8>
  // expected-error @+1 {{op failed to verify that acc_type must be one in [INT32] when type has value 8-bit signless integer}}
  %7 = spirv.Tosa.Conv3D pad = [1, 0, 0, 0, 0, 0], stride = [1, 2, 3], dilation = [7, 1, 1], acc_type = <INT48>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x65535x3x1x1xi8>, !spirv.arm.tensor<7x1x1x1x1xi8>, !spirv.arm.tensor<1xi32>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x65536x2x7x1xi32>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x65536x2x7x1xi32>
}

spirv.ARM.Graph @conv3d_accumulator_must_be_INT48_for_i16_input_element_type(%arg0: !spirv.arm.tensor<1x65535x3x1x1xi16>, %arg1: !spirv.arm.tensor<7x1x1x1x1xi8>, %arg2: !spirv.arm.tensor<1xi64>) -> (!spirv.arm.tensor<1x65536x2x7x1xi64>) {
  %5 = spirv.Constant dense<35> : !spirv.arm.tensor<1xi16>
  %6 = spirv.Constant dense<57> : !spirv.arm.tensor<1xi8>
  // expected-error @+1 {{op failed to verify that acc_type must be one in [INT48] when type has value 16-bit signless integer}}
  %7 = spirv.Tosa.Conv3D pad = [1, 0, 0, 0, 0, 0], stride = [1, 2, 3], dilation = [7, 1, 1], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x65535x3x1x1xi16>, !spirv.arm.tensor<7x1x1x1x1xi8>, !spirv.arm.tensor<1xi64>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x65536x2x7x1xi64>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x65536x2x7x1xi64>
}

spirv.ARM.Graph @conv3d_accumulator_must_be_either_FP16_or_FP32_for_f16_input_element_type(%arg0: !spirv.arm.tensor<1x34x18x27x1xf16>, %arg1: !spirv.arm.tensor<11x1x1x27x1xf16>, %arg2: !spirv.arm.tensor<11xf16>) -> (!spirv.arm.tensor<1x34x18x11x1xf16>) {
  %5 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf16>
  %6 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf16>
  // expected-error @+1 {{op failed to verify that acc_type must be one in [FP16,FP32] when type has value 16-bit float}}
  %7 = spirv.Tosa.Conv3D pad = [0, 0, 0, 0, 0, 0], stride = [1, 1, 1], dilation = [1, 1, 1], acc_type = <INT32>, local_bound = true, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x34x18x27x1xf16>, !spirv.arm.tensor<11x1x1x27x1xf16>, !spirv.arm.tensor<11xf16>, !spirv.arm.tensor<1xf16>, !spirv.arm.tensor<1xf16> -> !spirv.arm.tensor<1x34x18x11x1xf16>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x34x18x11x1xf16>
}

spirv.ARM.Graph @conv3d_accumulator_must_be_either_FP32_for_f32_input_element_type(%arg0: !spirv.arm.tensor<1x34x18x27x1xf32>, %arg1: !spirv.arm.tensor<11x1x1x27x1xf32>, %arg2: !spirv.arm.tensor<11xf32>) -> (!spirv.arm.tensor<1x34x18x11x1xf32>) {
  %5 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf32>
  %6 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf32>
  // expected-error @+1 {{op failed to verify that acc_type must be one in [FP32] when type has value 32-bit float}}
  %7 = spirv.Tosa.Conv3D pad = [0, 0, 0, 0, 0, 0], stride = [1, 1, 1], dilation = [1, 1, 1], acc_type = <FP16>, local_bound = true, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x34x18x27x1xf32>, !spirv.arm.tensor<11x1x1x27x1xf32>, !spirv.arm.tensor<11xf32>, !spirv.arm.tensor<1xf32>, !spirv.arm.tensor<1xf32> -> !spirv.arm.tensor<1x34x18x11x1xf32>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x34x18x11x1xf32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.DepthwiseConv2D
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @depthwise_conv2d_wrong_input_integer_element_type(%arg0: !spirv.arm.tensor<1x65535x3x1xi32>, %arg1: !spirv.arm.tensor<7x1x1x1xi32>, %arg2: !spirv.arm.tensor<1xi64>) -> (!spirv.arm.tensor<1x65536x2x7xi64>) {
  %5 = spirv.Constant dense<35> : !spirv.arm.tensor<1xi32>
  %6 = spirv.Constant dense<57> : !spirv.arm.tensor<1xi32>
  // expected-error @+1 {{op failed to verify that if input has type integer then input must have a type in [8-bit signless integer,16-bit signless integer]}}
  %7 = spirv.Tosa.DepthwiseConv2D pad = [1, 0, 0, 0], stride = [1, 2], dilation = [7, 1], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x65535x3x1xi32>, !spirv.arm.tensor<7x1x1x1xi32>, !spirv.arm.tensor<1xi64>, !spirv.arm.tensor<1xi32>, !spirv.arm.tensor<1xi32> -> !spirv.arm.tensor<1x65536x2x7xi64>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x65536x2x7xi64>
}

spirv.ARM.Graph @depthwise_conv2d_mismatch_result_element_type_i8_input(%arg0: !spirv.arm.tensor<1x65535x3x1xi8>, %arg1: !spirv.arm.tensor<7x1x1x1xi8>, %arg2: !spirv.arm.tensor<1xi16>) -> (!spirv.arm.tensor<1x65536x2x7xi16>) {
  %5 = spirv.Constant dense<35> : !spirv.arm.tensor<1xi8>
  %6 = spirv.Constant dense<57> : !spirv.arm.tensor<1xi8>
  // expected-error @+1 {{op failed to verify that if input has type 8-bit signless integer then output must have a type in [32-bit signless integer]}}
  %7 = spirv.Tosa.DepthwiseConv2D pad = [1, 0, 0, 0], stride = [1, 2], dilation = [7, 1], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x65535x3x1xi8>, !spirv.arm.tensor<7x1x1x1xi8>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x65536x2x7xi16>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x65536x2x7xi16>
}

spirv.ARM.Graph @depthwise_conv2d_mismatch_result_element_type_i16_input(%arg0: !spirv.arm.tensor<1x65535x3x1xi16>, %arg1: !spirv.arm.tensor<7x1x1x1xi8>, %arg2: !spirv.arm.tensor<1xi32>) -> (!spirv.arm.tensor<1x65536x2x7xi32>) {
  %5 = spirv.Constant dense<35> : !spirv.arm.tensor<1xi16>
  %6 = spirv.Constant dense<57> : !spirv.arm.tensor<1xi8>
  // expected-error @+1 {{op failed to verify that if input has type 16-bit signless integer then output must have a type in [64-bit signless integer]}}
  %7 = spirv.Tosa.DepthwiseConv2D pad = [1, 0, 0, 0], stride = [1, 2], dilation = [7, 1], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x65535x3x1xi16>, !spirv.arm.tensor<7x1x1x1xi8>, !spirv.arm.tensor<1xi32>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x65536x2x7xi32>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x65536x2x7xi32>
}

spirv.ARM.Graph @depthwise_conv2d_mismatch_result_element_type_f16_input(%arg0: !spirv.arm.tensor<1x34x18x27xf16>, %arg1: !spirv.arm.tensor<11x1x1x27xf16>, %arg2: !spirv.arm.tensor<11xf32>) -> (!spirv.arm.tensor<1x34x18x11xf32>) {
  %5 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf16>
  %6 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf16>
  // expected-error @+1 {{op failed to verify that if input has type 16-bit float then output must have a type in [16-bit float]}}
  %7 = spirv.Tosa.DepthwiseConv2D pad = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1], acc_type = <FP16>, local_bound = true, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x34x18x27xf16>, !spirv.arm.tensor<11x1x1x27xf16>, !spirv.arm.tensor<11xf32>, !spirv.arm.tensor<1xf16>, !spirv.arm.tensor<1xf16> -> !spirv.arm.tensor<1x34x18x11xf32>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x34x18x11xf32>
}

spirv.ARM.Graph @depthwise_conv2d_mismatch_result_element_type_f32_input(%arg0: !spirv.arm.tensor<1x34x18x27xf32>, %arg1: !spirv.arm.tensor<11x1x1x27xf32>, %arg2: !spirv.arm.tensor<11xf16>) -> (!spirv.arm.tensor<1x34x18x11xf16>) {
  %5 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf32>
  %6 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf32>
  // expected-error @+1 {{op failed to verify that if input has type 32-bit float then output must have a type in [32-bit float]}}
  %7 = spirv.Tosa.DepthwiseConv2D pad = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1], acc_type = <FP16>, local_bound = true, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x34x18x27xf32>, !spirv.arm.tensor<11x1x1x27xf32>, !spirv.arm.tensor<11xf16>, !spirv.arm.tensor<1xf32>, !spirv.arm.tensor<1xf32> -> !spirv.arm.tensor<1x34x18x11xf16>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x34x18x11xf16>
}

spirv.ARM.Graph @depthwise_conv2d_bias_element_type_must_be_same_as_result_element_type(%arg0: !spirv.arm.tensor<1x65535x3x1xi8>, %arg1: !spirv.arm.tensor<7x1x1x1xi8>, %arg2: !spirv.arm.tensor<1xi16>) -> (!spirv.arm.tensor<1x65536x2x7xi32>) {
  %5 = spirv.Constant dense<35> : !spirv.arm.tensor<1xi8>
  %6 = spirv.Constant dense<57> : !spirv.arm.tensor<1xi8>
  // expected-error @+1 {{op failed to verify that all of {bias, output} have same element type}}
  %7 = spirv.Tosa.DepthwiseConv2D pad = [1, 0, 0, 0], stride = [1, 2], dilation = [7, 1], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x65535x3x1xi8>, !spirv.arm.tensor<7x1x1x1xi8>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x65536x2x7xi32>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x65536x2x7xi32>
}

spirv.ARM.Graph @depthwise_conv2d_accumulator_must_be_INT32_for_i8_input_element_type(%arg0: !spirv.arm.tensor<1x65535x3x1xi8>, %arg1: !spirv.arm.tensor<7x1x1x1xi8>, %arg2: !spirv.arm.tensor<1xi32>) -> (!spirv.arm.tensor<1x65536x2x7xi32>) {
  %5 = spirv.Constant dense<35> : !spirv.arm.tensor<1xi8>
  %6 = spirv.Constant dense<57> : !spirv.arm.tensor<1xi8>
  // expected-error @+1 {{op failed to verify that acc_type must be one in [INT32] when type has value 8-bit signless integer}}
  %7 = spirv.Tosa.DepthwiseConv2D pad = [1, 0, 0, 0], stride = [1, 2], dilation = [7, 1], acc_type = <INT48>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x65535x3x1xi8>, !spirv.arm.tensor<7x1x1x1xi8>, !spirv.arm.tensor<1xi32>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x65536x2x7xi32>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x65536x2x7xi32>
}

spirv.ARM.Graph @depthwise_conv2d_accumulator_must_be_INT48_for_i16_input_element_type(%arg0: !spirv.arm.tensor<1x65535x3x1xi16>, %arg1: !spirv.arm.tensor<7x1x1x1xi8>, %arg2: !spirv.arm.tensor<1xi64>) -> (!spirv.arm.tensor<1x65536x2x7xi64>) {
  %5 = spirv.Constant dense<35> : !spirv.arm.tensor<1xi16>
  %6 = spirv.Constant dense<57> : !spirv.arm.tensor<1xi8>
  // expected-error @+1 {{op failed to verify that acc_type must be one in [INT48] when type has value 16-bit signless integer}}
  %7 = spirv.Tosa.DepthwiseConv2D pad = [1, 0, 0, 0], stride = [1, 2], dilation = [7, 1], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x65535x3x1xi16>, !spirv.arm.tensor<7x1x1x1xi8>, !spirv.arm.tensor<1xi64>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x65536x2x7xi64>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x65536x2x7xi64>
}

spirv.ARM.Graph @depthwise_conv2d_accumulator_must_be_either_FP16_or_FP32_for_f16_input_element_type(%arg0: !spirv.arm.tensor<1x34x18x27xf16>, %arg1: !spirv.arm.tensor<11x1x1x27xf16>, %arg2: !spirv.arm.tensor<11xf16>) -> (!spirv.arm.tensor<1x34x18x11xf16>) {
  %5 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf16>
  %6 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf16>
  // expected-error @+1 {{op failed to verify that acc_type must be one in [FP16,FP32] when type has value 16-bit float}}
  %7 = spirv.Tosa.DepthwiseConv2D pad = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1], acc_type = <INT32>, local_bound = true, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x34x18x27xf16>, !spirv.arm.tensor<11x1x1x27xf16>, !spirv.arm.tensor<11xf16>, !spirv.arm.tensor<1xf16>, !spirv.arm.tensor<1xf16> -> !spirv.arm.tensor<1x34x18x11xf16>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x34x18x11xf16>
}

spirv.ARM.Graph @depthwise_conv2d_accumulator_must_be_either_FP32_for_f32_input_element_type(%arg0: !spirv.arm.tensor<1x34x18x27xf32>, %arg1: !spirv.arm.tensor<11x1x1x27xf32>, %arg2: !spirv.arm.tensor<11xf32>) -> (!spirv.arm.tensor<1x34x18x11xf32>) {
  %5 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf32>
  %6 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf32>
  // expected-error @+1 {{op failed to verify that acc_type must be one in [FP32] when type has value 32-bit float}}
  %7 = spirv.Tosa.DepthwiseConv2D pad = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1], acc_type = <FP16>, local_bound = true, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x34x18x27xf32>, !spirv.arm.tensor<11x1x1x27xf32>, !spirv.arm.tensor<11xf32>, !spirv.arm.tensor<1xf32>, !spirv.arm.tensor<1xf32> -> !spirv.arm.tensor<1x34x18x11xf32>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x34x18x11xf32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.TransposeConv2D
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @transpose_conv2d_wrong_input_integer_element_type(%arg0: !spirv.arm.tensor<1x65535x3x1xi32>, %arg1: !spirv.arm.tensor<7x1x1x1xi32>, %arg2: !spirv.arm.tensor<1xi64>) -> (!spirv.arm.tensor<1x65536x2x7xi64>) {
  %5 = spirv.Constant dense<35> : !spirv.arm.tensor<1xi32>
  %6 = spirv.Constant dense<57> : !spirv.arm.tensor<1xi32>
  // expected-error @+1 {{op failed to verify that if input has type integer then input must have a type in [8-bit signless integer,16-bit signless integer]}}
  %7 = spirv.Tosa.TransposeConv2D out_pad = [1, 0, 0, 0], stride = [1, 2], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x65535x3x1xi32>, !spirv.arm.tensor<7x1x1x1xi32>, !spirv.arm.tensor<1xi64>, !spirv.arm.tensor<1xi32>, !spirv.arm.tensor<1xi32> -> !spirv.arm.tensor<1x65536x2x7xi64>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x65536x2x7xi64>
}

spirv.ARM.Graph @transpose_conv2d_mismatch_result_element_type_i8_input(%arg0: !spirv.arm.tensor<1x65535x3x1xi8>, %arg1: !spirv.arm.tensor<7x1x1x1xi8>, %arg2: !spirv.arm.tensor<1xi16>) -> (!spirv.arm.tensor<1x65536x2x7xi16>) {
  %5 = spirv.Constant dense<35> : !spirv.arm.tensor<1xi8>
  %6 = spirv.Constant dense<57> : !spirv.arm.tensor<1xi8>
  // expected-error @+1 {{op failed to verify that if input has type 8-bit signless integer then output must have a type in [32-bit signless integer]}}
  %7 = spirv.Tosa.TransposeConv2D out_pad = [1, 0, 0, 0], stride = [1, 2], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x65535x3x1xi8>, !spirv.arm.tensor<7x1x1x1xi8>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x65536x2x7xi16>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x65536x2x7xi16>
}

spirv.ARM.Graph @transpose_conv2d_mismatch_result_element_type_i16_input(%arg0: !spirv.arm.tensor<1x65535x3x1xi16>, %arg1: !spirv.arm.tensor<7x1x1x1xi16>, %arg2: !spirv.arm.tensor<1xi32>) -> (!spirv.arm.tensor<1x65536x2x7xi32>) {
  %5 = spirv.Constant dense<35> : !spirv.arm.tensor<1xi16>
  %6 = spirv.Constant dense<57> : !spirv.arm.tensor<1xi16>
  // expected-error @+1 {{op failed to verify that if input has type 16-bit signless integer then output must have a type in [64-bit signless integer]}}
  %7 = spirv.Tosa.TransposeConv2D out_pad = [1, 0, 0, 0], stride = [1, 2], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x65535x3x1xi16>, !spirv.arm.tensor<7x1x1x1xi16>, !spirv.arm.tensor<1xi32>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi16> -> !spirv.arm.tensor<1x65536x2x7xi32>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x65536x2x7xi32>
}

spirv.ARM.Graph @transpose_conv2d_mismatch_result_element_type_f16_input(%arg0: !spirv.arm.tensor<1x34x18x27xf16>, %arg1: !spirv.arm.tensor<11x1x1x27xf16>, %arg2: !spirv.arm.tensor<11xf32>) -> (!spirv.arm.tensor<1x34x18x11xf32>) {
  %5 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf16>
  %6 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf16>
  // expected-error @+1 {{op failed to verify that if input has type 16-bit float then output must have a type in [16-bit float]}}
  %7 = spirv.Tosa.TransposeConv2D out_pad = [0, 0, 0, 0], stride = [1, 1], acc_type = <FP16>, local_bound = true, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x34x18x27xf16>, !spirv.arm.tensor<11x1x1x27xf16>, !spirv.arm.tensor<11xf32>, !spirv.arm.tensor<1xf16>, !spirv.arm.tensor<1xf16> -> !spirv.arm.tensor<1x34x18x11xf32>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x34x18x11xf32>
}

spirv.ARM.Graph @transpose_conv2d_mismatch_result_element_type_f32_input(%arg0: !spirv.arm.tensor<1x34x18x27xf32>, %arg1: !spirv.arm.tensor<11x1x1x27xf32>, %arg2: !spirv.arm.tensor<11xf16>) -> (!spirv.arm.tensor<1x34x18x11xf16>) {
  %5 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf32>
  %6 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf32>
  // expected-error @+1 {{op failed to verify that if input has type 32-bit float then output must have a type in [32-bit float]}}
  %7 = spirv.Tosa.TransposeConv2D out_pad = [0, 0, 0, 0], stride = [1, 1], acc_type = <FP16>, local_bound = true, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x34x18x27xf32>, !spirv.arm.tensor<11x1x1x27xf32>, !spirv.arm.tensor<11xf16>, !spirv.arm.tensor<1xf32>, !spirv.arm.tensor<1xf32> -> !spirv.arm.tensor<1x34x18x11xf16>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x34x18x11xf16>
}

spirv.ARM.Graph @transpose_conv2d_bias_element_type_must_be_same_as_result_element_type(%arg0: !spirv.arm.tensor<1x65535x3x1xi8>, %arg1: !spirv.arm.tensor<7x1x1x1xi8>, %arg2: !spirv.arm.tensor<1xi16>) -> (!spirv.arm.tensor<1x65536x2x7xi32>) {
  %5 = spirv.Constant dense<35> : !spirv.arm.tensor<1xi8>
  %6 = spirv.Constant dense<57> : !spirv.arm.tensor<1xi8>
  // expected-error @+1 {{op failed to verify that all of {bias, output} have same element type}}
  %7 = spirv.Tosa.TransposeConv2D out_pad = [1, 0, 0, 0], stride = [1, 2], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x65535x3x1xi8>, !spirv.arm.tensor<7x1x1x1xi8>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x65536x2x7xi32>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x65536x2x7xi32>
}

spirv.ARM.Graph @transpose_conv2d_accumulator_must_be_INT32_for_i8_input_element_type(%arg0: !spirv.arm.tensor<1x65535x3x1xi8>, %arg1: !spirv.arm.tensor<7x1x1x1xi8>, %arg2: !spirv.arm.tensor<1xi32>) -> (!spirv.arm.tensor<1x65536x2x7xi32>) {
  %5 = spirv.Constant dense<35> : !spirv.arm.tensor<1xi8>
  %6 = spirv.Constant dense<57> : !spirv.arm.tensor<1xi8>
  // expected-error @+1 {{op failed to verify that acc_type must be one in [INT32] when type has value 8-bit signless integer}}
  %7 = spirv.Tosa.TransposeConv2D out_pad = [1, 0, 0, 0], stride = [1, 2], acc_type = <INT48>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x65535x3x1xi8>, !spirv.arm.tensor<7x1x1x1xi8>, !spirv.arm.tensor<1xi32>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x65536x2x7xi32>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x65536x2x7xi32>
}

spirv.ARM.Graph @transpose_conv2d_accumulator_must_be_INT48_for_i16_input_element_type(%arg0: !spirv.arm.tensor<1x65535x3x1xi16>, %arg1: !spirv.arm.tensor<7x1x1x1xi8>, %arg2: !spirv.arm.tensor<1xi64>) -> (!spirv.arm.tensor<1x65536x2x7xi64>) {
  %5 = spirv.Constant dense<35> : !spirv.arm.tensor<1xi16>
  %6 = spirv.Constant dense<57> : !spirv.arm.tensor<1xi8>
  // expected-error @+1 {{op failed to verify that acc_type must be one in [INT48] when type has value 16-bit signless integer}}
  %7 = spirv.Tosa.TransposeConv2D out_pad = [1, 0, 0, 0], stride = [1, 2], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x65535x3x1xi16>, !spirv.arm.tensor<7x1x1x1xi8>, !spirv.arm.tensor<1xi64>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x65536x2x7xi64>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x65536x2x7xi64>
}

spirv.ARM.Graph @transpose_conv2d_accumulator_must_be_either_FP16_or_FP32_for_f16_input_element_type(%arg0: !spirv.arm.tensor<1x34x18x27xf16>, %arg1: !spirv.arm.tensor<11x1x1x27xf16>, %arg2: !spirv.arm.tensor<11xf16>) -> (!spirv.arm.tensor<1x34x18x11xf16>) {
  %5 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf16>
  %6 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf16>
  // expected-error @+1 {{op failed to verify that acc_type must be one in [FP16,FP32] when type has value 16-bit float}}
  %7 = spirv.Tosa.TransposeConv2D out_pad = [0, 0, 0, 0], stride = [1, 1], acc_type = <INT32>, local_bound = true, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x34x18x27xf16>, !spirv.arm.tensor<11x1x1x27xf16>, !spirv.arm.tensor<11xf16>, !spirv.arm.tensor<1xf16>, !spirv.arm.tensor<1xf16> -> !spirv.arm.tensor<1x34x18x11xf16>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x34x18x11xf16>
}

spirv.ARM.Graph @transpose_conv2d_accumulator_must_be_either_FP32_for_f32_input_element_type(%arg0: !spirv.arm.tensor<1x34x18x27xf32>, %arg1: !spirv.arm.tensor<11x1x1x27xf32>, %arg2: !spirv.arm.tensor<11xf32>) -> (!spirv.arm.tensor<1x34x18x11xf32>) {
  %5 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf32>
  %6 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf32>
  // expected-error @+1 {{op failed to verify that acc_type must be one in [FP32] when type has value 32-bit float}}
  %7 = spirv.Tosa.TransposeConv2D out_pad = [0, 0, 0, 0], stride = [1, 1], acc_type = <FP16>, local_bound = true, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x34x18x27xf32>, !spirv.arm.tensor<11x1x1x27xf32>, !spirv.arm.tensor<11xf32>, !spirv.arm.tensor<1xf32>, !spirv.arm.tensor<1xf32> -> !spirv.arm.tensor<1x34x18x11xf32>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x34x18x11xf32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.AvgPool2D
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @avgpool2d_input_output_different_type(%arg0: !spirv.arm.tensor<1x3x65537x1xi8>) -> (!spirv.arm.tensor<1x2x32768x1xi16>) {
  %4 = spirv.Constant dense<125> : !spirv.arm.tensor<1xi8>
  %5 = spirv.Constant dense<-90> : !spirv.arm.tensor<1xi8>
  // expected-error @+1 {{op failed to verify that all of {input, input_zp, output, output_zp} have same element type}}
  %6 = spirv.Tosa.AvgPool2D kernel = [3, 3], stride = [1, 2], pad = [0, 1, 0, 0], acc_type = <INT32>, %arg0, %4, %5 : !spirv.arm.tensor<1x3x65537x1xi8>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x2x32768x1xi16>
  spirv.ARM.GraphOutputs %6 : !spirv.arm.tensor<1x2x32768x1xi16>
}

spirv.ARM.Graph @avgpool2d_accumulator_should_be_INT32_for_integer_element_types(%arg0: !spirv.arm.tensor<1x3x65537x1xi8>) -> (!spirv.arm.tensor<1x2x32768x1xi8>) {
  %4 = spirv.Constant dense<125> : !spirv.arm.tensor<1xi8>
  %5 = spirv.Constant dense<-90> : !spirv.arm.tensor<1xi8>
  // expected-error @+1 {{op failed to verify that acc_type must be one in [INT32] when type has value 8-bit signless integer}}
  %6 = spirv.Tosa.AvgPool2D kernel = [3, 3], stride = [1, 2], pad = [0, 1, 0, 0], acc_type = <INT48>, %arg0, %4, %5 : !spirv.arm.tensor<1x3x65537x1xi8>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x2x32768x1xi8>
  spirv.ARM.GraphOutputs %6 : !spirv.arm.tensor<1x2x32768x1xi8>
}

spirv.ARM.Graph @avgpool2d_accumulator_should_be_either_FP16_or_FP32_for_fp16_element_types(%arg0: !spirv.arm.tensor<1x2x65533x2xf16>) -> (!spirv.arm.tensor<1x2x65532x2xf16>) {
  %4 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf16>
  %5 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf16>
  // expected-error @+1 {{op failed to verify that acc_type must be one in [FP16,FP32] when type has value 16-bit float}}
  %6 = spirv.Tosa.AvgPool2D kernel = [2, 2], stride = [1, 1], pad = [1, 0, 0, 0], acc_type = <INT32>, %arg0, %4, %5 : !spirv.arm.tensor<1x2x65533x2xf16>, !spirv.arm.tensor<1xf16>, !spirv.arm.tensor<1xf16> -> !spirv.arm.tensor<1x2x65532x2xf16>
  spirv.ARM.GraphOutputs %6 : !spirv.arm.tensor<1x2x65532x2xf16>
}


spirv.ARM.Graph @avgpool2d_accumulator_should_be_either_FP32_for_fp32_element_types(%arg0: !spirv.arm.tensor<1x2x65533x2xf32>) -> (!spirv.arm.tensor<1x2x65532x2xf32>) {
  %4 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf32>
  %5 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf32>
  // expected-error @+1 {{op failed to verify that acc_type must be one in [FP32] when type has value 32-bit float}}
  %6 = spirv.Tosa.AvgPool2D kernel = [2, 2], stride = [1, 1], pad = [1, 0, 0, 0], acc_type = <FP16>, %arg0, %4, %5 : !spirv.arm.tensor<1x2x65533x2xf32>, !spirv.arm.tensor<1xf32>, !spirv.arm.tensor<1xf32> -> !spirv.arm.tensor<1x2x65532x2xf32>
  spirv.ARM.GraphOutputs %6 : !spirv.arm.tensor<1x2x65532x2xf32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.MatMul
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @matmul_invalid_input_output_element_type_combination(%arg0: !spirv.arm.tensor<1x4x4xi16>, %arg1: !spirv.arm.tensor<1x4x4xi16>, %arg2: !spirv.arm.tensor<1xi16>, %arg3: !spirv.arm.tensor<1xi16>) -> (!spirv.arm.tensor<1x4x4xi32>) {
  // expected-error @+1 {{op failed to verify that if A has type 16-bit signless integer then output must have a type in [64-bit signless integer]}}
  %0 = spirv.Tosa.MatMul %arg0, %arg1, %arg2, %arg3 : !spirv.arm.tensor<1x4x4xi16>, !spirv.arm.tensor<1x4x4xi16>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi16> -> !spirv.arm.tensor<1x4x4xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<1x4x4xi32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.MaxPool2D
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @maxpool2d_input_output_different_element_types(%arg0: !spirv.arm.tensor<1x3x65537x1xi8>) -> (!spirv.arm.tensor<1x2x32769x1xi16>) {
  // expected-error @+1 {{op failed to verify that all of {input, output} have same element type}}
  %4 = spirv.Tosa.MaxPool2D kernel = [3, 2], stride = [1, 2], pad = [1, 0, 0, 1], nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<1x3x65537x1xi8> -> !spirv.arm.tensor<1x2x32769x1xi16>
  spirv.ARM.GraphOutputs %4 : !spirv.arm.tensor<1x2x32769x1xi16>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Clamp
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @clamp_min_val_different_element_type_wrt_input_output(%arg0: !spirv.arm.tensor<27x44x55xi8>) -> (!spirv.arm.tensor<27x44x55xi8>) {
  // expected-error @+1 {{op failed to verify that all of {input, output, min_val, max_val} have same element type}}
  %3 = spirv.Tosa.Clamp min_val = -102 : i16, max_val = -100 : i8, nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<27x44x55xi8> -> !spirv.arm.tensor<27x44x55xi8>
  spirv.ARM.GraphOutputs %3 : !spirv.arm.tensor<27x44x55xi8>
}

spirv.ARM.Graph @clamp_max_val_different_element_type_wrt_input_output(%arg0: !spirv.arm.tensor<27x44x55xi8>) -> (!spirv.arm.tensor<27x44x55xi8>) {
  // expected-error @+1 {{op failed to verify that all of {input, output, min_val, max_val} have same element type}}
  %3 = spirv.Tosa.Clamp min_val = -102 : i8, max_val = -100 : i16, nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<27x44x55xi8> -> !spirv.arm.tensor<27x44x55xi8>
  spirv.ARM.GraphOutputs %3 : !spirv.arm.tensor<27x44x55xi8>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Add
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @add_input_ranks_not_matching(%arg0: !spirv.arm.tensor<6x10x6xi32>, %arg1: !spirv.arm.tensor<1x10x6x6xi32>) -> (!spirv.arm.tensor<6x10x6x6xi32>) {
  // expected-error @+1 {{op failed to verify that all of {input1, input2} have same rank}}
  %0 = spirv.Tosa.Add %arg0, %arg1 : !spirv.arm.tensor<6x10x6xi32>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi32>
}

spirv.ARM.Graph @add_input_element_types_not_matching(%arg0: !spirv.arm.tensor<6x10x6x6xi16>, %arg1: !spirv.arm.tensor<1x10x6x6xi32>) -> (!spirv.arm.tensor<6x10x6x6xi32>) {
  // expected-error @+1 {{op failed to verify that all of {input1, input2, output} have same element type}}
  %0 = spirv.Tosa.Add %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi16>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi32>
}

spirv.ARM.Graph @add_input_output_element_types_not_matching(%arg0: !spirv.arm.tensor<6x10x6x6xi32>, %arg1: !spirv.arm.tensor<1x10x6x6xi32>) -> (!spirv.arm.tensor<6x10x6x6xi16>) {
  // expected-error @+1 {{op failed to verify that all of {input1, input2, output} have same element type}}
  %0 = spirv.Tosa.Add %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi32>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi16>
}
