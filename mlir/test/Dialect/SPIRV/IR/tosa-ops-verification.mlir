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

//===----------------------------------------------------------------------===//
// spirv.TOSA.Conv2D
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @conv2d_wrong_input_integer_element_type(%arg0: !spirv.arm.tensor<1x65535x3x1xi32>, %arg1: !spirv.arm.tensor<7x1x1x1xi32>, %arg2: !spirv.arm.tensor<1xi64>) -> (!spirv.arm.tensor<1x65536x2x7xi64>) {
  %5 = spirv.Constant dense<35> : !spirv.arm.tensor<1xi32>
  %6 = spirv.Constant dense<57> : !spirv.arm.tensor<1xi32>
  // expected-error @+1 {{op input element type can only be of width 8 or 16 when integer type}}
  %7 = spirv.Tosa.Conv2D pad = [1, 0, 0, 0], stride = [1, 2], dilation = [7, 1], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x65535x3x1xi32>, !spirv.arm.tensor<7x1x1x1xi32>, !spirv.arm.tensor<1xi64>, !spirv.arm.tensor<1xi32>, !spirv.arm.tensor<1xi32> -> !spirv.arm.tensor<1x65536x2x7xi64>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x65536x2x7xi64>
}

spirv.ARM.Graph @conv2d_mismatch_result_element_type_i8_input(%arg0: !spirv.arm.tensor<1x65535x3x1xi8>, %arg1: !spirv.arm.tensor<7x1x1x1xi8>, %arg2: !spirv.arm.tensor<1xi16>) -> (!spirv.arm.tensor<1x65536x2x7xi16>) {
  %5 = spirv.Constant dense<35> : !spirv.arm.tensor<1xi8>
  %6 = spirv.Constant dense<57> : !spirv.arm.tensor<1xi8>
  // expected-error @+1 {{op expect result type to be i32, got 'i16'}}
  %7 = spirv.Tosa.Conv2D pad = [1, 0, 0, 0], stride = [1, 2], dilation = [7, 1], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x65535x3x1xi8>, !spirv.arm.tensor<7x1x1x1xi8>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x65536x2x7xi16>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x65536x2x7xi16>
}

spirv.ARM.Graph @conv2d_mismatch_result_element_type_i16_input(%arg0: !spirv.arm.tensor<1x65535x3x1xi16>, %arg1: !spirv.arm.tensor<7x1x1x1xi16>, %arg2: !spirv.arm.tensor<1xi32>) -> (!spirv.arm.tensor<1x65536x2x7xi32>) {
  %5 = spirv.Constant dense<35> : !spirv.arm.tensor<1xi16>
  %6 = spirv.Constant dense<57> : !spirv.arm.tensor<1xi16>
  // expected-error @+1 {{op expect result type to be i64, got 'i32'}}
  %7 = spirv.Tosa.Conv2D pad = [1, 0, 0, 0], stride = [1, 2], dilation = [7, 1], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x65535x3x1xi16>, !spirv.arm.tensor<7x1x1x1xi16>, !spirv.arm.tensor<1xi32>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi16> -> !spirv.arm.tensor<1x65536x2x7xi32>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x65536x2x7xi32>
}

spirv.ARM.Graph @conv2d_mismatch_result_element_type_f16_input(%arg0: !spirv.arm.tensor<1x34x18x27xf16>, %arg1: !spirv.arm.tensor<11x1x1x27xf16>, %arg2: !spirv.arm.tensor<11xf32>) -> (!spirv.arm.tensor<1x34x18x11xf32>) {
  %5 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf16>
  %6 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf16>
  // expected-error @+1 {{op expect result type to be f16, got 'f32'}}
  %7 = spirv.Tosa.Conv2D pad = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1], acc_type = <FP16>, local_bound = true, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x34x18x27xf16>, !spirv.arm.tensor<11x1x1x27xf16>, !spirv.arm.tensor<11xf32>, !spirv.arm.tensor<1xf16>, !spirv.arm.tensor<1xf16> -> !spirv.arm.tensor<1x34x18x11xf32>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x34x18x11xf32>
}

spirv.ARM.Graph @conv2d_mismatch_result_element_type_f32_input(%arg0: !spirv.arm.tensor<1x34x18x27xf32>, %arg1: !spirv.arm.tensor<11x1x1x27xf32>, %arg2: !spirv.arm.tensor<11xf16>) -> (!spirv.arm.tensor<1x34x18x11xf16>) {
  %5 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf32>
  %6 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf32>
  // expected-error @+1 {{op expect result type to be f32, got 'f16'}}
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
  // expected-error @+1 {{op accumulator type for i8 tensorARM is not i32}}
  %7 = spirv.Tosa.Conv2D pad = [1, 0, 0, 0], stride = [1, 2], dilation = [7, 1], acc_type = <INT48>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x65535x3x1xi8>, !spirv.arm.tensor<7x1x1x1xi8>, !spirv.arm.tensor<1xi32>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x65536x2x7xi32>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x65536x2x7xi32>
}

spirv.ARM.Graph @conv2d_accumulator_must_be_INT48_for_i16_input_element_type(%arg0: !spirv.arm.tensor<1x65535x3x1xi16>, %arg1: !spirv.arm.tensor<7x1x1x1xi16>, %arg2: !spirv.arm.tensor<1xi64>) -> (!spirv.arm.tensor<1x65536x2x7xi64>) {
  %5 = spirv.Constant dense<35> : !spirv.arm.tensor<1xi16>
  %6 = spirv.Constant dense<57> : !spirv.arm.tensor<1xi16>
  // expected-error @+1 {{op accumulator type for i16 tensorARM is not i48}}
  %7 = spirv.Tosa.Conv2D pad = [1, 0, 0, 0], stride = [1, 2], dilation = [7, 1], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x65535x3x1xi16>, !spirv.arm.tensor<7x1x1x1xi16>, !spirv.arm.tensor<1xi64>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi16> -> !spirv.arm.tensor<1x65536x2x7xi64>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x65536x2x7xi64>
}

spirv.ARM.Graph @conv2d_accumulator_must_be_either_FP16_or_FP32_for_f16_input_element_type(%arg0: !spirv.arm.tensor<1x34x18x27xf16>, %arg1: !spirv.arm.tensor<11x1x1x27xf16>, %arg2: !spirv.arm.tensor<11xf16>) -> (!spirv.arm.tensor<1x34x18x11xf16>) {
  %5 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf16>
  %6 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf16>
  // expected-error @+1 {{op accumulator type for f16 tensorARM is not f16 or f32}}
  %7 = spirv.Tosa.Conv2D pad = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1], acc_type = <INT32>, local_bound = true, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x34x18x27xf16>, !spirv.arm.tensor<11x1x1x27xf16>, !spirv.arm.tensor<11xf16>, !spirv.arm.tensor<1xf16>, !spirv.arm.tensor<1xf16> -> !spirv.arm.tensor<1x34x18x11xf16>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x34x18x11xf16>
}

spirv.ARM.Graph @conv2d_accumulator_must_be_either_FP32_for_f32_input_element_type(%arg0: !spirv.arm.tensor<1x34x18x27xf32>, %arg1: !spirv.arm.tensor<11x1x1x27xf32>, %arg2: !spirv.arm.tensor<11xf32>) -> (!spirv.arm.tensor<1x34x18x11xf32>) {
  %5 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf32>
  %6 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf32>
  // expected-error @+1 {{op accumulator type for f32 tensorARM is not f32}}
  %7 = spirv.Tosa.Conv2D pad = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1], acc_type = <FP16>, local_bound = true, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x34x18x27xf32>, !spirv.arm.tensor<11x1x1x27xf32>, !spirv.arm.tensor<11xf32>, !spirv.arm.tensor<1xf32>, !spirv.arm.tensor<1xf32> -> !spirv.arm.tensor<1x34x18x11xf32>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x34x18x11xf32>
}


//===----------------------------------------------------------------------===//
// spirv.TOSA.Conv3D
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @conv3d_wrong_input_integer_element_type(%arg0: !spirv.arm.tensor<1x65535x3x1x1xi32>, %arg1: !spirv.arm.tensor<7x1x1x1x1xi32>, %arg2: !spirv.arm.tensor<1xi64>) -> (!spirv.arm.tensor<1x65536x2x7x1xi64>) {
  %5 = spirv.Constant dense<35> : !spirv.arm.tensor<1xi32>
  %6 = spirv.Constant dense<57> : !spirv.arm.tensor<1xi32>
  // expected-error @+1 {{op input element type can only be of width 8 or 16 when integer type}}
  %7 = spirv.Tosa.Conv3D pad = [1, 0, 0, 0, 0, 0], stride = [1, 2, 3], dilation = [7, 1, 1], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x65535x3x1x1xi32>, !spirv.arm.tensor<7x1x1x1x1xi32>, !spirv.arm.tensor<1xi64>, !spirv.arm.tensor<1xi32>, !spirv.arm.tensor<1xi32> -> !spirv.arm.tensor<1x65536x2x7x1xi64>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x65536x2x7x1xi64>
}

spirv.ARM.Graph @conv3d_mismatch_result_element_type_i8_input(%arg0: !spirv.arm.tensor<1x65535x3x1x1xi8>, %arg1: !spirv.arm.tensor<7x1x1x1x1xi8>, %arg2: !spirv.arm.tensor<1xi16>) -> (!spirv.arm.tensor<1x65536x2x7x1xi16>) {
  %5 = spirv.Constant dense<35> : !spirv.arm.tensor<1xi8>
  %6 = spirv.Constant dense<57> : !spirv.arm.tensor<1xi8>
  // expected-error @+1 {{op expect result type to be i32, got 'i16'}}
  %7 = spirv.Tosa.Conv3D pad = [1, 0, 0, 0, 0, 0], stride = [1, 2, 3], dilation = [7, 1, 1], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x65535x3x1x1xi8>, !spirv.arm.tensor<7x1x1x1x1xi8>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x65536x2x7x1xi16>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x65536x2x7x1xi16>
}

spirv.ARM.Graph @conv3d_mismatch_result_element_type_i16_input(%arg0: !spirv.arm.tensor<1x65535x3x1x1xi16>, %arg1: !spirv.arm.tensor<7x1x1x1x1xi16>, %arg2: !spirv.arm.tensor<1xi32>) -> (!spirv.arm.tensor<1x65536x2x7x1xi32>) {
  %5 = spirv.Constant dense<35> : !spirv.arm.tensor<1xi16>
  %6 = spirv.Constant dense<57> : !spirv.arm.tensor<1xi16>
  // expected-error @+1 {{op expect result type to be i64, got 'i32'}}
  %7 = spirv.Tosa.Conv3D pad = [1, 0, 0, 0, 0, 0], stride = [1, 2, 3], dilation = [7, 1, 1], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x65535x3x1x1xi16>, !spirv.arm.tensor<7x1x1x1x1xi16>, !spirv.arm.tensor<1xi32>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi16> -> !spirv.arm.tensor<1x65536x2x7x1xi32>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x65536x2x7x1xi32>
}

spirv.ARM.Graph @conv3d_mismatch_result_element_type_f16_input(%arg0: !spirv.arm.tensor<1x34x18x27x1xf16>, %arg1: !spirv.arm.tensor<11x1x1x27x1xf16>, %arg2: !spirv.arm.tensor<11xf32>) -> (!spirv.arm.tensor<1x34x18x11x1xf32>) {
  %5 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf16>
  %6 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf16>
  // expected-error @+1 {{op expect result type to be f16, got 'f32'}}
  %7 = spirv.Tosa.Conv3D pad = [0, 0, 0, 0, 0, 0], stride = [1, 1, 1], dilation = [1, 1, 1], acc_type = <FP16>, local_bound = true, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x34x18x27x1xf16>, !spirv.arm.tensor<11x1x1x27x1xf16>, !spirv.arm.tensor<11xf32>, !spirv.arm.tensor<1xf16>, !spirv.arm.tensor<1xf16> -> !spirv.arm.tensor<1x34x18x11x1xf32>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x34x18x11x1xf32>
}

spirv.ARM.Graph @conv3d_mismatch_result_element_type_f32_input(%arg0: !spirv.arm.tensor<1x34x18x27x1xf32>, %arg1: !spirv.arm.tensor<11x1x1x27x1xf32>, %arg2: !spirv.arm.tensor<11xf16>) -> (!spirv.arm.tensor<1x34x18x11x1xf16>) {
  %5 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf32>
  %6 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf32>
  // expected-error @+1 {{op expect result type to be f32, got 'f16'}}
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
  // expected-error @+1 {{op accumulator type for i8 tensorARM is not i32}}
  %7 = spirv.Tosa.Conv3D pad = [1, 0, 0, 0, 0, 0], stride = [1, 2, 3], dilation = [7, 1, 1], acc_type = <INT48>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x65535x3x1x1xi8>, !spirv.arm.tensor<7x1x1x1x1xi8>, !spirv.arm.tensor<1xi32>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x65536x2x7x1xi32>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x65536x2x7x1xi32>
}

spirv.ARM.Graph @conv3d_accumulator_must_be_INT48_for_i16_input_element_type(%arg0: !spirv.arm.tensor<1x65535x3x1x1xi16>, %arg1: !spirv.arm.tensor<7x1x1x1x1xi16>, %arg2: !spirv.arm.tensor<1xi64>) -> (!spirv.arm.tensor<1x65536x2x7x1xi64>) {
  %5 = spirv.Constant dense<35> : !spirv.arm.tensor<1xi16>
  %6 = spirv.Constant dense<57> : !spirv.arm.tensor<1xi16>
  // expected-error @+1 {{op accumulator type for i16 tensorARM is not i48}}
  %7 = spirv.Tosa.Conv3D pad = [1, 0, 0, 0, 0, 0], stride = [1, 2, 3], dilation = [7, 1, 1], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x65535x3x1x1xi16>, !spirv.arm.tensor<7x1x1x1x1xi16>, !spirv.arm.tensor<1xi64>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi16> -> !spirv.arm.tensor<1x65536x2x7x1xi64>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x65536x2x7x1xi64>
}

spirv.ARM.Graph @conv3d_accumulator_must_be_either_FP16_or_FP32_for_f16_input_element_type(%arg0: !spirv.arm.tensor<1x34x18x27x1xf16>, %arg1: !spirv.arm.tensor<11x1x1x27x1xf16>, %arg2: !spirv.arm.tensor<11xf16>) -> (!spirv.arm.tensor<1x34x18x11x1xf16>) {
  %5 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf16>
  %6 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf16>
  // expected-error @+1 {{op accumulator type for f16 tensorARM is not f16 or f32}}
  %7 = spirv.Tosa.Conv3D pad = [0, 0, 0, 0, 0, 0], stride = [1, 1, 1], dilation = [1, 1, 1], acc_type = <INT32>, local_bound = true, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x34x18x27x1xf16>, !spirv.arm.tensor<11x1x1x27x1xf16>, !spirv.arm.tensor<11xf16>, !spirv.arm.tensor<1xf16>, !spirv.arm.tensor<1xf16> -> !spirv.arm.tensor<1x34x18x11x1xf16>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x34x18x11x1xf16>
}

spirv.ARM.Graph @conv3d_accumulator_must_be_either_FP32_for_f32_input_element_type(%arg0: !spirv.arm.tensor<1x34x18x27x1xf32>, %arg1: !spirv.arm.tensor<11x1x1x27x1xf32>, %arg2: !spirv.arm.tensor<11xf32>) -> (!spirv.arm.tensor<1x34x18x11x1xf32>) {
  %5 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf32>
  %6 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf32>
  // expected-error @+1 {{op accumulator type for f32 tensorARM is not f32}}
  %7 = spirv.Tosa.Conv3D pad = [0, 0, 0, 0, 0, 0], stride = [1, 1, 1], dilation = [1, 1, 1], acc_type = <FP16>, local_bound = true, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x34x18x27x1xf32>, !spirv.arm.tensor<11x1x1x27x1xf32>, !spirv.arm.tensor<11xf32>, !spirv.arm.tensor<1xf32>, !spirv.arm.tensor<1xf32> -> !spirv.arm.tensor<1x34x18x11x1xf32>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x34x18x11x1xf32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.DepthwiseConv2D
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @depthwise_conv2d_wrong_input_integer_element_type(%arg0: !spirv.arm.tensor<1x65535x3x1xi32>, %arg1: !spirv.arm.tensor<7x1x1x1xi32>, %arg2: !spirv.arm.tensor<1xi64>) -> (!spirv.arm.tensor<1x65536x2x7xi64>) {
  %5 = spirv.Constant dense<35> : !spirv.arm.tensor<1xi32>
  %6 = spirv.Constant dense<57> : !spirv.arm.tensor<1xi32>
  // expected-error @+1 {{op input element type can only be of width 8 or 16 when integer type}}
  %7 = spirv.Tosa.DepthwiseConv2D pad = [1, 0, 0, 0], stride = [1, 2], dilation = [7, 1], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x65535x3x1xi32>, !spirv.arm.tensor<7x1x1x1xi32>, !spirv.arm.tensor<1xi64>, !spirv.arm.tensor<1xi32>, !spirv.arm.tensor<1xi32> -> !spirv.arm.tensor<1x65536x2x7xi64>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x65536x2x7xi64>
}

spirv.ARM.Graph @depthwise_conv2d_mismatch_result_element_type_i8_input(%arg0: !spirv.arm.tensor<1x65535x3x1xi8>, %arg1: !spirv.arm.tensor<7x1x1x1xi8>, %arg2: !spirv.arm.tensor<1xi16>) -> (!spirv.arm.tensor<1x65536x2x7xi16>) {
  %5 = spirv.Constant dense<35> : !spirv.arm.tensor<1xi8>
  %6 = spirv.Constant dense<57> : !spirv.arm.tensor<1xi8>
  // expected-error @+1 {{op expect result type to be i32, got 'i16'}}
  %7 = spirv.Tosa.DepthwiseConv2D pad = [1, 0, 0, 0], stride = [1, 2], dilation = [7, 1], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x65535x3x1xi8>, !spirv.arm.tensor<7x1x1x1xi8>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x65536x2x7xi16>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x65536x2x7xi16>
}

spirv.ARM.Graph @depthwise_conv2d_mismatch_result_element_type_i16_input(%arg0: !spirv.arm.tensor<1x65535x3x1xi16>, %arg1: !spirv.arm.tensor<7x1x1x1xi16>, %arg2: !spirv.arm.tensor<1xi32>) -> (!spirv.arm.tensor<1x65536x2x7xi32>) {
  %5 = spirv.Constant dense<35> : !spirv.arm.tensor<1xi16>
  %6 = spirv.Constant dense<57> : !spirv.arm.tensor<1xi16>
  // expected-error @+1 {{op expect result type to be i64, got 'i32'}}
  %7 = spirv.Tosa.DepthwiseConv2D pad = [1, 0, 0, 0], stride = [1, 2], dilation = [7, 1], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x65535x3x1xi16>, !spirv.arm.tensor<7x1x1x1xi16>, !spirv.arm.tensor<1xi32>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi16> -> !spirv.arm.tensor<1x65536x2x7xi32>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x65536x2x7xi32>
}

spirv.ARM.Graph @depthwise_conv2d_mismatch_result_element_type_f16_input(%arg0: !spirv.arm.tensor<1x34x18x27xf16>, %arg1: !spirv.arm.tensor<11x1x1x27xf16>, %arg2: !spirv.arm.tensor<11xf32>) -> (!spirv.arm.tensor<1x34x18x11xf32>) {
  %5 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf16>
  %6 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf16>
  // expected-error @+1 {{op expect result type to be f16, got 'f32'}}
  %7 = spirv.Tosa.DepthwiseConv2D pad = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1], acc_type = <FP16>, local_bound = true, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x34x18x27xf16>, !spirv.arm.tensor<11x1x1x27xf16>, !spirv.arm.tensor<11xf32>, !spirv.arm.tensor<1xf16>, !spirv.arm.tensor<1xf16> -> !spirv.arm.tensor<1x34x18x11xf32>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x34x18x11xf32>
}

spirv.ARM.Graph @depthwise_conv2d_mismatch_result_element_type_f32_input(%arg0: !spirv.arm.tensor<1x34x18x27xf32>, %arg1: !spirv.arm.tensor<11x1x1x27xf32>, %arg2: !spirv.arm.tensor<11xf16>) -> (!spirv.arm.tensor<1x34x18x11xf16>) {
  %5 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf32>
  %6 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf32>
  // expected-error @+1 {{op expect result type to be f32, got 'f16'}}
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
  // expected-error @+1 {{op accumulator type for i8 tensorARM is not i32}}
  %7 = spirv.Tosa.DepthwiseConv2D pad = [1, 0, 0, 0], stride = [1, 2], dilation = [7, 1], acc_type = <INT48>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x65535x3x1xi8>, !spirv.arm.tensor<7x1x1x1xi8>, !spirv.arm.tensor<1xi32>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x65536x2x7xi32>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x65536x2x7xi32>
}

spirv.ARM.Graph @depthwise_conv2d_accumulator_must_be_INT48_for_i16_input_element_type(%arg0: !spirv.arm.tensor<1x65535x3x1xi16>, %arg1: !spirv.arm.tensor<7x1x1x1xi16>, %arg2: !spirv.arm.tensor<1xi64>) -> (!spirv.arm.tensor<1x65536x2x7xi64>) {
  %5 = spirv.Constant dense<35> : !spirv.arm.tensor<1xi16>
  %6 = spirv.Constant dense<57> : !spirv.arm.tensor<1xi16>
  // expected-error @+1 {{op accumulator type for i16 tensorARM is not i48}}
  %7 = spirv.Tosa.DepthwiseConv2D pad = [1, 0, 0, 0], stride = [1, 2], dilation = [7, 1], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x65535x3x1xi16>, !spirv.arm.tensor<7x1x1x1xi16>, !spirv.arm.tensor<1xi64>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi16> -> !spirv.arm.tensor<1x65536x2x7xi64>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x65536x2x7xi64>
}

spirv.ARM.Graph @depthwise_conv2d_accumulator_must_be_either_FP16_or_FP32_for_f16_input_element_type(%arg0: !spirv.arm.tensor<1x34x18x27xf16>, %arg1: !spirv.arm.tensor<11x1x1x27xf16>, %arg2: !spirv.arm.tensor<11xf16>) -> (!spirv.arm.tensor<1x34x18x11xf16>) {
  %5 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf16>
  %6 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf16>
  // expected-error @+1 {{op accumulator type for f16 tensorARM is not f16 or f32}}
  %7 = spirv.Tosa.DepthwiseConv2D pad = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1], acc_type = <INT32>, local_bound = true, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x34x18x27xf16>, !spirv.arm.tensor<11x1x1x27xf16>, !spirv.arm.tensor<11xf16>, !spirv.arm.tensor<1xf16>, !spirv.arm.tensor<1xf16> -> !spirv.arm.tensor<1x34x18x11xf16>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x34x18x11xf16>
}

spirv.ARM.Graph @depthwise_conv2d_accumulator_must_be_either_FP32_for_f32_input_element_type(%arg0: !spirv.arm.tensor<1x34x18x27xf32>, %arg1: !spirv.arm.tensor<11x1x1x27xf32>, %arg2: !spirv.arm.tensor<11xf32>) -> (!spirv.arm.tensor<1x34x18x11xf32>) {
  %5 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf32>
  %6 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf32>
  // expected-error @+1 {{op accumulator type for f32 tensorARM is not f32}}
  %7 = spirv.Tosa.DepthwiseConv2D pad = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1], acc_type = <FP16>, local_bound = true, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x34x18x27xf32>, !spirv.arm.tensor<11x1x1x27xf32>, !spirv.arm.tensor<11xf32>, !spirv.arm.tensor<1xf32>, !spirv.arm.tensor<1xf32> -> !spirv.arm.tensor<1x34x18x11xf32>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x34x18x11xf32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.TransposeConv2D
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @transpose_conv2d_wrong_input_integer_element_type(%arg0: !spirv.arm.tensor<1x65535x3x1xi32>, %arg1: !spirv.arm.tensor<7x1x1x1xi32>, %arg2: !spirv.arm.tensor<1xi64>) -> (!spirv.arm.tensor<1x65536x2x7xi64>) {
  %5 = spirv.Constant dense<35> : !spirv.arm.tensor<1xi32>
  %6 = spirv.Constant dense<57> : !spirv.arm.tensor<1xi32>
  // expected-error @+1 {{op input element type can only be of width 8 or 16 when integer type}}
  %7 = spirv.Tosa.TransposeConv2D out_pad = [1, 0, 0, 0], stride = [1, 2], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x65535x3x1xi32>, !spirv.arm.tensor<7x1x1x1xi32>, !spirv.arm.tensor<1xi64>, !spirv.arm.tensor<1xi32>, !spirv.arm.tensor<1xi32> -> !spirv.arm.tensor<1x65536x2x7xi64>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x65536x2x7xi64>
}

spirv.ARM.Graph @transpose_conv2d_mismatch_result_element_type_i8_input(%arg0: !spirv.arm.tensor<1x65535x3x1xi8>, %arg1: !spirv.arm.tensor<7x1x1x1xi8>, %arg2: !spirv.arm.tensor<1xi16>) -> (!spirv.arm.tensor<1x65536x2x7xi16>) {
  %5 = spirv.Constant dense<35> : !spirv.arm.tensor<1xi8>
  %6 = spirv.Constant dense<57> : !spirv.arm.tensor<1xi8>
  // expected-error @+1 {{op expect result type to be i32, got 'i16'}}
  %7 = spirv.Tosa.TransposeConv2D out_pad = [1, 0, 0, 0], stride = [1, 2], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x65535x3x1xi8>, !spirv.arm.tensor<7x1x1x1xi8>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x65536x2x7xi16>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x65536x2x7xi16>
}

spirv.ARM.Graph @transpose_conv2d_mismatch_result_element_type_i16_input(%arg0: !spirv.arm.tensor<1x65535x3x1xi16>, %arg1: !spirv.arm.tensor<7x1x1x1xi16>, %arg2: !spirv.arm.tensor<1xi32>) -> (!spirv.arm.tensor<1x65536x2x7xi32>) {
  %5 = spirv.Constant dense<35> : !spirv.arm.tensor<1xi16>
  %6 = spirv.Constant dense<57> : !spirv.arm.tensor<1xi16>
  // expected-error @+1 {{op expect result type to be i64, got 'i32'}}
  %7 = spirv.Tosa.TransposeConv2D out_pad = [1, 0, 0, 0], stride = [1, 2], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x65535x3x1xi16>, !spirv.arm.tensor<7x1x1x1xi16>, !spirv.arm.tensor<1xi32>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi16> -> !spirv.arm.tensor<1x65536x2x7xi32>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x65536x2x7xi32>
}

spirv.ARM.Graph @transpose_conv2d_mismatch_result_element_type_f16_input(%arg0: !spirv.arm.tensor<1x34x18x27xf16>, %arg1: !spirv.arm.tensor<11x1x1x27xf16>, %arg2: !spirv.arm.tensor<11xf32>) -> (!spirv.arm.tensor<1x34x18x11xf32>) {
  %5 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf16>
  %6 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf16>
  // expected-error @+1 {{op expect result type to be f16, got 'f32'}}
  %7 = spirv.Tosa.TransposeConv2D out_pad = [0, 0, 0, 0], stride = [1, 1], acc_type = <FP16>, local_bound = true, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x34x18x27xf16>, !spirv.arm.tensor<11x1x1x27xf16>, !spirv.arm.tensor<11xf32>, !spirv.arm.tensor<1xf16>, !spirv.arm.tensor<1xf16> -> !spirv.arm.tensor<1x34x18x11xf32>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x34x18x11xf32>
}

spirv.ARM.Graph @transpose_conv2d_mismatch_result_element_type_f32_input(%arg0: !spirv.arm.tensor<1x34x18x27xf32>, %arg1: !spirv.arm.tensor<11x1x1x27xf32>, %arg2: !spirv.arm.tensor<11xf16>) -> (!spirv.arm.tensor<1x34x18x11xf16>) {
  %5 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf32>
  %6 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf32>
  // expected-error @+1 {{op expect result type to be f32, got 'f16'}}
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
  // expected-error @+1 {{op accumulator type for i8 tensorARM is not i32}}
  %7 = spirv.Tosa.TransposeConv2D out_pad = [1, 0, 0, 0], stride = [1, 2], acc_type = <INT48>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x65535x3x1xi8>, !spirv.arm.tensor<7x1x1x1xi8>, !spirv.arm.tensor<1xi32>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x65536x2x7xi32>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x65536x2x7xi32>
}

spirv.ARM.Graph @transpose_conv2d_accumulator_must_be_INT48_for_i16_input_element_type(%arg0: !spirv.arm.tensor<1x65535x3x1xi16>, %arg1: !spirv.arm.tensor<7x1x1x1xi16>, %arg2: !spirv.arm.tensor<1xi64>) -> (!spirv.arm.tensor<1x65536x2x7xi64>) {
  %5 = spirv.Constant dense<35> : !spirv.arm.tensor<1xi16>
  %6 = spirv.Constant dense<57> : !spirv.arm.tensor<1xi16>
  // expected-error @+1 {{op accumulator type for i16 tensorARM is not i48}}
  %7 = spirv.Tosa.TransposeConv2D out_pad = [1, 0, 0, 0], stride = [1, 2], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x65535x3x1xi16>, !spirv.arm.tensor<7x1x1x1xi16>, !spirv.arm.tensor<1xi64>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi16> -> !spirv.arm.tensor<1x65536x2x7xi64>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x65536x2x7xi64>
}

spirv.ARM.Graph @transpose_conv2d_accumulator_must_be_either_FP16_or_FP32_for_f16_input_element_type(%arg0: !spirv.arm.tensor<1x34x18x27xf16>, %arg1: !spirv.arm.tensor<11x1x1x27xf16>, %arg2: !spirv.arm.tensor<11xf16>) -> (!spirv.arm.tensor<1x34x18x11xf16>) {
  %5 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf16>
  %6 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf16>
  // expected-error @+1 {{op accumulator type for f16 tensorARM is not f16 or f32}}
  %7 = spirv.Tosa.TransposeConv2D out_pad = [0, 0, 0, 0], stride = [1, 1], acc_type = <INT32>, local_bound = true, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x34x18x27xf16>, !spirv.arm.tensor<11x1x1x27xf16>, !spirv.arm.tensor<11xf16>, !spirv.arm.tensor<1xf16>, !spirv.arm.tensor<1xf16> -> !spirv.arm.tensor<1x34x18x11xf16>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x34x18x11xf16>
}

spirv.ARM.Graph @transpose_conv2d_accumulator_must_be_either_FP32_for_f32_input_element_type(%arg0: !spirv.arm.tensor<1x34x18x27xf32>, %arg1: !spirv.arm.tensor<11x1x1x27xf32>, %arg2: !spirv.arm.tensor<11xf32>) -> (!spirv.arm.tensor<1x34x18x11xf32>) {
  %5 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf32>
  %6 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf32>
  // expected-error @+1 {{op accumulator type for f32 tensorARM is not f32}}
  %7 = spirv.Tosa.TransposeConv2D out_pad = [0, 0, 0, 0], stride = [1, 1], acc_type = <FP16>, local_bound = true, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x34x18x27xf32>, !spirv.arm.tensor<11x1x1x27xf32>, !spirv.arm.tensor<11xf32>, !spirv.arm.tensor<1xf32>, !spirv.arm.tensor<1xf32> -> !spirv.arm.tensor<1x34x18x11xf32>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x34x18x11xf32>
}
