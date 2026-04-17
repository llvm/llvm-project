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
// spirv.TOSA.AvgPool2D
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @avgpool2d_input_output_different_elemnt_type(%arg0: !spirv.arm.tensor<1x3x65537x1xi8>) -> (!spirv.arm.tensor<1x2x32768x1xi16>) {
  %4 = spirv.Constant dense<125> : !spirv.arm.tensor<1xi8>
  %5 = spirv.Constant dense<-90> : !spirv.arm.tensor<1xi8>
  // expected-error @+1 {{op failed to verify that all of {input, input_zp, output, output_zp} have same element type}}
  %6 = spirv.Tosa.AvgPool2D kernel = [3, 3], stride = [1, 2], pad = [0, 1, 0, 0], acc_type = <INT32>, %arg0, %4, %5 : !spirv.arm.tensor<1x3x65537x1xi8>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x2x32768x1xi16>
  spirv.ARM.GraphOutputs %6 : !spirv.arm.tensor<1x2x32768x1xi16>
}

spirv.ARM.Graph @avgpool2d_input_input_zero_point_different_elemnt_type(%arg0: !spirv.arm.tensor<1x3x65537x1xi8>) -> (!spirv.arm.tensor<1x2x32768x1xi8>) {
  %4 = spirv.Constant dense<125> : !spirv.arm.tensor<1xi16>
  %5 = spirv.Constant dense<-90> : !spirv.arm.tensor<1xi8>
  // expected-error @+1 {{op failed to verify that all of {input, input_zp, output, output_zp} have same element type}}
  %6 = spirv.Tosa.AvgPool2D kernel = [3, 3], stride = [1, 2], pad = [0, 1, 0, 0], acc_type = <INT32>, %arg0, %4, %5 : !spirv.arm.tensor<1x3x65537x1xi8>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x2x32768x1xi8>
  spirv.ARM.GraphOutputs %6 : !spirv.arm.tensor<1x2x32768x1xi8>
}

spirv.ARM.Graph @avgpool2d_output_output_zero_point_different_elemnt_type(%arg0: !spirv.arm.tensor<1x3x65537x1xi8>) -> (!spirv.arm.tensor<1x2x32768x1xi8>) {
  %4 = spirv.Constant dense<125> : !spirv.arm.tensor<1xi8>
  %5 = spirv.Constant dense<-90> : !spirv.arm.tensor<1xi16>
  // expected-error @+1 {{op failed to verify that all of {input, input_zp, output, output_zp} have same element type}}
  %6 = spirv.Tosa.AvgPool2D kernel = [3, 3], stride = [1, 2], pad = [0, 1, 0, 0], acc_type = <INT32>, %arg0, %4, %5 : !spirv.arm.tensor<1x3x65537x1xi8>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi16> -> !spirv.arm.tensor<1x2x32768x1xi8>
  spirv.ARM.GraphOutputs %6 : !spirv.arm.tensor<1x2x32768x1xi8>
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
// spirv.TOSA.MatMul
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @matmul_mismatch_result_element_type_i8_input(%arg0: !spirv.arm.tensor<1x4x4xi8>, %arg1: !spirv.arm.tensor<1x4x4xi8>, %arg2: !spirv.arm.tensor<1xi8>, %arg3: !spirv.arm.tensor<1xi8>) -> (!spirv.arm.tensor<1x4x4xi16>) {
  // expected-error @+1 {{op failed to verify that if A has type 8-bit signless integer then output must have a type in [32-bit signless integer]}}
  %0 = spirv.Tosa.MatMul %arg0, %arg1, %arg2, %arg3 : !spirv.arm.tensor<1x4x4xi8>, !spirv.arm.tensor<1x4x4xi8>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x4x4xi16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<1x4x4xi16>
}

spirv.ARM.Graph @matmul_mismatch_result_element_type_i16_input(%arg0: !spirv.arm.tensor<1x4x4xi16>, %arg1: !spirv.arm.tensor<1x4x4xi16>, %arg2: !spirv.arm.tensor<1xi16>, %arg3: !spirv.arm.tensor<1xi16>) -> (!spirv.arm.tensor<1x4x4xi32>) {
  // expected-error @+1 {{op failed to verify that if A has type 16-bit signless integer then output must have a type in [64-bit signless integer]}}
  %0 = spirv.Tosa.MatMul %arg0, %arg1, %arg2, %arg3 : !spirv.arm.tensor<1x4x4xi16>, !spirv.arm.tensor<1x4x4xi16>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi16> -> !spirv.arm.tensor<1x4x4xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<1x4x4xi32>
}

spirv.ARM.Graph @matmul_mismatch_result_element_type_bf16_input(%arg0: !spirv.arm.tensor<1x4x4xbf16>, %arg1: !spirv.arm.tensor<1x4x4xbf16>, %arg2: !spirv.arm.tensor<1xbf16>, %arg3: !spirv.arm.tensor<1xbf16>) -> (!spirv.arm.tensor<1x4x4xbf16>) {
  // expected-error @+1 {{op failed to verify that if A has type bfloat16 type then output must have a type in [32-bit float]}}
  %0 = spirv.Tosa.MatMul %arg0, %arg1, %arg2, %arg3 : !spirv.arm.tensor<1x4x4xbf16>, !spirv.arm.tensor<1x4x4xbf16>, !spirv.arm.tensor<1xbf16>, !spirv.arm.tensor<1xbf16> -> !spirv.arm.tensor<1x4x4xbf16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<1x4x4xbf16>
}

spirv.ARM.Graph @matmul_mismatch_result_element_type_f16_input(%arg0: !spirv.arm.tensor<1x4x4xf16>, %arg1: !spirv.arm.tensor<1x4x4xf16>, %arg2: !spirv.arm.tensor<1xf16>, %arg3: !spirv.arm.tensor<1xf16>) -> (!spirv.arm.tensor<1x4x4xi32>) {
  // expected-error @+1 {{op failed to verify that if A has type 16-bit float then output must have a type in [16-bit float,32-bit float]}}
  %0 = spirv.Tosa.MatMul %arg0, %arg1, %arg2, %arg3 : !spirv.arm.tensor<1x4x4xf16>, !spirv.arm.tensor<1x4x4xf16>, !spirv.arm.tensor<1xf16>, !spirv.arm.tensor<1xf16> -> !spirv.arm.tensor<1x4x4xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<1x4x4xi32>
}

spirv.ARM.Graph @matmul_mismatch_result_element_type_f32_input(%arg0: !spirv.arm.tensor<1x4x4xf32>, %arg1: !spirv.arm.tensor<1x4x4xf32>, %arg2: !spirv.arm.tensor<1xf32>, %arg3: !spirv.arm.tensor<1xf32>) -> (!spirv.arm.tensor<1x4x4xf16>) {
  // expected-error @+1 {{op failed to verify that if A has type 32-bit float then output must have a type in [32-bit float]}}
  %0 = spirv.Tosa.MatMul %arg0, %arg1, %arg2, %arg3 : !spirv.arm.tensor<1x4x4xf32>, !spirv.arm.tensor<1x4x4xf32>, !spirv.arm.tensor<1xf32>, !spirv.arm.tensor<1xf32> -> !spirv.arm.tensor<1x4x4xf16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<1x4x4xf16>
}

spirv.ARM.Graph @matmul_element_types_must_match_between_input_A_and_A_zero_point(%arg0: !spirv.arm.tensor<1x4x4xi8>, %arg1: !spirv.arm.tensor<1x4x4xi8>, %arg2: !spirv.arm.tensor<1xi16>, %arg3: !spirv.arm.tensor<1xi8>) -> (!spirv.arm.tensor<1x4x4xi32>) {
  // expected-error @+1 {{op failed to verify that all of {A, A_zp, B, B_zp} have same element type}}
  %0 = spirv.Tosa.MatMul %arg0, %arg1, %arg2, %arg3 : !spirv.arm.tensor<1x4x4xi8>, !spirv.arm.tensor<1x4x4xi8>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x4x4xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<1x4x4xi32>
}

spirv.ARM.Graph @matmul_element_types_must_match_between_inputs_A_and_B(%arg0: !spirv.arm.tensor<1x4x4xi8>, %arg1: !spirv.arm.tensor<1x4x4xi16>, %arg2: !spirv.arm.tensor<1xi8>, %arg3: !spirv.arm.tensor<1xi8>) -> (!spirv.arm.tensor<1x4x4xi32>) {
  // expected-error @+1 {{op failed to verify that all of {A, A_zp, B, B_zp} have same element type}}
  %0 = spirv.Tosa.MatMul %arg0, %arg1, %arg2, %arg3 : !spirv.arm.tensor<1x4x4xi8>, !spirv.arm.tensor<1x4x4xi16>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x4x4xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<1x4x4xi32>
}

spirv.ARM.Graph @matmul_element_types_must_match_between_input_B_and_B_zero_point(%arg0: !spirv.arm.tensor<1x4x4xi8>, %arg1: !spirv.arm.tensor<1x4x4xi8>, %arg2: !spirv.arm.tensor<1xi8>, %arg3: !spirv.arm.tensor<1xi16>) -> (!spirv.arm.tensor<1x4x4xi32>) {
  // expected-error @+1 {{op failed to verify that all of {A, A_zp, B, B_zp} have same element type}}
  %0 = spirv.Tosa.MatMul %arg0, %arg1, %arg2, %arg3 : !spirv.arm.tensor<1x4x4xi8>, !spirv.arm.tensor<1x4x4xi8>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi16> -> !spirv.arm.tensor<1x4x4xi32>
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
// spirv.TOSA.Erf
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @erf_input_output_element_types_not_matching(%arg0: !spirv.arm.tensor<47x38x51xf16>) -> (!spirv.arm.tensor<47x38x51xf32>) {
  // expected-error @+1 {{op failed to verify that all of {input, output} have same type}}
  %0 = spirv.Tosa.Erf %arg0 : !spirv.arm.tensor<47x38x51xf16> -> !spirv.arm.tensor<47x38x51xf32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<47x38x51xf32>
}

spirv.ARM.Graph @erf_input_output_shapes_not_matching(%arg0: !spirv.arm.tensor<47x38x51xf16>) -> (!spirv.arm.tensor<47x38x52xf16>) {
  // expected-error @+1 {{op failed to verify that all of {input, output} have same type}}
  %0 = spirv.Tosa.Erf %arg0 : !spirv.arm.tensor<47x38x51xf16> -> !spirv.arm.tensor<47x38x52xf16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<47x38x52xf16>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Sigmoid
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @sigmoid_input_output_element_types_not_matching(%arg0: !spirv.arm.tensor<28x43x45xf16>) -> (!spirv.arm.tensor<28x43x45xf32>) {
  // expected-error @+1 {{op failed to verify that all of {input, output} have same type}}
  %0 = spirv.Tosa.Sigmoid %arg0 : !spirv.arm.tensor<28x43x45xf16> -> !spirv.arm.tensor<28x43x45xf32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<28x43x45xf32>
}

spirv.ARM.Graph @sigmoid_input_output_shapes_not_matching(%arg0: !spirv.arm.tensor<28x43x45xf16>) -> (!spirv.arm.tensor<29x43x45xf16>) {
  // expected-error @+1 {{op failed to verify that all of {input, output} have same type}}
  %0 = spirv.Tosa.Sigmoid %arg0 : !spirv.arm.tensor<28x43x45xf16> -> !spirv.arm.tensor<29x43x45xf16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<29x43x45xf16>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Tanh
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @tanh_input_output_element_types_not_matching(%arg0: !spirv.arm.tensor<46x50x36xf16>) -> (!spirv.arm.tensor<46x50x36xf32>) {
  // expected-error @+1 {{op failed to verify that all of {input, output} have same type}}
  %0 = spirv.Tosa.Tanh %arg0 : !spirv.arm.tensor<46x50x36xf16> -> !spirv.arm.tensor<46x50x36xf32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<46x50x36xf32>
}

spirv.ARM.Graph @tanh_input_output_shapes_not_matching(%arg0: !spirv.arm.tensor<46x50x36xf16>) -> (!spirv.arm.tensor<46x51x36xf16>) {
  // expected-error @+1 {{op failed to verify that all of {input, output} have same type}}
  %0 = spirv.Tosa.Tanh %arg0 : !spirv.arm.tensor<46x50x36xf16> -> !spirv.arm.tensor<46x51x36xf16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<46x51x36xf16>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Add
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @add_input_ranks_not_matching(%arg0: !spirv.arm.tensor<6x10x6xi32>, %arg1: !spirv.arm.tensor<1x10x6x6xi32>) -> (!spirv.arm.tensor<6x10x6x6xi32>) {
  // expected-error @+1 {{op failed to verify that all of {input1, input2, output} have same rank}}
  %0 = spirv.Tosa.Add %arg0, %arg1 : !spirv.arm.tensor<6x10x6xi32>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi32>
}

spirv.ARM.Graph @add_input_element_types_not_matching(%arg0: !spirv.arm.tensor<6x10x6x6xi16>, %arg1: !spirv.arm.tensor<1x10x6x6xi32>) -> (!spirv.arm.tensor<6x10x6x6xi16>) {
  // expected-error @+1 {{op failed to verify that all of {input1, input2} have same element type}}
  %0 = spirv.Tosa.Add %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi16>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi16>
}

spirv.ARM.Graph @add_input_output_element_types_not_matching(%arg0: !spirv.arm.tensor<6x10x6x6xi32>, %arg1: !spirv.arm.tensor<1x10x6x6xi32>) -> (!spirv.arm.tensor<6x10x6x6xi16>) {
  // expected-error @+1 {{op failed to verify that all of {input1, output} have same element type}}
  %0 = spirv.Tosa.Add %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi32>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi16>
}

spirv.ARM.Graph @add_inputs_not_broadcastable(%arg0: !spirv.arm.tensor<6x10x6x6xi32>, %arg1: !spirv.arm.tensor<2x10x6x6xi32>) -> (!spirv.arm.tensor<6x10x6x6xi32>) {
  // expected-error @+1 {{op failed to verify that the shape of input1 and input2 are compatible for broadcasting and the broadcast shape is equal to the output shape}}
  %0 = spirv.Tosa.Add %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi32>, !spirv.arm.tensor<2x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi32>
}

spirv.ARM.Graph @add_output_shape_does_not_match_broadcast_shape(%arg0: !spirv.arm.tensor<6x10x6x6xi32>, %arg1: !spirv.arm.tensor<1x10x6x6xi32>) -> (!spirv.arm.tensor<1x10x6x6xi32>) {
  // expected-error @+1 {{op failed to verify that the shape of input1 and input2 are compatible for broadcasting and the broadcast shape is equal to the output shape}}
  %0 = spirv.Tosa.Add %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi32>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<1x10x6x6xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<1x10x6x6xi32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.ArithmeticRightShift
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @arithmeticrightshift_input_ranks_not_matching(%arg0: !spirv.arm.tensor<6x10x6xi32>, %arg1: !spirv.arm.tensor<1x10x6x6xi32>) -> (!spirv.arm.tensor<6x10x6x6xi32>) {
  // expected-error @+1 {{op failed to verify that all of {input1, input2, output} have same rank}}
  %0 = spirv.Tosa.ArithmeticRightShift round = true, %arg0, %arg1 : !spirv.arm.tensor<6x10x6xi32>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi32>
}

spirv.ARM.Graph @arithmeticrightshift_input_element_types_not_matching(%arg0: !spirv.arm.tensor<6x10x6x6xi16>, %arg1: !spirv.arm.tensor<1x10x6x6xi32>) -> (!spirv.arm.tensor<6x10x6x6xi16>) {
  // expected-error @+1 {{op failed to verify that all of {input1, input2} have same element type}}
  %0 = spirv.Tosa.ArithmeticRightShift round = true, %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi16>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi16>
}

spirv.ARM.Graph @arithmeticrightshift_input_output_element_types_not_matching(%arg0: !spirv.arm.tensor<6x10x6x6xi32>, %arg1: !spirv.arm.tensor<1x10x6x6xi32>) -> (!spirv.arm.tensor<6x10x6x6xi16>) {
  // expected-error @+1 {{op failed to verify that all of {input1, output} have same element type}}
  %0 = spirv.Tosa.ArithmeticRightShift round = true, %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi32>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi16>
}

spirv.ARM.Graph @arithmeticrightshift_inputs_not_broadcastable(%arg0: !spirv.arm.tensor<6x10x6x6xi32>, %arg1: !spirv.arm.tensor<2x10x6x6xi32>) -> (!spirv.arm.tensor<6x10x6x6xi32>) {
  // expected-error @+1 {{op failed to verify that the shape of input1 and input2 are compatible for broadcasting and the broadcast shape is equal to the output shape}}
  %0 = spirv.Tosa.ArithmeticRightShift round = true, %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi32>, !spirv.arm.tensor<2x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi32>
}

spirv.ARM.Graph @arithmeticrightshift_output_shape_does_not_match_broadcast_shape(%arg0: !spirv.arm.tensor<6x10x6x6xi32>, %arg1: !spirv.arm.tensor<1x10x6x6xi32>) -> (!spirv.arm.tensor<1x10x6x6xi32>) {
  // expected-error @+1 {{op failed to verify that the shape of input1 and input2 are compatible for broadcasting and the broadcast shape is equal to the output shape}}
  %0 = spirv.Tosa.ArithmeticRightShift round = true, %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi32>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<1x10x6x6xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<1x10x6x6xi32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.BitwiseAnd
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @bitwiseand_input_ranks_not_matching(%arg0: !spirv.arm.tensor<6x10x6xi32>, %arg1: !spirv.arm.tensor<1x10x6x6xi32>) -> (!spirv.arm.tensor<6x10x6x6xi32>) {
  // expected-error @+1 {{op failed to verify that all of {input1, input2, output} have same rank}}
  %0 = spirv.Tosa.BitwiseAnd %arg0, %arg1 : !spirv.arm.tensor<6x10x6xi32>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi32>
}

spirv.ARM.Graph @bitwiseand_input_element_types_not_matching(%arg0: !spirv.arm.tensor<6x10x6x6xi16>, %arg1: !spirv.arm.tensor<1x10x6x6xi32>) -> (!spirv.arm.tensor<6x10x6x6xi16>) {
  // expected-error @+1 {{op failed to verify that all of {input1, input2} have same element type}}
  %0 = spirv.Tosa.BitwiseAnd %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi16>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi16>
}

spirv.ARM.Graph @bitwiseand_input_output_element_types_not_matching(%arg0: !spirv.arm.tensor<6x10x6x6xi32>, %arg1: !spirv.arm.tensor<1x10x6x6xi32>) -> (!spirv.arm.tensor<6x10x6x6xi16>) {
  // expected-error @+1 {{op failed to verify that all of {input1, output} have same element type}}
  %0 = spirv.Tosa.BitwiseAnd %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi32>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi16>
}

spirv.ARM.Graph @bitwiseand_inputs_not_broadcastable(%arg0: !spirv.arm.tensor<6x10x6x6xi32>, %arg1: !spirv.arm.tensor<2x10x6x6xi32>) -> (!spirv.arm.tensor<6x10x6x6xi32>) {
  // expected-error @+1 {{op failed to verify that the shape of input1 and input2 are compatible for broadcasting and the broadcast shape is equal to the output shape}}
  %0 = spirv.Tosa.BitwiseAnd %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi32>, !spirv.arm.tensor<2x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi32>
}

spirv.ARM.Graph @bitwiseand_output_shape_does_not_match_broadcast_shape(%arg0: !spirv.arm.tensor<6x10x6x6xi32>, %arg1: !spirv.arm.tensor<1x10x6x6xi32>) -> (!spirv.arm.tensor<1x10x6x6xi32>) {
  // expected-error @+1 {{op failed to verify that the shape of input1 and input2 are compatible for broadcasting and the broadcast shape is equal to the output shape}}
  %0 = spirv.Tosa.BitwiseAnd %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi32>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<1x10x6x6xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<1x10x6x6xi32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.BitwiseOr
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @bitwiseor_input_ranks_not_matching(%arg0: !spirv.arm.tensor<6x10x6xi32>, %arg1: !spirv.arm.tensor<1x10x6x6xi32>) -> (!spirv.arm.tensor<6x10x6x6xi32>) {
  // expected-error @+1 {{op failed to verify that all of {input1, input2, output} have same rank}}
  %0 = spirv.Tosa.BitwiseOr %arg0, %arg1 : !spirv.arm.tensor<6x10x6xi32>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi32>
}

spirv.ARM.Graph @bitwiseor_input_element_types_not_matching(%arg0: !spirv.arm.tensor<6x10x6x6xi16>, %arg1: !spirv.arm.tensor<1x10x6x6xi32>) -> (!spirv.arm.tensor<6x10x6x6xi16>) {
  // expected-error @+1 {{op failed to verify that all of {input1, input2} have same element type}}
  %0 = spirv.Tosa.BitwiseOr %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi16>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi16>
}

spirv.ARM.Graph @bitwiseor_input_output_element_types_not_matching(%arg0: !spirv.arm.tensor<6x10x6x6xi32>, %arg1: !spirv.arm.tensor<1x10x6x6xi32>) -> (!spirv.arm.tensor<6x10x6x6xi16>) {
  // expected-error @+1 {{op failed to verify that all of {input1, output} have same element type}}
  %0 = spirv.Tosa.BitwiseOr %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi32>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi16>
}

spirv.ARM.Graph @bitwiseor_inputs_not_broadcastable(%arg0: !spirv.arm.tensor<6x10x6x6xi32>, %arg1: !spirv.arm.tensor<2x10x6x6xi32>) -> (!spirv.arm.tensor<6x10x6x6xi32>) {
  // expected-error @+1 {{op failed to verify that the shape of input1 and input2 are compatible for broadcasting and the broadcast shape is equal to the output shape}}
  %0 = spirv.Tosa.BitwiseOr %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi32>, !spirv.arm.tensor<2x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi32>
}

spirv.ARM.Graph @bitwiseor_output_shape_does_not_match_broadcast_shape(%arg0: !spirv.arm.tensor<6x10x6x6xi32>, %arg1: !spirv.arm.tensor<1x10x6x6xi32>) -> (!spirv.arm.tensor<1x10x6x6xi32>) {
  // expected-error @+1 {{op failed to verify that the shape of input1 and input2 are compatible for broadcasting and the broadcast shape is equal to the output shape}}
  %0 = spirv.Tosa.BitwiseOr %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi32>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<1x10x6x6xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<1x10x6x6xi32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.BitwiseXor
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @bitwisexor_input_ranks_not_matching(%arg0: !spirv.arm.tensor<6x10x6xi32>, %arg1: !spirv.arm.tensor<1x10x6x6xi32>) -> (!spirv.arm.tensor<6x10x6x6xi32>) {
  // expected-error @+1 {{op failed to verify that all of {input1, input2, output} have same rank}}
  %0 = spirv.Tosa.BitwiseXor %arg0, %arg1 : !spirv.arm.tensor<6x10x6xi32>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi32>
}

spirv.ARM.Graph @bitwisexor_input_element_types_not_matching(%arg0: !spirv.arm.tensor<6x10x6x6xi16>, %arg1: !spirv.arm.tensor<1x10x6x6xi32>) -> (!spirv.arm.tensor<6x10x6x6xi16>) {
  // expected-error @+1 {{op failed to verify that all of {input1, input2} have same element type}}
  %0 = spirv.Tosa.BitwiseXor %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi16>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi16>
}

spirv.ARM.Graph @bitwisexor_input_output_element_types_not_matching(%arg0: !spirv.arm.tensor<6x10x6x6xi32>, %arg1: !spirv.arm.tensor<1x10x6x6xi32>) -> (!spirv.arm.tensor<6x10x6x6xi16>) {
  // expected-error @+1 {{op failed to verify that all of {input1, output} have same element type}}
  %0 = spirv.Tosa.BitwiseXor %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi32>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi16>
}

spirv.ARM.Graph @bitwisexor_inputs_not_broadcastable(%arg0: !spirv.arm.tensor<6x10x6x6xi32>, %arg1: !spirv.arm.tensor<2x10x6x6xi32>) -> (!spirv.arm.tensor<6x10x6x6xi32>) {
  // expected-error @+1 {{op failed to verify that the shape of input1 and input2 are compatible for broadcasting and the broadcast shape is equal to the output shape}}
  %0 = spirv.Tosa.BitwiseXor %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi32>, !spirv.arm.tensor<2x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi32>
}

spirv.ARM.Graph @bitwisexor_output_shape_does_not_match_broadcast_shape(%arg0: !spirv.arm.tensor<6x10x6x6xi32>, %arg1: !spirv.arm.tensor<1x10x6x6xi32>) -> (!spirv.arm.tensor<1x10x6x6xi32>) {
  // expected-error @+1 {{op failed to verify that the shape of input1 and input2 are compatible for broadcasting and the broadcast shape is equal to the output shape}}
  %0 = spirv.Tosa.BitwiseXor %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi32>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<1x10x6x6xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<1x10x6x6xi32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.IntDiv
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @intdiv_input_ranks_not_matching(%arg0: !spirv.arm.tensor<6x10x6xi32>, %arg1: !spirv.arm.tensor<1x10x6x6xi32>) -> (!spirv.arm.tensor<6x10x6x6xi32>) {
  // expected-error @+1 {{op failed to verify that all of {input1, input2, output} have same rank}}
  %0 = spirv.Tosa.IntDiv %arg0, %arg1 : !spirv.arm.tensor<6x10x6xi32>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi32>
}

spirv.ARM.Graph @intdiv_input_element_is_not_i32(%arg0: !spirv.arm.tensor<6x10x6x6xi16>, %arg1: !spirv.arm.tensor<1x10x6x6xi32>) -> (!spirv.arm.tensor<6x10x6x6xi32>) {
  // expected-error @+1 {{op operand #0 must be 1D/2D/3D/4D/5D/6D tensorArm of Int32 values, but got '!spirv.arm.tensor<6x10x6x6xi16>'}}
  %0 = spirv.Tosa.IntDiv %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi16>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi32>
}

spirv.ARM.Graph @intdiv_output_element_types_is_not_i32(%arg0: !spirv.arm.tensor<6x10x6x6xi32>, %arg1: !spirv.arm.tensor<1x10x6x6xi32>) -> (!spirv.arm.tensor<6x10x6x6xi16>) {
  // expected-error @+1 {{op result #0 must be 1D/2D/3D/4D/5D/6D tensorArm of Int32 values, but got '!spirv.arm.tensor<6x10x6x6xi16>'}}
  %0 = spirv.Tosa.IntDiv %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi32>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi16>
}

spirv.ARM.Graph @intdiv_inputs_not_broadcastable(%arg0: !spirv.arm.tensor<6x10x6x6xi32>, %arg1: !spirv.arm.tensor<2x10x6x6xi32>) -> (!spirv.arm.tensor<6x10x6x6xi32>) {
  // expected-error @+1 {{op failed to verify that the shape of input1 and input2 are compatible for broadcasting and the broadcast shape is equal to the output shape}}
  %0 = spirv.Tosa.IntDiv %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi32>, !spirv.arm.tensor<2x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi32>
}

spirv.ARM.Graph @intdiv_output_shape_does_not_match_broadcast_shape(%arg0: !spirv.arm.tensor<6x10x6x6xi32>, %arg1: !spirv.arm.tensor<1x10x6x6xi32>) -> (!spirv.arm.tensor<1x10x6x6xi32>) {
  // expected-error @+1 {{op failed to verify that the shape of input1 and input2 are compatible for broadcasting and the broadcast shape is equal to the output shape}}
  %0 = spirv.Tosa.IntDiv %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi32>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<1x10x6x6xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<1x10x6x6xi32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.LogicalAnd
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @logicaland_input_ranks_not_matching(%arg0: !spirv.arm.tensor<6x10x6xi1>, %arg1: !spirv.arm.tensor<1x10x6x6xi1>) -> (!spirv.arm.tensor<6x10x6x6xi1>) {
  // expected-error @+1 {{op failed to verify that all of {input1, input2, output} have same rank}}
  %0 = spirv.Tosa.LogicalAnd %arg0, %arg1 : !spirv.arm.tensor<6x10x6xi1>, !spirv.arm.tensor<1x10x6x6xi1> -> !spirv.arm.tensor<6x10x6x6xi1>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi1>
}

spirv.ARM.Graph @logicaland_input_element_types_not_bool(%arg0: !spirv.arm.tensor<6x10x6x6xi16>, %arg1: !spirv.arm.tensor<1x10x6x6xi1>) -> (!spirv.arm.tensor<6x10x6x6xi1>) {
  // expected-error @+1 {{op operand #0 must be 1D/2D/3D/4D/5D/6D tensorArm of bool values, but got '!spirv.arm.tensor<6x10x6x6xi16>'}}
  %0 = spirv.Tosa.LogicalAnd %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi16>, !spirv.arm.tensor<1x10x6x6xi1> -> !spirv.arm.tensor<6x10x6x6xi1>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi1>
}

spirv.ARM.Graph @logicaland_output_element_types_not_bool(%arg0: !spirv.arm.tensor<6x10x6x6xi1>, %arg1: !spirv.arm.tensor<1x10x6x6xi1>) -> (!spirv.arm.tensor<6x10x6x6xi16>) {
  // expected-error @+1 {{ op result #0 must be 1D/2D/3D/4D/5D/6D tensorArm of bool values, but got '!spirv.arm.tensor<6x10x6x6xi16>'}}
  %0 = spirv.Tosa.LogicalAnd %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi1>, !spirv.arm.tensor<1x10x6x6xi1> -> !spirv.arm.tensor<6x10x6x6xi16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi16>
}

spirv.ARM.Graph @logicaland_inputs_not_broadcastable(%arg0: !spirv.arm.tensor<6x10x6x6xi1>, %arg1: !spirv.arm.tensor<2x10x6x6xi1>) -> (!spirv.arm.tensor<6x10x6x6xi1>) {
  // expected-error @+1 {{op failed to verify that the shape of input1 and input2 are compatible for broadcasting and the broadcast shape is equal to the output shape}}
  %0 = spirv.Tosa.LogicalAnd %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi1>, !spirv.arm.tensor<2x10x6x6xi1> -> !spirv.arm.tensor<6x10x6x6xi1>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi1>
}

spirv.ARM.Graph @logicaland_output_shape_does_not_match_broadcast_shape(%arg0: !spirv.arm.tensor<6x10x6x6xi1>, %arg1: !spirv.arm.tensor<1x10x6x6xi1>) -> (!spirv.arm.tensor<1x10x6x6xi1>) {
  // expected-error @+1 {{op failed to verify that the shape of input1 and input2 are compatible for broadcasting and the broadcast shape is equal to the output shape}}
  %0 = spirv.Tosa.LogicalAnd %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi1>, !spirv.arm.tensor<1x10x6x6xi1> -> !spirv.arm.tensor<1x10x6x6xi1>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<1x10x6x6xi1>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.LogicalLeftShift
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @logicalleftshift_input_ranks_not_matching(%arg0: !spirv.arm.tensor<6x10x6xi32>, %arg1: !spirv.arm.tensor<1x10x6x6xi32>) -> (!spirv.arm.tensor<6x10x6x6xi32>) {
  // expected-error @+1 {{op failed to verify that all of {input1, input2, output} have same rank}}
  %0 = spirv.Tosa.LogicalLeftShift %arg0, %arg1 : !spirv.arm.tensor<6x10x6xi32>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi32>
}

spirv.ARM.Graph @logicalleftshift_input_element_types_not_matching(%arg0: !spirv.arm.tensor<6x10x6x6xi16>, %arg1: !spirv.arm.tensor<1x10x6x6xi32>) -> (!spirv.arm.tensor<6x10x6x6xi16>) {
  // expected-error @+1 {{op failed to verify that all of {input1, input2} have same element type}}
  %0 = spirv.Tosa.LogicalLeftShift %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi16>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi16>
}

spirv.ARM.Graph @logicalleftshift_input_output_element_types_not_matching(%arg0: !spirv.arm.tensor<6x10x6x6xi32>, %arg1: !spirv.arm.tensor<1x10x6x6xi32>) -> (!spirv.arm.tensor<6x10x6x6xi16>) {
  // expected-error @+1 {{op failed to verify that all of {input1, output} have same element type}}
  %0 = spirv.Tosa.LogicalLeftShift %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi32>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi16>
}

spirv.ARM.Graph @logicalleftshift_inputs_not_broadcastable(%arg0: !spirv.arm.tensor<6x10x6x6xi32>, %arg1: !spirv.arm.tensor<2x10x6x6xi32>) -> (!spirv.arm.tensor<6x10x6x6xi32>) {
  // expected-error @+1 {{op failed to verify that the shape of input1 and input2 are compatible for broadcasting and the broadcast shape is equal to the output shape}}
  %0 = spirv.Tosa.LogicalLeftShift %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi32>, !spirv.arm.tensor<2x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi32>
}

spirv.ARM.Graph @logicalleftshift_output_shape_does_not_match_broadcast_shape(%arg0: !spirv.arm.tensor<6x10x6x6xi32>, %arg1: !spirv.arm.tensor<1x10x6x6xi32>) -> (!spirv.arm.tensor<1x10x6x6xi32>) {
  // expected-error @+1 {{op failed to verify that the shape of input1 and input2 are compatible for broadcasting and the broadcast shape is equal to the output shape}}
  %0 = spirv.Tosa.LogicalLeftShift %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi32>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<1x10x6x6xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<1x10x6x6xi32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.LogicalRightShift
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @logicalrightshift_input_ranks_not_matching(%arg0: !spirv.arm.tensor<6x10x6xi32>, %arg1: !spirv.arm.tensor<1x10x6x6xi32>) -> (!spirv.arm.tensor<6x10x6x6xi32>) {
  // expected-error @+1 {{op failed to verify that all of {input1, input2, output} have same rank}}
  %0 = spirv.Tosa.LogicalRightShift %arg0, %arg1 : !spirv.arm.tensor<6x10x6xi32>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi32>
}

spirv.ARM.Graph @logicalrightshift_input_element_types_not_matching(%arg0: !spirv.arm.tensor<6x10x6x6xi16>, %arg1: !spirv.arm.tensor<1x10x6x6xi32>) -> (!spirv.arm.tensor<6x10x6x6xi16>) {
  // expected-error @+1 {{op failed to verify that all of {input1, input2} have same element type}}
  %0 = spirv.Tosa.LogicalRightShift %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi16>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi16>
}

spirv.ARM.Graph @logicalrightshift_input_output_element_types_not_matching(%arg0: !spirv.arm.tensor<6x10x6x6xi32>, %arg1: !spirv.arm.tensor<1x10x6x6xi32>) -> (!spirv.arm.tensor<6x10x6x6xi16>) {
  // expected-error @+1 {{op failed to verify that all of {input1, output} have same element type}}
  %0 = spirv.Tosa.LogicalRightShift %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi32>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi16>
}

spirv.ARM.Graph @logicalrightshift_inputs_not_broadcastable(%arg0: !spirv.arm.tensor<6x10x6x6xi32>, %arg1: !spirv.arm.tensor<2x10x6x6xi32>) -> (!spirv.arm.tensor<6x10x6x6xi32>) {
  // expected-error @+1 {{op failed to verify that the shape of input1 and input2 are compatible for broadcasting and the broadcast shape is equal to the output shape}}
  %0 = spirv.Tosa.LogicalRightShift %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi32>, !spirv.arm.tensor<2x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi32>
}

spirv.ARM.Graph @logicalrightshift_output_shape_does_not_match_broadcast_shape(%arg0: !spirv.arm.tensor<6x10x6x6xi32>, %arg1: !spirv.arm.tensor<1x10x6x6xi32>) -> (!spirv.arm.tensor<1x10x6x6xi32>) {
  // expected-error @+1 {{op failed to verify that the shape of input1 and input2 are compatible for broadcasting and the broadcast shape is equal to the output shape}}
  %0 = spirv.Tosa.LogicalRightShift %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi32>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<1x10x6x6xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<1x10x6x6xi32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.LogicalOr
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @logicalor_input_ranks_not_matching(%arg0: !spirv.arm.tensor<6x10x6xi1>, %arg1: !spirv.arm.tensor<1x10x6x6xi1>) -> (!spirv.arm.tensor<6x10x6x6xi1>) {
  // expected-error @+1 {{op failed to verify that all of {input1, input2, output} have same rank}}
  %0 = spirv.Tosa.LogicalOr %arg0, %arg1 : !spirv.arm.tensor<6x10x6xi1>, !spirv.arm.tensor<1x10x6x6xi1> -> !spirv.arm.tensor<6x10x6x6xi1>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi1>
}

spirv.ARM.Graph @logicalor_input_element_types_not_bool(%arg0: !spirv.arm.tensor<6x10x6x6xi16>, %arg1: !spirv.arm.tensor<1x10x6x6xi1>) -> (!spirv.arm.tensor<6x10x6x6xi1>) {
  // expected-error @+1 {{op operand #0 must be 1D/2D/3D/4D/5D/6D tensorArm of bool values, but got '!spirv.arm.tensor<6x10x6x6xi16>'}}
  %0 = spirv.Tosa.LogicalOr %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi16>, !spirv.arm.tensor<1x10x6x6xi1> -> !spirv.arm.tensor<6x10x6x6xi1>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi1>
}

spirv.ARM.Graph @logicalor_output_element_types_not_bool(%arg0: !spirv.arm.tensor<6x10x6x6xi1>, %arg1: !spirv.arm.tensor<1x10x6x6xi1>) -> (!spirv.arm.tensor<6x10x6x6xi16>) {
  // expected-error @+1 {{ op result #0 must be 1D/2D/3D/4D/5D/6D tensorArm of bool values, but got '!spirv.arm.tensor<6x10x6x6xi16>'}}
  %0 = spirv.Tosa.LogicalOr %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi1>, !spirv.arm.tensor<1x10x6x6xi1> -> !spirv.arm.tensor<6x10x6x6xi16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi16>
}

spirv.ARM.Graph @logicalor_inputs_not_broadcastable(%arg0: !spirv.arm.tensor<6x10x6x6xi1>, %arg1: !spirv.arm.tensor<2x10x6x6xi1>) -> (!spirv.arm.tensor<6x10x6x6xi1>) {
  // expected-error @+1 {{op failed to verify that the shape of input1 and input2 are compatible for broadcasting and the broadcast shape is equal to the output shape}}
  %0 = spirv.Tosa.LogicalOr %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi1>, !spirv.arm.tensor<2x10x6x6xi1> -> !spirv.arm.tensor<6x10x6x6xi1>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi1>
}

spirv.ARM.Graph @logicalor_output_shape_does_not_match_broadcast_shape(%arg0: !spirv.arm.tensor<6x10x6x6xi1>, %arg1: !spirv.arm.tensor<1x10x6x6xi1>) -> (!spirv.arm.tensor<1x10x6x6xi1>) {
  // expected-error @+1 {{op failed to verify that the shape of input1 and input2 are compatible for broadcasting and the broadcast shape is equal to the output shape}}
  %0 = spirv.Tosa.LogicalOr %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi1>, !spirv.arm.tensor<1x10x6x6xi1> -> !spirv.arm.tensor<1x10x6x6xi1>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<1x10x6x6xi1>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.LogicalXor
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @logicalxor_input_ranks_not_matching(%arg0: !spirv.arm.tensor<6x10x6xi1>, %arg1: !spirv.arm.tensor<1x10x6x6xi1>) -> (!spirv.arm.tensor<6x10x6x6xi1>) {
  // expected-error @+1 {{op failed to verify that all of {input1, input2, output} have same rank}}
  %0 = spirv.Tosa.LogicalXor %arg0, %arg1 : !spirv.arm.tensor<6x10x6xi1>, !spirv.arm.tensor<1x10x6x6xi1> -> !spirv.arm.tensor<6x10x6x6xi1>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi1>
}

spirv.ARM.Graph @logicalxor_input_element_types_not_bool(%arg0: !spirv.arm.tensor<6x10x6x6xi16>, %arg1: !spirv.arm.tensor<1x10x6x6xi1>) -> (!spirv.arm.tensor<6x10x6x6xi1>) {
  // expected-error @+1 {{op operand #0 must be 1D/2D/3D/4D/5D/6D tensorArm of bool values, but got '!spirv.arm.tensor<6x10x6x6xi16>'}}
  %0 = spirv.Tosa.LogicalXor %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi16>, !spirv.arm.tensor<1x10x6x6xi1> -> !spirv.arm.tensor<6x10x6x6xi1>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi1>
}

spirv.ARM.Graph @logicalxor_output_element_types_not_bool(%arg0: !spirv.arm.tensor<6x10x6x6xi1>, %arg1: !spirv.arm.tensor<1x10x6x6xi1>) -> (!spirv.arm.tensor<6x10x6x6xi16>) {
  // expected-error @+1 {{ op result #0 must be 1D/2D/3D/4D/5D/6D tensorArm of bool values, but got '!spirv.arm.tensor<6x10x6x6xi16>'}}
  %0 = spirv.Tosa.LogicalXor %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi1>, !spirv.arm.tensor<1x10x6x6xi1> -> !spirv.arm.tensor<6x10x6x6xi16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi16>
}

spirv.ARM.Graph @logicalxor_inputs_not_broadcastable(%arg0: !spirv.arm.tensor<6x10x6x6xi1>, %arg1: !spirv.arm.tensor<2x10x6x6xi1>) -> (!spirv.arm.tensor<6x10x6x6xi1>) {
  // expected-error @+1 {{op failed to verify that the shape of input1 and input2 are compatible for broadcasting and the broadcast shape is equal to the output shape}}
  %0 = spirv.Tosa.LogicalXor %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi1>, !spirv.arm.tensor<2x10x6x6xi1> -> !spirv.arm.tensor<6x10x6x6xi1>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi1>
}

spirv.ARM.Graph @logicalxor_output_shape_does_not_match_broadcast_shape(%arg0: !spirv.arm.tensor<6x10x6x6xi1>, %arg1: !spirv.arm.tensor<1x10x6x6xi1>) -> (!spirv.arm.tensor<1x10x6x6xi1>) {
  // expected-error @+1 {{op failed to verify that the shape of input1 and input2 are compatible for broadcasting and the broadcast shape is equal to the output shape}}
  %0 = spirv.Tosa.LogicalXor %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi1>, !spirv.arm.tensor<1x10x6x6xi1> -> !spirv.arm.tensor<1x10x6x6xi1>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<1x10x6x6xi1>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Maximum
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @maximum_input_ranks_not_matching(%arg0: !spirv.arm.tensor<6x10x6xi32>, %arg1: !spirv.arm.tensor<1x10x6x6xi32>) -> (!spirv.arm.tensor<6x10x6x6xi32>) {
  // expected-error @+1 {{op failed to verify that all of {input1, input2, output} have same rank}}
  %0 = spirv.Tosa.Maximum nan_mode = <Propagate>, %arg0, %arg1 : !spirv.arm.tensor<6x10x6xi32>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi32>
}

spirv.ARM.Graph @maximum_input_element_types_not_matching(%arg0: !spirv.arm.tensor<6x10x6x6xf16>, %arg1: !spirv.arm.tensor<1x10x6x6xf32>) -> (!spirv.arm.tensor<6x10x6x6xf16>) {
  // expected-error @+1 {{op failed to verify that all of {input1, input2} have same element type}}
  %0 = spirv.Tosa.Maximum nan_mode = <Propagate>, %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xf16>, !spirv.arm.tensor<1x10x6x6xf32> -> !spirv.arm.tensor<6x10x6x6xf16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xf16>
}

spirv.ARM.Graph @maximum_input_output_element_types_not_matching(%arg0: !spirv.arm.tensor<6x10x6x6xf32>, %arg1: !spirv.arm.tensor<1x10x6x6xf32>) -> (!spirv.arm.tensor<6x10x6x6xf16>) {
  // expected-error @+1 {{op failed to verify that all of {input1, output} have same element type}}
  %0 = spirv.Tosa.Maximum nan_mode = <Propagate>, %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xf32>, !spirv.arm.tensor<1x10x6x6xf32> -> !spirv.arm.tensor<6x10x6x6xf16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xf16>
}

spirv.ARM.Graph @maximum_inputs_not_broadcastable(%arg0: !spirv.arm.tensor<6x10x6x6xi32>, %arg1: !spirv.arm.tensor<2x10x6x6xi32>) -> (!spirv.arm.tensor<6x10x6x6xi32>) {
  // expected-error @+1 {{op failed to verify that the shape of input1 and input2 are compatible for broadcasting and the broadcast shape is equal to the output shape}}
  %0 = spirv.Tosa.Maximum nan_mode = <Propagate>, %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi32>, !spirv.arm.tensor<2x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi32>
}

spirv.ARM.Graph @maximum_output_shape_does_not_match_broadcast_shape(%arg0: !spirv.arm.tensor<6x10x6x6xi32>, %arg1: !spirv.arm.tensor<1x10x6x6xi32>) -> (!spirv.arm.tensor<1x10x6x6xi32>) {
  // expected-error @+1 {{op failed to verify that the shape of input1 and input2 are compatible for broadcasting and the broadcast shape is equal to the output shape}}
  %0 = spirv.Tosa.Maximum nan_mode = <Propagate>, %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi32>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<1x10x6x6xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<1x10x6x6xi32>
}

spirv.ARM.Graph @maximum_integer_input1_must_be_i32(%arg0: !spirv.arm.tensor<6x10x6x6xi8>, %arg1: !spirv.arm.tensor<1x10x6x6xi32>) -> (!spirv.arm.tensor<6x10x6x6xi32>) {
  // expected-error @+1 {{op failed to verify that if input1 has type integer then input1 must have a type in [32-bit signless integer]}}
  %0 = spirv.Tosa.Maximum nan_mode = <Propagate>, %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi8>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi32>
}

spirv.ARM.Graph @maximum_integer_input2_must_be_i32(%arg0: !spirv.arm.tensor<6x10x6x6xi32>, %arg1: !spirv.arm.tensor<1x10x6x6xi8>) -> (!spirv.arm.tensor<6x10x6x6xi32>) {
  // expected-error @+1 {{op failed to verify that if input2 has type integer then input2 must have a type in [32-bit signless integer]}}
  %0 = spirv.Tosa.Maximum nan_mode = <Propagate>, %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi32>, !spirv.arm.tensor<1x10x6x6xi8> -> !spirv.arm.tensor<6x10x6x6xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi32>
}

spirv.ARM.Graph @maximum_integer_output_must_be_i32(%arg0: !spirv.arm.tensor<6x10x6x6xi32>, %arg1: !spirv.arm.tensor<1x10x6x6xi32>) -> (!spirv.arm.tensor<6x10x6x6xi8>) {
  // expected-error @+1 {{op failed to verify that if output has type integer then output must have a type in [32-bit signless integer]}}
  %0 = spirv.Tosa.Maximum nan_mode = <Propagate>, %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi32>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi8>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi8>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Minimum
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @minimum_input_ranks_not_matching(%arg0: !spirv.arm.tensor<6x10x6xi32>, %arg1: !spirv.arm.tensor<1x10x6x6xi32>) -> (!spirv.arm.tensor<6x10x6x6xi32>) {
  // expected-error @+1 {{op failed to verify that all of {input1, input2, output} have same rank}}
  %0 = spirv.Tosa.Minimum nan_mode = <Propagate>, %arg0, %arg1 : !spirv.arm.tensor<6x10x6xi32>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi32>
}

spirv.ARM.Graph @minimum_input_element_types_not_matching(%arg0: !spirv.arm.tensor<6x10x6x6xf16>, %arg1: !spirv.arm.tensor<1x10x6x6xf32>) -> (!spirv.arm.tensor<6x10x6x6xf16>) {
  // expected-error @+1 {{op failed to verify that all of {input1, input2} have same element type}}
  %0 = spirv.Tosa.Minimum nan_mode = <Propagate>, %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xf16>, !spirv.arm.tensor<1x10x6x6xf32> -> !spirv.arm.tensor<6x10x6x6xf16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xf16>
}

spirv.ARM.Graph @minimum_input_output_element_types_not_matching(%arg0: !spirv.arm.tensor<6x10x6x6xf32>, %arg1: !spirv.arm.tensor<1x10x6x6xf32>) -> (!spirv.arm.tensor<6x10x6x6xf16>) {
  // expected-error @+1 {{op failed to verify that all of {input1, output} have same element type}}
  %0 = spirv.Tosa.Minimum nan_mode = <Propagate>, %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xf32>, !spirv.arm.tensor<1x10x6x6xf32> -> !spirv.arm.tensor<6x10x6x6xf16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xf16>
}

spirv.ARM.Graph @minimum_inputs_not_broadcastable(%arg0: !spirv.arm.tensor<6x10x6x6xi32>, %arg1: !spirv.arm.tensor<2x10x6x6xi32>) -> (!spirv.arm.tensor<6x10x6x6xi32>) {
  // expected-error @+1 {{op failed to verify that the shape of input1 and input2 are compatible for broadcasting and the broadcast shape is equal to the output shape}}
  %0 = spirv.Tosa.Minimum nan_mode = <Propagate>, %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi32>, !spirv.arm.tensor<2x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi32>
}

spirv.ARM.Graph @minimum_output_shape_does_not_match_broadcast_shape(%arg0: !spirv.arm.tensor<6x10x6x6xi32>, %arg1: !spirv.arm.tensor<1x10x6x6xi32>) -> (!spirv.arm.tensor<1x10x6x6xi32>) {
  // expected-error @+1 {{op failed to verify that the shape of input1 and input2 are compatible for broadcasting and the broadcast shape is equal to the output shape}}
  %0 = spirv.Tosa.Minimum nan_mode = <Propagate>, %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi32>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<1x10x6x6xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<1x10x6x6xi32>
}

spirv.ARM.Graph @minimum_integer_input1_must_be_i32(%arg0: !spirv.arm.tensor<6x10x6x6xi8>, %arg1: !spirv.arm.tensor<1x10x6x6xi32>) -> (!spirv.arm.tensor<6x10x6x6xi32>) {
  // expected-error @+1 {{op failed to verify that if input1 has type integer then input1 must have a type in [32-bit signless integer]}}
  %0 = spirv.Tosa.Minimum nan_mode = <Propagate>, %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi8>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi32>
}

spirv.ARM.Graph @minimum_integer_input2_must_be_i32(%arg0: !spirv.arm.tensor<6x10x6x6xi32>, %arg1: !spirv.arm.tensor<1x10x6x6xi8>) -> (!spirv.arm.tensor<6x10x6x6xi32>) {
  // expected-error @+1 {{op failed to verify that if input2 has type integer then input2 must have a type in [32-bit signless integer]}}
  %0 = spirv.Tosa.Minimum nan_mode = <Propagate>, %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi32>, !spirv.arm.tensor<1x10x6x6xi8> -> !spirv.arm.tensor<6x10x6x6xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi32>
}

spirv.ARM.Graph @minimum_integer_output_must_be_i32(%arg0: !spirv.arm.tensor<6x10x6x6xi32>, %arg1: !spirv.arm.tensor<1x10x6x6xi32>) -> (!spirv.arm.tensor<6x10x6x6xi8>) {
  // expected-error @+1 {{op failed to verify that if output has type integer then output must have a type in [32-bit signless integer]}}
  %0 = spirv.Tosa.Minimum nan_mode = <Propagate>, %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi32>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi8>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi8>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Mul
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @mul_input_ranks_not_matching(%arg0: !spirv.arm.tensor<34x21xi8>, %arg1: !spirv.arm.tensor<34x21x1xi8>) -> (!spirv.arm.tensor<34x21x39xi32>) {
  %0 = spirv.Constant dense<31> : !spirv.arm.tensor<1xi8>
  // expected-error @+1 {{op failed to verify that all of {input1, input2, output} have same rank}}
  %1 = spirv.Tosa.Mul %arg0, %arg1, %0 : !spirv.arm.tensor<34x21xi8>, !spirv.arm.tensor<34x21x1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<34x21x39xi32>
  spirv.ARM.GraphOutputs %1 : !spirv.arm.tensor<34x21x39xi32>
}

spirv.ARM.Graph @mul_input_element_types_not_matching(%arg0: !spirv.arm.tensor<34x21x39xi8>, %arg1: !spirv.arm.tensor<34x21x1xi16>) -> (!spirv.arm.tensor<34x21x39xi32>) {
  %0 = spirv.Constant dense<31> : !spirv.arm.tensor<1xi8>
  // expected-error @+1 {{op failed to verify that all of {input1, input2} have same element type}}
  %1 = spirv.Tosa.Mul %arg0, %arg1, %0 : !spirv.arm.tensor<34x21x39xi8>, !spirv.arm.tensor<34x21x1xi16>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<34x21x39xi32>
  spirv.ARM.GraphOutputs %1 : !spirv.arm.tensor<34x21x39xi32>
}

spirv.ARM.Graph @mul_inputs_not_broadcastable(%arg0: !spirv.arm.tensor<34x21x39xi8>, %arg1: !spirv.arm.tensor<34x20x1xi8>) -> (!spirv.arm.tensor<34x21x39xi32>) {
  %0 = spirv.Constant dense<31> : !spirv.arm.tensor<1xi8>
  // expected-error @+1 {{op failed to verify that the shape of input1 and input2 are compatible for broadcasting and the broadcast shape is equal to the output shape}}
  %1 = spirv.Tosa.Mul %arg0, %arg1, %0 : !spirv.arm.tensor<34x21x39xi8>, !spirv.arm.tensor<34x20x1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<34x21x39xi32>
  spirv.ARM.GraphOutputs %1 : !spirv.arm.tensor<34x21x39xi32>
}

spirv.ARM.Graph @mul_output_shape_does_not_match_broadcast_shape(%arg0: !spirv.arm.tensor<34x21x39xi8>, %arg1: !spirv.arm.tensor<34x21x1xi8>) -> (!spirv.arm.tensor<34x21x1xi32>) {
  %0 = spirv.Constant dense<31> : !spirv.arm.tensor<1xi8>
  // expected-error @+1 {{op failed to verify that the shape of input1 and input2 are compatible for broadcasting and the broadcast shape is equal to the output shape}}
  %1 = spirv.Tosa.Mul %arg0, %arg1, %0 : !spirv.arm.tensor<34x21x39xi8>, !spirv.arm.tensor<34x21x1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<34x21x1xi32>
  spirv.ARM.GraphOutputs %1 : !spirv.arm.tensor<34x21x1xi32>
}

spirv.ARM.Graph @mul_ouput_must_have_i32_as_element_type(%arg0: !spirv.arm.tensor<34x21x39xi8>, %arg1: !spirv.arm.tensor<34x21x1xi8>) -> (!spirv.arm.tensor<34x21x39xi16>) {
  %0 = spirv.Constant dense<31> : !spirv.arm.tensor<1xi8>
  // expected-error @+1 {{op failed to verify that if input1 has type integer then output must have a type in [32-bit signless integer]}}
  %1 = spirv.Tosa.Mul %arg0, %arg1, %0 : !spirv.arm.tensor<34x21x39xi8>, !spirv.arm.tensor<34x21x1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<34x21x39xi16>
  spirv.ARM.GraphOutputs %1 : !spirv.arm.tensor<34x21x39xi16>
}

spirv.ARM.Graph @mul_input_with_element_type_f16_must_produce_an_output_with_element_type_f16(%arg0: !spirv.arm.tensor<57x1x55xf16>, %arg1: !spirv.arm.tensor<57x37x55xf16>) -> (!spirv.arm.tensor<57x37x55xf32>) {
  %0 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi8>
  // expected-error @+1 {{op failed to verify that if input1 has type 16-bit float then output must have a type in [16-bit float]}}
  %1 = spirv.Tosa.Mul %arg0, %arg1, %0 : !spirv.arm.tensor<57x1x55xf16>, !spirv.arm.tensor<57x37x55xf16>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<57x37x55xf32>
  spirv.ARM.GraphOutputs %1 : !spirv.arm.tensor<57x37x55xf32>
}

spirv.ARM.Graph @mul_input_with_element_type_f32_must_produce_an_output_with_element_type_f32(%arg0: !spirv.arm.tensor<57x1x55xf32>, %arg1: !spirv.arm.tensor<57x37x55xf32>) -> (!spirv.arm.tensor<57x37x55xf16>) {
  %0 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi8>
  // expected-error @+1 {{op failed to verify that if input1 has type 32-bit float then output must have a type in [32-bit float]}}
  %1 = spirv.Tosa.Mul %arg0, %arg1, %0 : !spirv.arm.tensor<57x1x55xf32>, !spirv.arm.tensor<57x37x55xf32>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<57x37x55xf16>
  spirv.ARM.GraphOutputs %1 : !spirv.arm.tensor<57x37x55xf16>
}

spirv.ARM.Graph @mul_input_with_element_type_bf16_must_produce_an_output_with_element_type_bf16(%arg0: !spirv.arm.tensor<57x1x55xbf16>, %arg1: !spirv.arm.tensor<57x37x55xbf16>) -> (!spirv.arm.tensor<57x37x55xf32>) {
  %0 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi8>
  // expected-error @+1 {{op failed to verify that if input1 has type bfloat16 type then output must have a type in [bfloat16 type]}}
  %1 = spirv.Tosa.Mul %arg0, %arg1, %0 : !spirv.arm.tensor<57x1x55xbf16>, !spirv.arm.tensor<57x37x55xbf16>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<57x37x55xf32>
  spirv.ARM.GraphOutputs %1 : !spirv.arm.tensor<57x37x55xf32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Pow
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @pow_input_ranks_not_matching(%arg0: !spirv.arm.tensor<52x53xf16>, %arg1: !spirv.arm.tensor<44x52x53xf16>) -> (!spirv.arm.tensor<44x52x53xf16>) {
  // expected-error @+1 {{op failed to verify that all of {input1, input2, output} have same rank}}
  %0 = spirv.Tosa.Pow %arg0, %arg1 : !spirv.arm.tensor<52x53xf16>, !spirv.arm.tensor<44x52x53xf16> -> !spirv.arm.tensor<44x52x53xf16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<44x52x53xf16>
}

spirv.ARM.Graph @pow_inputs_with_non_matching_element_types(%arg0: !spirv.arm.tensor<1x52x53xf16>, %arg1: !spirv.arm.tensor<44x52x53xf32>) -> (!spirv.arm.tensor<44x52x53xf16>) {
  // expected-error @+1 {{op failed to verify that all of {input1, input2} have same element type}}
  %0 = spirv.Tosa.Pow %arg0, %arg1 : !spirv.arm.tensor<1x52x53xf16>, !spirv.arm.tensor<44x52x53xf32> -> !spirv.arm.tensor<44x52x53xf16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<44x52x53xf16>
}

spirv.ARM.Graph @pow_input_output_element_types_not_matching(%arg0: !spirv.arm.tensor<1x52x53xf16>, %arg1: !spirv.arm.tensor<44x52x53xf16>) -> (!spirv.arm.tensor<44x52x53xf32>) {
  // expected-error @+1 {{op failed to verify that all of {input1, output} have same element type}}
  %0 = spirv.Tosa.Pow %arg0, %arg1 : !spirv.arm.tensor<1x52x53xf16>, !spirv.arm.tensor<44x52x53xf16> -> !spirv.arm.tensor<44x52x53xf32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<44x52x53xf32>
}

spirv.ARM.Graph @pow_inputs_not_broadcastable(%arg0: !spirv.arm.tensor<6x10x6x6xf32>, %arg1: !spirv.arm.tensor<2x10x6x6xf32>) -> (!spirv.arm.tensor<6x10x6x6xf32>) {
  // expected-error @+1 {{op failed to verify that the shape of input1 and input2 are compatible for broadcasting and the broadcast shape is equal to the output shape}}
  %0 = spirv.Tosa.Pow %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xf32>, !spirv.arm.tensor<2x10x6x6xf32> -> !spirv.arm.tensor<6x10x6x6xf32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xf32>
}

spirv.ARM.Graph @pow_output_shape_does_not_match_broadcast_shape(%arg0: !spirv.arm.tensor<6x10x6x6xf32>, %arg1: !spirv.arm.tensor<1x10x6x6xf32>) -> (!spirv.arm.tensor<1x10x6x6xf32>) {
  // expected-error @+1 {{op failed to verify that the shape of input1 and input2 are compatible for broadcasting and the broadcast shape is equal to the output shape}}
  %0 = spirv.Tosa.Pow %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xf32>, !spirv.arm.tensor<1x10x6x6xf32> -> !spirv.arm.tensor<1x10x6x6xf32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<1x10x6x6xf32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Sub
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @sub_input_ranks_not_matching(%arg0: !spirv.arm.tensor<6x10x6xi32>, %arg1: !spirv.arm.tensor<1x10x6x6xi32>) -> (!spirv.arm.tensor<6x10x6x6xi32>) {
  // expected-error @+1 {{op failed to verify that all of {input1, input2, output} have same rank}}
  %0 = spirv.Tosa.Sub %arg0, %arg1 : !spirv.arm.tensor<6x10x6xi32>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi32>
}

spirv.ARM.Graph @sub_input_element_types_not_matching(%arg0: !spirv.arm.tensor<6x10x6x6xf16>, %arg1: !spirv.arm.tensor<1x10x6x6xf32>) -> (!spirv.arm.tensor<6x10x6x6xf32>) {
  // expected-error @+1 {{op failed to verify that all of {input1, output} have same element type}}
  %0 = spirv.Tosa.Sub %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xf16>, !spirv.arm.tensor<1x10x6x6xf32> -> !spirv.arm.tensor<6x10x6x6xf32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xf32>
}

spirv.ARM.Graph @sub_input1_input2_element_types_not_matching(%arg0: !spirv.arm.tensor<6x10x6x6xf16>, %arg1: !spirv.arm.tensor<1x10x6x6xf32>) -> (!spirv.arm.tensor<6x10x6x6xf16>) {
  // expected-error @+1 {{op failed to verify that all of {input1, input2} have same element type}}
  %0 = spirv.Tosa.Sub %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xf16>, !spirv.arm.tensor<1x10x6x6xf32> -> !spirv.arm.tensor<6x10x6x6xf16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xf16>
}

spirv.ARM.Graph @sub_inputs_not_broadcastable(%arg0: !spirv.arm.tensor<6x10x6x6xi32>, %arg1: !spirv.arm.tensor<2x10x6x6xi32>) -> (!spirv.arm.tensor<6x10x6x6xi32>) {
  // expected-error @+1 {{op failed to verify that the shape of input1 and input2 are compatible for broadcasting and the broadcast shape is equal to the output shape}}
  %0 = spirv.Tosa.Sub %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi32>, !spirv.arm.tensor<2x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi32>
}

spirv.ARM.Graph @sub_output_shape_does_not_match_broadcast_shape(%arg0: !spirv.arm.tensor<6x10x6x6xi32>, %arg1: !spirv.arm.tensor<1x10x6x6xi32>) -> (!spirv.arm.tensor<1x10x6x6xi32>) {
  // expected-error @+1 {{op failed to verify that the shape of input1 and input2 are compatible for broadcasting and the broadcast shape is equal to the output shape}}
  %0 = spirv.Tosa.Sub %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi32>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<1x10x6x6xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<1x10x6x6xi32>
}

spirv.ARM.Graph @sub_integer_input1_must_be_i32(%arg0: !spirv.arm.tensor<6x10x6x6xi8>, %arg1: !spirv.arm.tensor<1x10x6x6xi32>) -> (!spirv.arm.tensor<6x10x6x6xi32>) {
  // expected-error @+1 {{op failed to verify that if input1 has type integer then input1 must have a type in [32-bit signless integer]}}
  %0 = spirv.Tosa.Sub %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi8>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi32>
}

spirv.ARM.Graph @sub_integer_input2_must_be_i32(%arg0: !spirv.arm.tensor<6x10x6x6xi32>, %arg1: !spirv.arm.tensor<1x10x6x6xi8>) -> (!spirv.arm.tensor<6x10x6x6xi32>) {
  // expected-error @+1 {{op failed to verify that if input2 has type integer then input2 must have a type in [32-bit signless integer]}}
  %0 = spirv.Tosa.Sub %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi32>, !spirv.arm.tensor<1x10x6x6xi8> -> !spirv.arm.tensor<6x10x6x6xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi32>
}

spirv.ARM.Graph @sub_integer_output_must_be_i32(%arg0: !spirv.arm.tensor<6x10x6x6xi32>, %arg1: !spirv.arm.tensor<1x10x6x6xi32>) -> (!spirv.arm.tensor<6x10x6x6xi8>) {
  // expected-error @+1 {{op failed to verify that if output has type integer then output must have a type in [32-bit signless integer]}}
  %0 = spirv.Tosa.Sub %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi32>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi8>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi8>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Table
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @table_input_and_table_must_have_same_element_type(%arg0: !spirv.arm.tensor<3x2x15x7xi8>) -> (!spirv.arm.tensor<3x2x15x7xi8>) {
  %0 = spirv.ARM.GraphConstant {graph_constant_id = 0 : i32} : !spirv.arm.tensor<256xi16>
  // expected-error @+1 {{op failed to verify that all of {input1, table} have same element type}}
  %1 = spirv.Tosa.Table %arg0, %0 : !spirv.arm.tensor<3x2x15x7xi8>, !spirv.arm.tensor<256xi16> -> !spirv.arm.tensor<3x2x15x7xi8>
  spirv.ARM.GraphOutputs %1 : !spirv.arm.tensor<3x2x15x7xi8>
}

spirv.ARM.Graph @table_input_output_shapes_must_match(%arg0: !spirv.arm.tensor<3x2x15x7xi8>) -> (!spirv.arm.tensor<3x2x15x6xi8>) {
  %0 = spirv.ARM.GraphConstant {graph_constant_id = 0 : i32} : !spirv.arm.tensor<256xi8>
  // expected-error @+1 {{op failed to verify that all of {input1, output} have same shape}}
  %1 = spirv.Tosa.Table %arg0, %0 : !spirv.arm.tensor<3x2x15x7xi8>, !spirv.arm.tensor<256xi8> -> !spirv.arm.tensor<3x2x15x6xi8>
  spirv.ARM.GraphOutputs %1 : !spirv.arm.tensor<3x2x15x6xi8>
}

spirv.ARM.Graph @table_input_with_element_type_i8_requires_a_table_of_size_256(%arg0: !spirv.arm.tensor<3x2x15x7xi8>) -> (!spirv.arm.tensor<3x2x15x7xi8>) {
  %0 = spirv.ARM.GraphConstant {graph_constant_id = 0 : i32} : !spirv.arm.tensor<513xi8>
  // expected-error @+1 {{op failed to verify that table must have size 256 if input1 has element type 8-bit signless integer}}
  %1 = spirv.Tosa.Table %arg0, %0 : !spirv.arm.tensor<3x2x15x7xi8>, !spirv.arm.tensor<513xi8> -> !spirv.arm.tensor<3x2x15x7xi8>
  spirv.ARM.GraphOutputs %1 : !spirv.arm.tensor<3x2x15x7xi8>
}

spirv.ARM.Graph @table_input_with_element_type_i16_requires_a_table_of_size_513(%arg0: !spirv.arm.tensor<3x2x15x7xi16>) -> (!spirv.arm.tensor<3x2x15x7xi32>) {
  %0 = spirv.ARM.GraphConstant {graph_constant_id = 0 : i32} : !spirv.arm.tensor<256xi16>
  // expected-error @+1 {{op failed to verify that table must have size 513 if input1 has element type 16-bit signless integer}}
  %1 = spirv.Tosa.Table %arg0, %0 : !spirv.arm.tensor<3x2x15x7xi16>, !spirv.arm.tensor<256xi16> -> !spirv.arm.tensor<3x2x15x7xi32>
  spirv.ARM.GraphOutputs %1 : !spirv.arm.tensor<3x2x15x7xi32>
}

spirv.ARM.Graph @table_input_with_element_type_i8_requires_an_output_with_element_type_i8(%arg0: !spirv.arm.tensor<3x2x15x7xi8>) -> (!spirv.arm.tensor<3x2x15x7xi16>) {
  %0 = spirv.ARM.GraphConstant {graph_constant_id = 0 : i32} : !spirv.arm.tensor<256xi8>
  // expected-error @+1 {{op failed to verify that if input1 has type 8-bit signless integer then output must have a type in [8-bit signless integer]}}
  %1 = spirv.Tosa.Table %arg0, %0 : !spirv.arm.tensor<3x2x15x7xi8>, !spirv.arm.tensor<256xi8> -> !spirv.arm.tensor<3x2x15x7xi16>
  spirv.ARM.GraphOutputs %1 : !spirv.arm.tensor<3x2x15x7xi16>
}

spirv.ARM.Graph @table_input_with_element_type_i16_requires_an_output_with_element_type_i32(%arg0: !spirv.arm.tensor<3x2x15x7xi16>) -> (!spirv.arm.tensor<3x2x15x7xi16>) {
  %0 = spirv.ARM.GraphConstant {graph_constant_id = 0 : i32} : !spirv.arm.tensor<513xi16>
  // expected-error @+1 {{op failed to verify that if input1 has type 16-bit signless integer then output must have a type in [32-bit signless integer]}}
  %1 = spirv.Tosa.Table %arg0, %0 : !spirv.arm.tensor<3x2x15x7xi16>, !spirv.arm.tensor<513xi16> -> !spirv.arm.tensor<3x2x15x7xi16>
  spirv.ARM.GraphOutputs %1 : !spirv.arm.tensor<3x2x15x7xi16>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Abs
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @abs_input_output_element_types_not_matching(%arg0: !spirv.arm.tensor<3x6x14x8xf16>) -> (!spirv.arm.tensor<3x6x14x8xf32>) {
  // expected-error @+1 {{op failed to verify that all of {input1, output} have same type}}
  %0 = spirv.Tosa.Abs %arg0 : !spirv.arm.tensor<3x6x14x8xf16> -> !spirv.arm.tensor<3x6x14x8xf32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<3x6x14x8xf32>
}

spirv.ARM.Graph @abs_input_output_shapes_not_matching(%arg0: !spirv.arm.tensor<3x6x14x8xf16>) -> (!spirv.arm.tensor<3x6x14x9xf16>) {
  // expected-error @+1 {{op failed to verify that all of {input1, output} have same type}}
  %0 = spirv.Tosa.Abs %arg0 : !spirv.arm.tensor<3x6x14x8xf16> -> !spirv.arm.tensor<3x6x14x9xf16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<3x6x14x9xf16>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.BitwiseNot
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @bitwise_not_input_output_element_types_not_matching(%arg0: !spirv.arm.tensor<12x56x50xi16>) -> (!spirv.arm.tensor<12x56x50xi32>) {
  // expected-error @+1 {{op failed to verify that all of {input1, output} have same type}}
  %0 = spirv.Tosa.BitwiseNot %arg0 : !spirv.arm.tensor<12x56x50xi16> -> !spirv.arm.tensor<12x56x50xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<12x56x50xi32>
}

spirv.ARM.Graph @bitwise_not_input_output_shapes_not_matching(%arg0: !spirv.arm.tensor<12x56x50xi16>) -> (!spirv.arm.tensor<12x56x51xi16>) {
  // expected-error @+1 {{op failed to verify that all of {input1, output} have same type}}
  %0 = spirv.Tosa.BitwiseNot %arg0 : !spirv.arm.tensor<12x56x50xi16> -> !spirv.arm.tensor<12x56x51xi16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<12x56x51xi16>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Ceil
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @ceil_input_output_element_types_not_matching(%arg0: !spirv.arm.tensor<46x55x53xf16>) -> (!spirv.arm.tensor<46x55x53xf32>) {
  // expected-error @+1 {{op failed to verify that all of {input1, output} have same type}}
  %0 = spirv.Tosa.Ceil %arg0 : !spirv.arm.tensor<46x55x53xf16> -> !spirv.arm.tensor<46x55x53xf32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<46x55x53xf32>
}

spirv.ARM.Graph @ceil_input_output_shapes_not_matching(%arg0: !spirv.arm.tensor<46x55x53xf16>) -> (!spirv.arm.tensor<46x55x54xf16>) {
  // expected-error @+1 {{op failed to verify that all of {input1, output} have same type}}
  %0 = spirv.Tosa.Ceil %arg0 : !spirv.arm.tensor<46x55x53xf16> -> !spirv.arm.tensor<46x55x54xf16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<46x55x54xf16>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Clz
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @clz_input_output_shapes_not_matching(%arg0: !spirv.arm.tensor<14x10x7x5xi32>) -> (!spirv.arm.tensor<14x10x7x6xi32>) {
  // expected-error @+1 {{op failed to verify that all of {input1, output} have same type}}
  %0 = spirv.Tosa.Clz %arg0 : !spirv.arm.tensor<14x10x7x5xi32> -> !spirv.arm.tensor<14x10x7x6xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<14x10x7x6xi32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Cos
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @cos_input_output_element_types_not_matching(%arg0: !spirv.arm.tensor<44x49x51xf16>) -> (!spirv.arm.tensor<44x49x51xf32>) {
  // expected-error @+1 {{op failed to verify that all of {input1, output} have same type}}
  %0 = spirv.Tosa.Cos %arg0 : !spirv.arm.tensor<44x49x51xf16> -> !spirv.arm.tensor<44x49x51xf32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<44x49x51xf32>
}

spirv.ARM.Graph @cos_input_output_shapes_not_matching(%arg0: !spirv.arm.tensor<44x49x51xf16>) -> (!spirv.arm.tensor<44x49x52xf16>) {
  // expected-error @+1 {{op failed to verify that all of {input1, output} have same type}}
  %0 = spirv.Tosa.Cos %arg0 : !spirv.arm.tensor<44x49x51xf16> -> !spirv.arm.tensor<44x49x52xf16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<44x49x52xf16>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Exp
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @exp_input_output_element_types_not_matching(%arg0: !spirv.arm.tensor<37x53x47xf16>) -> (!spirv.arm.tensor<37x53x47xf32>) {
  // expected-error @+1 {{op failed to verify that all of {input1, output} have same type}}
  %0 = spirv.Tosa.Exp %arg0 : !spirv.arm.tensor<37x53x47xf16> -> !spirv.arm.tensor<37x53x47xf32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<37x53x47xf32>
}

spirv.ARM.Graph @exp_input_output_shapes_not_matching(%arg0: !spirv.arm.tensor<37x53x47xf16>) -> (!spirv.arm.tensor<37x53x48xf16>) {
  // expected-error @+1 {{op failed to verify that all of {input1, output} have same type}}
  %0 = spirv.Tosa.Exp %arg0 : !spirv.arm.tensor<37x53x47xf16> -> !spirv.arm.tensor<37x53x48xf16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<37x53x48xf16>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Floor
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @floor_input_output_element_types_not_matching(%arg0: !spirv.arm.tensor<40x52x42xf16>) -> (!spirv.arm.tensor<40x52x42xf32>) {
  // expected-error @+1 {{op failed to verify that all of {input1, output} have same type}}
  %0 = spirv.Tosa.Floor %arg0 : !spirv.arm.tensor<40x52x42xf16> -> !spirv.arm.tensor<40x52x42xf32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<40x52x42xf32>
}

spirv.ARM.Graph @floor_input_output_shapes_not_matching(%arg0: !spirv.arm.tensor<40x52x42xf16>) -> (!spirv.arm.tensor<40x52x43xf16>) {
  // expected-error @+1 {{op failed to verify that all of {input1, output} have same type}}
  %0 = spirv.Tosa.Floor %arg0 : !spirv.arm.tensor<40x52x42xf16> -> !spirv.arm.tensor<40x52x43xf16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<40x52x43xf16>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Log
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @log_input_output_element_types_not_matching(%arg0: !spirv.arm.tensor<45x43x36xf16>) -> (!spirv.arm.tensor<45x43x36xf32>) {
  // expected-error @+1 {{op failed to verify that all of {input1, output} have same type}}
  %0 = spirv.Tosa.Log %arg0 : !spirv.arm.tensor<45x43x36xf16> -> !spirv.arm.tensor<45x43x36xf32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<45x43x36xf32>
}

spirv.ARM.Graph @log_input_output_shapes_not_matching(%arg0: !spirv.arm.tensor<45x43x36xf16>) -> (!spirv.arm.tensor<45x43x37xf16>) {
  // expected-error @+1 {{op failed to verify that all of {input1, output} have same type}}
  %0 = spirv.Tosa.Log %arg0 : !spirv.arm.tensor<45x43x36xf16> -> !spirv.arm.tensor<45x43x37xf16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<45x43x37xf16>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.LogicalNot
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @logical_not_input_output_shapes_not_matching(%arg0: !spirv.arm.tensor<54x26x10xi1>) -> (!spirv.arm.tensor<54x26x11xi1>) {
  // expected-error @+1 {{op failed to verify that all of {input1, output} have same type}}
  %0 = spirv.Tosa.LogicalNot %arg0 : !spirv.arm.tensor<54x26x10xi1> -> !spirv.arm.tensor<54x26x11xi1>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<54x26x11xi1>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Negate
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @negate_input_output_element_types_not_matching(%arg0: !spirv.arm.tensor<3x1x65540x1xi8>) -> (!spirv.arm.tensor<3x1x65540x2xi16>) {
  %0 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi8>
  %1 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi8>
  // expected-error @+1 {{op failed to verify that all of {input1, output, input1_zp, output_zp} have same element type}}
  %2 = spirv.Tosa.Negate %arg0, %0, %1 : !spirv.arm.tensor<3x1x65540x1xi8>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<3x1x65540x2xi16>
  spirv.ARM.GraphOutputs %2 : !spirv.arm.tensor<3x1x65540x2xi16>
}

spirv.ARM.Graph @negate_input_output_shapes_not_matching(%arg0: !spirv.arm.tensor<3x1x65540x1xi8>) -> (!spirv.arm.tensor<3x1x65540x2xi8>) {
  %0 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi8>
  %1 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi8>
  // expected-error @+1 {{op failed to verify that all of {input1, output} have same type}}
  %2 = spirv.Tosa.Negate %arg0, %0, %1 : !spirv.arm.tensor<3x1x65540x1xi8>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<3x1x65540x2xi8>
  spirv.ARM.GraphOutputs %2 : !spirv.arm.tensor<3x1x65540x2xi8>
}

spirv.ARM.Graph @negate_input_zero_point_element_types_not_matching(%arg0: !spirv.arm.tensor<3x1x65540x1xi8>) -> (!spirv.arm.tensor<3x1x65540x1xi8>) {
  %0 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi16>
  %1 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi8>
  // expected-error @+1 {{op failed to verify that all of {input1, output, input1_zp, output_zp} have same element type}}
  %2 = spirv.Tosa.Negate %arg0, %0, %1 : !spirv.arm.tensor<3x1x65540x1xi8>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<3x1x65540x1xi8>
  spirv.ARM.GraphOutputs %2 : !spirv.arm.tensor<3x1x65540x1xi8>
}

spirv.ARM.Graph @negate_output_zero_point_element_types_not_matching(%arg0: !spirv.arm.tensor<3x1x65540x1xi8>) -> (!spirv.arm.tensor<3x1x65540x1xi8>) {
  %0 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi8>
  %1 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi16>
  // expected-error @+1 {{op failed to verify that all of {input1, output, input1_zp, output_zp} have same element type}}
  %2 = spirv.Tosa.Negate %arg0, %0, %1 : !spirv.arm.tensor<3x1x65540x1xi8>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi16> -> !spirv.arm.tensor<3x1x65540x1xi8>
  spirv.ARM.GraphOutputs %2 : !spirv.arm.tensor<3x1x65540x1xi8>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Reciprocal
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @reciprocal_input_output_element_types_not_matching(%arg0: !spirv.arm.tensor<38x47x44xf16>) -> (!spirv.arm.tensor<38x47x44xf32>) {
  // expected-error @+1 {{op failed to verify that all of {input1, output} have same type}}
  %0 = spirv.Tosa.Reciprocal %arg0 : !spirv.arm.tensor<38x47x44xf16> -> !spirv.arm.tensor<38x47x44xf32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<38x47x44xf32>
}

spirv.ARM.Graph @reciprocal_input_output_shapes_not_matching(%arg0: !spirv.arm.tensor<38x47x44xf16>) -> (!spirv.arm.tensor<38x47x45xf16>) {
  // expected-error @+1 {{op failed to verify that all of {input1, output} have same type}}
  %0 = spirv.Tosa.Reciprocal %arg0 : !spirv.arm.tensor<38x47x44xf16> -> !spirv.arm.tensor<38x47x45xf16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<38x47x45xf16>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Rsqrt
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @rsqrt_input_output_element_types_not_matching(%arg0: !spirv.arm.tensor<40x57x56xf16>) -> (!spirv.arm.tensor<40x57x56xf32>) {
  // expected-error @+1 {{op failed to verify that all of {input1, output} have same type}}
  %0 = spirv.Tosa.Rsqrt %arg0 : !spirv.arm.tensor<40x57x56xf16> -> !spirv.arm.tensor<40x57x56xf32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<40x57x56xf32>
}

spirv.ARM.Graph @rsqrt_input_output_shapes_not_matching(%arg0: !spirv.arm.tensor<40x57x56xf16>) -> (!spirv.arm.tensor<40x57x57xf16>) {
  // expected-error @+1 {{op failed to verify that all of {input1, output} have same type}}
  %0 = spirv.Tosa.Rsqrt %arg0 : !spirv.arm.tensor<40x57x56xf16> -> !spirv.arm.tensor<40x57x57xf16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<40x57x57xf16>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Sin
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @sin_input_output_element_types_not_matching(%arg0: !spirv.arm.tensor<49x38x58xf16>) -> (!spirv.arm.tensor<49x38x58xf32>) {
  // expected-error @+1 {{op failed to verify that all of {input1, output} have same type}}
  %0 = spirv.Tosa.Sin %arg0 : !spirv.arm.tensor<49x38x58xf16> -> !spirv.arm.tensor<49x38x58xf32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<49x38x58xf32>
}

spirv.ARM.Graph @sin_input_output_shapes_not_matching(%arg0: !spirv.arm.tensor<49x38x58xf16>) -> (!spirv.arm.tensor<49x38x59xf16>) {
  // expected-error @+1 {{op failed to verify that all of {input1, output} have same type}}
  %0 = spirv.Tosa.Sin %arg0 : !spirv.arm.tensor<49x38x58xf16> -> !spirv.arm.tensor<49x38x59xf16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<49x38x59xf16>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Select
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @select_true_value_output_element_types_not_matching(%arg0: !spirv.arm.tensor<4x6x4x5xi1>, %arg1: !spirv.arm.tensor<4x6x4x5xi8>, %arg2: !spirv.arm.tensor<4x6x4x5xi8>) -> (!spirv.arm.tensor<4x6x4x5xi16>) {
  // expected-error @+1 {{op failed to verify that all of {true_value, false_value, output} have same element type}}
  %0 = spirv.Tosa.Select %arg0, %arg1, %arg2 : !spirv.arm.tensor<4x6x4x5xi1>, !spirv.arm.tensor<4x6x4x5xi8>, !spirv.arm.tensor<4x6x4x5xi8> -> !spirv.arm.tensor<4x6x4x5xi16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<4x6x4x5xi16>
}

spirv.ARM.Graph @select_true_value_false_value_element_types_not_matching(%arg0: !spirv.arm.tensor<4x6x4x5xi1>, %arg1: !spirv.arm.tensor<4x6x4x5xi8>, %arg2: !spirv.arm.tensor<4x6x4x5xi16>) -> (!spirv.arm.tensor<4x6x4x5xi8>) {
  // expected-error @+1 {{op failed to verify that all of {true_value, false_value, output} have same element type}}
  %0 = spirv.Tosa.Select %arg0, %arg1, %arg2 : !spirv.arm.tensor<4x6x4x5xi1>, !spirv.arm.tensor<4x6x4x5xi8>, !spirv.arm.tensor<4x6x4x5xi16> -> !spirv.arm.tensor<4x6x4x5xi8>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<4x6x4x5xi8>
}

spirv.ARM.Graph @select_false_value_output_element_types_not_matching(%arg0: !spirv.arm.tensor<4x6x4x5xi1>, %arg1: !spirv.arm.tensor<4x6x4x5xi16>, %arg2: !spirv.arm.tensor<4x6x4x5xi8>) -> (!spirv.arm.tensor<4x6x4x5xi16>) {
  // expected-error @+1 {{op failed to verify that all of {true_value, false_value, output} have same element type}}
  %0 = spirv.Tosa.Select %arg0, %arg1, %arg2 : !spirv.arm.tensor<4x6x4x5xi1>, !spirv.arm.tensor<4x6x4x5xi16>, !spirv.arm.tensor<4x6x4x5xi8> -> !spirv.arm.tensor<4x6x4x5xi16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<4x6x4x5xi16>
}

spirv.ARM.Graph @select_condition_must_be_bool_tensor(%arg0: !spirv.arm.tensor<4x6x4x5xi8>, %arg1: !spirv.arm.tensor<4x6x4x5xi8>, %arg2: !spirv.arm.tensor<4x6x4x5xi8>) -> (!spirv.arm.tensor<4x6x4x5xi8>) {
  // expected-error @+1 {{op operand #0 must be 1D/2D/3D/4D/5D/6D tensorArm of bool values, but got '!spirv.arm.tensor<4x6x4x5xi8>'}}
  %0 = spirv.Tosa.Select %arg0, %arg1, %arg2 : !spirv.arm.tensor<4x6x4x5xi8>, !spirv.arm.tensor<4x6x4x5xi8>, !spirv.arm.tensor<4x6x4x5xi8> -> !spirv.arm.tensor<4x6x4x5xi8>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<4x6x4x5xi8>
}

spirv.ARM.Graph @select_condition_rank_not_matching(%arg0: !spirv.arm.tensor<4x6x4xi1>, %arg1: !spirv.arm.tensor<4x6x4x5xi8>, %arg2: !spirv.arm.tensor<4x6x4x5xi8>) -> (!spirv.arm.tensor<4x6x4x5xi8>) {
  // expected-error @+1 {{op failed to verify that all of {condition, true_value, false_value, output} have same rank}}
  %0 = spirv.Tosa.Select %arg0, %arg1, %arg2 : !spirv.arm.tensor<4x6x4xi1>, !spirv.arm.tensor<4x6x4x5xi8>, !spirv.arm.tensor<4x6x4x5xi8> -> !spirv.arm.tensor<4x6x4x5xi8>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<4x6x4x5xi8>
}

spirv.ARM.Graph @select_true_value_rank_not_matching(%arg0: !spirv.arm.tensor<4x6x4x5xi1>, %arg1: !spirv.arm.tensor<4x6x4xi8>, %arg2: !spirv.arm.tensor<4x6x4x5xi8>) -> (!spirv.arm.tensor<4x6x4x5xi8>) {
  // expected-error @+1 {{op failed to verify that all of {condition, true_value, false_value, output} have same rank}}
  %0 = spirv.Tosa.Select %arg0, %arg1, %arg2 : !spirv.arm.tensor<4x6x4x5xi1>, !spirv.arm.tensor<4x6x4xi8>, !spirv.arm.tensor<4x6x4x5xi8> -> !spirv.arm.tensor<4x6x4x5xi8>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<4x6x4x5xi8>
}

spirv.ARM.Graph @select_false_value_rank_not_matching(%arg0: !spirv.arm.tensor<4x6x4x5xi1>, %arg1: !spirv.arm.tensor<4x6x4x5xi8>, %arg2: !spirv.arm.tensor<4x6x4xi8>) -> (!spirv.arm.tensor<4x6x4x5xi8>) {
  // expected-error @+1 {{op failed to verify that all of {condition, true_value, false_value, output} have same rank}}
  %0 = spirv.Tosa.Select %arg0, %arg1, %arg2 : !spirv.arm.tensor<4x6x4x5xi1>, !spirv.arm.tensor<4x6x4x5xi8>, !spirv.arm.tensor<4x6x4xi8> -> !spirv.arm.tensor<4x6x4x5xi8>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<4x6x4x5xi8>
}

spirv.ARM.Graph @select_output_rank_not_matching_inputs(%arg0: !spirv.arm.tensor<4x1x4x5xi1>, %arg1: !spirv.arm.tensor<4x6x4x5xi8>, %arg2: !spirv.arm.tensor<4x1x4x5xi8>) -> (!spirv.arm.tensor<4x6x5xi8>) {
  // expected-error @+1 {{op failed to verify that all of {condition, true_value, false_value, output} have same rank}}
  %0 = spirv.Tosa.Select %arg0, %arg1, %arg2 : !spirv.arm.tensor<4x1x4x5xi1>, !spirv.arm.tensor<4x6x4x5xi8>, !spirv.arm.tensor<4x1x4x5xi8> -> !spirv.arm.tensor<4x6x5xi8>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<4x6x5xi8>
}

spirv.ARM.Graph @select_condition_true_value_not_broadcastable_false_value_compatible(%arg0: !spirv.arm.tensor<4x2x4x5xi1>, %arg1: !spirv.arm.tensor<4x3x4x5xi8>, %arg2: !spirv.arm.tensor<4x1x4x5xi8>) -> (!spirv.arm.tensor<4x3x4x5xi8>) {
  // expected-error @+1 {{op failed to verify that the shape of inputs: condition, true_value, and false_value are compatible for broadcasting}}
  %0 = spirv.Tosa.Select %arg0, %arg1, %arg2 : !spirv.arm.tensor<4x2x4x5xi1>, !spirv.arm.tensor<4x3x4x5xi8>, !spirv.arm.tensor<4x1x4x5xi8> -> !spirv.arm.tensor<4x3x4x5xi8>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<4x3x4x5xi8>
}

spirv.ARM.Graph @select_condition_false_value_not_broadcastable_true_value_compatible(%arg0: !spirv.arm.tensor<4x2x4x5xi1>, %arg1: !spirv.arm.tensor<4x1x4x5xi8>, %arg2: !spirv.arm.tensor<4x3x4x5xi8>) -> (!spirv.arm.tensor<4x3x4x5xi8>) {
  // expected-error @+1 {{op failed to verify that the shape of inputs: condition, true_value, and false_value are compatible for broadcasting}}
  %0 = spirv.Tosa.Select %arg0, %arg1, %arg2 : !spirv.arm.tensor<4x2x4x5xi1>, !spirv.arm.tensor<4x1x4x5xi8>, !spirv.arm.tensor<4x3x4x5xi8> -> !spirv.arm.tensor<4x3x4x5xi8>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<4x3x4x5xi8>
}

spirv.ARM.Graph @select_true_value_false_value_not_broadcastable_condition_compatible(%arg0: !spirv.arm.tensor<4x1x4x5xi1>, %arg1: !spirv.arm.tensor<4x2x4x5xi8>, %arg2: !spirv.arm.tensor<4x3x4x5xi8>) -> (!spirv.arm.tensor<4x3x4x5xi8>) {
  // expected-error @+1 {{op failed to verify that the shape of inputs: condition, true_value, and false_value are compatible for broadcasting}}
  %0 = spirv.Tosa.Select %arg0, %arg1, %arg2 : !spirv.arm.tensor<4x1x4x5xi1>, !spirv.arm.tensor<4x2x4x5xi8>, !spirv.arm.tensor<4x3x4x5xi8> -> !spirv.arm.tensor<4x3x4x5xi8>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<4x3x4x5xi8>
}

spirv.ARM.Graph @select_inputs_broadcastable_output_shape_not_broadcast_shape(%arg0: !spirv.arm.tensor<4x1x4x5xi1>, %arg1: !spirv.arm.tensor<4x6x4x5xi8>, %arg2: !spirv.arm.tensor<4x1x4x5xi8>) -> (!spirv.arm.tensor<4x1x4x5xi8>) {
  // expected-error @+1 {{op failed to verify that the broadcast shape of inputs: condition, true_value, and false_value is equal to the output shape}}
  %0 = spirv.Tosa.Select %arg0, %arg1, %arg2 : !spirv.arm.tensor<4x1x4x5xi1>, !spirv.arm.tensor<4x6x4x5xi8>, !spirv.arm.tensor<4x1x4x5xi8> -> !spirv.arm.tensor<4x1x4x5xi8>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<4x1x4x5xi8>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Equal
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @equal_input_element_types_not_matching(%arg0: !spirv.arm.tensor<16x11x5x3xf32>, %arg1: !spirv.arm.tensor<16x11x5x3xf16>) -> (!spirv.arm.tensor<16x11x5x3xi1>) {
  // expected-error @+1 {{op failed to verify that all of {input1, input2} have same element type}}
  %0 = spirv.Tosa.Equal %arg0, %arg1 : !spirv.arm.tensor<16x11x5x3xf32>, !spirv.arm.tensor<16x11x5x3xf16> -> !spirv.arm.tensor<16x11x5x3xi1>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<16x11x5x3xi1>
}

spirv.ARM.Graph @equal_input_shapes_not_broadcastable(%arg0: !spirv.arm.tensor<16x11x5x3xf32>, %arg1: !spirv.arm.tensor<16x7x5x3xf32>) -> (!spirv.arm.tensor<16x11x5x3xi1>) {
  // expected-error @+1 {{op failed to verify that the shape of input1 and input2 are compatible for broadcasting and the broadcast shape is equal to the output shape}}
  %0 = spirv.Tosa.Equal %arg0, %arg1 : !spirv.arm.tensor<16x11x5x3xf32>, !spirv.arm.tensor<16x7x5x3xf32> -> !spirv.arm.tensor<16x11x5x3xi1>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<16x11x5x3xi1>
}

spirv.ARM.Graph @equal_output_shape_not_broadcast_shape(%arg0: !spirv.arm.tensor<16x11x5x3xf32>, %arg1: !spirv.arm.tensor<16x1x5x3xf32>) -> (!spirv.arm.tensor<16x1x5x3xi1>) {
  // expected-error @+1 {{op failed to verify that the shape of input1 and input2 are compatible for broadcasting and the broadcast shape is equal to the output shape}}
  %0 = spirv.Tosa.Equal %arg0, %arg1 : !spirv.arm.tensor<16x11x5x3xf32>, !spirv.arm.tensor<16x1x5x3xf32> -> !spirv.arm.tensor<16x1x5x3xi1>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<16x1x5x3xi1>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Greater
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @greater_input_element_types_not_matching(%arg0: !spirv.arm.tensor<11x10x10x2xi32>, %arg1: !spirv.arm.tensor<11x10x10x2xf16>) -> (!spirv.arm.tensor<11x10x10x2xi1>) {
  // expected-error @+1 {{op failed to verify that all of {input1, input2} have same element type}}
  %0 = spirv.Tosa.Greater %arg0, %arg1 : !spirv.arm.tensor<11x10x10x2xi32>, !spirv.arm.tensor<11x10x10x2xf16> -> !spirv.arm.tensor<11x10x10x2xi1>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<11x10x10x2xi1>
}

spirv.ARM.Graph @greater_output_shape_not_broadcast_shape(%arg0: !spirv.arm.tensor<11x10x10x2xi32>, %arg1: !spirv.arm.tensor<11x10x10x1xi32>) -> (!spirv.arm.tensor<11x10x10x1xi1>) {
  // expected-error @+1 {{op failed to verify that the shape of input1 and input2 are compatible for broadcasting and the broadcast shape is equal to the output shape}}
  %0 = spirv.Tosa.Greater %arg0, %arg1 : !spirv.arm.tensor<11x10x10x2xi32>, !spirv.arm.tensor<11x10x10x1xi32> -> !spirv.arm.tensor<11x10x10x1xi1>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<11x10x10x1xi1>
}

spirv.ARM.Graph @greater_output_shape_not_broadcast_shape_batch_dim(%arg0: !spirv.arm.tensor<11x10x10x2xf16>, %arg1: !spirv.arm.tensor<1x10x10x2xf16>) -> (!spirv.arm.tensor<1x10x10x2xi1>) {
  // expected-error @+1 {{op failed to verify that the shape of input1 and input2 are compatible for broadcasting and the broadcast shape is equal to the output shape}}
  %0 = spirv.Tosa.Greater %arg0, %arg1 : !spirv.arm.tensor<11x10x10x2xf16>, !spirv.arm.tensor<1x10x10x2xf16> -> !spirv.arm.tensor<1x10x10x2xi1>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<1x10x10x2xi1>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.GreaterEqual
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @greaterequal_input_element_types_not_matching(%arg0: !spirv.arm.tensor<10x17x7x16xi32>, %arg1: !spirv.arm.tensor<10x17x7x16xf32>) -> (!spirv.arm.tensor<10x17x7x16xi1>) {
  // expected-error @+1 {{op failed to verify that all of {input1, input2} have same element type}}
  %0 = spirv.Tosa.GreaterEqual %arg0, %arg1 : !spirv.arm.tensor<10x17x7x16xi32>, !spirv.arm.tensor<10x17x7x16xf32> -> !spirv.arm.tensor<10x17x7x16xi1>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<10x17x7x16xi1>
}

spirv.ARM.Graph @greaterequal_input_shapes_not_broadcastable(%arg0: !spirv.arm.tensor<10x17x7x16xf32>, %arg1: !spirv.arm.tensor<10x17x5x16xf32>) -> (!spirv.arm.tensor<10x17x7x16xi1>) {
  // expected-error @+1 {{op failed to verify that the shape of input1 and input2 are compatible for broadcasting and the broadcast shape is equal to the output shape}}
  %0 = spirv.Tosa.GreaterEqual %arg0, %arg1 : !spirv.arm.tensor<10x17x7x16xf32>, !spirv.arm.tensor<10x17x5x16xf32> -> !spirv.arm.tensor<10x17x7x16xi1>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<10x17x7x16xi1>
}

spirv.ARM.Graph @greaterequal_output_shape_not_broadcast_shape(%arg0: !spirv.arm.tensor<10x17x7x16xf32>, %arg1: !spirv.arm.tensor<1x17x7x16xf32>) -> (!spirv.arm.tensor<1x17x7x16xi1>) {
  // expected-error @+1 {{op failed to verify that the shape of input1 and input2 are compatible for broadcasting and the broadcast shape is equal to the output shape}}
  %0 = spirv.Tosa.GreaterEqual %arg0, %arg1 : !spirv.arm.tensor<10x17x7x16xf32>, !spirv.arm.tensor<1x17x7x16xf32> -> !spirv.arm.tensor<1x17x7x16xi1>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<1x17x7x16xi1>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.ReduceAll
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @reduceall_input_output_ranks_not_matching(%arg0: !spirv.arm.tensor<18x22x23x12xi1>) -> (!spirv.arm.tensor<18x22x12xi1>) {
  // expected-error @+1 {{op failed to verify that all of {input, output} have same rank}}
  %0 = spirv.Tosa.ReduceAll axis = 2, %arg0 : !spirv.arm.tensor<18x22x23x12xi1> -> !spirv.arm.tensor<18x22x12xi1>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<18x22x12xi1>
}

spirv.ARM.Graph @reduceall_axis_value_not_in_input_rank_range(%arg0: !spirv.arm.tensor<18x22x23x12xi1>) -> (!spirv.arm.tensor<18x22x23x12xi1>) {
  // expected-error @+1 {{op failed to verify that axis attribute value should be lower than rank(input)}}
  %0 = spirv.Tosa.ReduceAll axis = 4, %arg0 : !spirv.arm.tensor<18x22x23x12xi1> -> !spirv.arm.tensor<18x22x23x12xi1>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<18x22x23x12xi1>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.ReduceAny
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @reduceany_input_output_ranks_not_matching(%arg0: !spirv.arm.tensor<25x13x30x8xi1>) -> (!spirv.arm.tensor<25x13x8xi1>) {
  // expected-error @+1 {{op failed to verify that all of {input, output} have same rank}}
  %0 = spirv.Tosa.ReduceAny axis = 2, %arg0 : !spirv.arm.tensor<25x13x30x8xi1> -> !spirv.arm.tensor<25x13x8xi1>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<25x13x8xi1>
}

spirv.ARM.Graph @reduceany_axis_value_not_in_input_rank_range(%arg0: !spirv.arm.tensor<25x13x30x8xi1>) -> (!spirv.arm.tensor<25x13x30x8xi1>) {
  // expected-error @+1 {{op failed to verify that axis attribute value should be lower than rank(input)}}
  %0 = spirv.Tosa.ReduceAny axis = 4, %arg0 : !spirv.arm.tensor<25x13x30x8xi1> -> !spirv.arm.tensor<25x13x30x8xi1>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<25x13x30x8xi1>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.ReduceMax
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @reducemax_input_output_element_types_not_matching(%arg0: !spirv.arm.tensor<16x20x10xf16>) -> (!spirv.arm.tensor<16x20x10xf32>) {
  // expected-error @+1 {{op failed to verify that all of {input, output} have same element type}}
  %0 = spirv.Tosa.ReduceMax axis = 2, nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<16x20x10xf16> -> !spirv.arm.tensor<16x20x10xf32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<16x20x10xf32>
}

spirv.ARM.Graph @reducemax_input_output_ranks_not_matching(%arg0: !spirv.arm.tensor<8x30x12x3xi8>) -> (!spirv.arm.tensor<8x30x3xi8>) {
  // expected-error @+1 {{op failed to verify that all of {input, output} have same rank}}
  %0 = spirv.Tosa.ReduceMax axis = 2, nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<8x30x12x3xi8> -> !spirv.arm.tensor<8x30x3xi8>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<8x30x3xi8>
}

spirv.ARM.Graph @reducemax_axis_value_not_in_input_rank_range(%arg0: !spirv.arm.tensor<8x30x12x3xi8>) -> (!spirv.arm.tensor<8x30x12x3xi8>) {
  // expected-error @+1 {{op failed to verify that axis attribute value should be lower than rank(input)}}
  %0 = spirv.Tosa.ReduceMax axis = 4, nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<8x30x12x3xi8> -> !spirv.arm.tensor<8x30x12x3xi8>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<8x30x12x3xi8>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.ReduceMin
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @reducemin_input_output_element_types_not_matching(%arg0: !spirv.arm.tensor<27x10x25x9xf16>) -> (!spirv.arm.tensor<27x10x25x9xf32>) {
  // expected-error @+1 {{op failed to verify that all of {input, output} have same element type}}
  %0 = spirv.Tosa.ReduceMin axis = 2, nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<27x10x25x9xf16> -> !spirv.arm.tensor<27x10x25x9xf32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<27x10x25x9xf32>
}

spirv.ARM.Graph @reducemin_input_output_ranks_not_matching(%arg0: !spirv.arm.tensor<2x5x5x1xi8>) -> (!spirv.arm.tensor<2x5x1xi8>) {
  // expected-error @+1 {{op failed to verify that all of {input, output} have same rank}}
  %0 = spirv.Tosa.ReduceMin axis = 2, nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<2x5x5x1xi8> -> !spirv.arm.tensor<2x5x1xi8>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<2x5x1xi8>
}

spirv.ARM.Graph @reducemin_axis_value_not_in_input_rank_range(%arg0: !spirv.arm.tensor<27x10x25x9xf16>) -> (!spirv.arm.tensor<27x10x25x9xf16>) {
  // expected-error @+1 {{op failed to verify that axis attribute value should be lower than rank(input)}}
  %0 = spirv.Tosa.ReduceMin axis = 4, nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<27x10x25x9xf16> -> !spirv.arm.tensor<27x10x25x9xf16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<27x10x25x9xf16>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.ReduceProduct
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @reduceproduct_input_output_element_types_not_matching(%arg0: !spirv.arm.tensor<2x16x25xf16>) -> (!spirv.arm.tensor<2x16x25xf32>) {
  // expected-error @+1 {{op failed to verify that all of {input, output} have same element type}}
  %0 = spirv.Tosa.ReduceProduct axis = 2, %arg0 : !spirv.arm.tensor<2x16x25xf16> -> !spirv.arm.tensor<2x16x25xf32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<2x16x25xf32>
}

spirv.ARM.Graph @reduceproduct_input_output_ranks_not_matching(%arg0: !spirv.arm.tensor<2x16x25xf16>) -> (!spirv.arm.tensor<2x25xf16>) {
  // expected-error @+1 {{op failed to verify that all of {input, output} have same rank}}
  %0 = spirv.Tosa.ReduceProduct axis = 1, %arg0 : !spirv.arm.tensor<2x16x25xf16> -> !spirv.arm.tensor<2x25xf16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<2x25xf16>
}

spirv.ARM.Graph @reduceproduct_axis_value_not_in_input_rank_range(%arg0: !spirv.arm.tensor<2x16x25xf16>) -> (!spirv.arm.tensor<2x16x25xf16>) {
  // expected-error @+1 {{op failed to verify that axis attribute value should be lower than rank(input)}}
  %0 = spirv.Tosa.ReduceProduct axis = 3, %arg0 : !spirv.arm.tensor<2x16x25xf16> -> !spirv.arm.tensor<2x16x25xf16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<2x16x25xf16>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.ReduceSum
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @reducesum_input_output_element_types_not_matching(%arg0: !spirv.arm.tensor<32x32x33xf32>) -> (!spirv.arm.tensor<32x32x33xf16>) {
  // expected-error @+1 {{op failed to verify that all of {input, output} have same element type}}
  %0 = spirv.Tosa.ReduceSum axis = 1, %arg0 : !spirv.arm.tensor<32x32x33xf32> -> !spirv.arm.tensor<32x32x33xf16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<32x32x33xf16>
}

spirv.ARM.Graph @reducesum_input_output_ranks_not_matching(%arg0: !spirv.arm.tensor<20x24x22xi32>) -> (!spirv.arm.tensor<20x22xi32>) {
  // expected-error @+1 {{op failed to verify that all of {input, output} have same rank}}
  %0 = spirv.Tosa.ReduceSum axis = 1, %arg0 : !spirv.arm.tensor<20x24x22xi32> -> !spirv.arm.tensor<20x22xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<20x22xi32>
}

spirv.ARM.Graph @reducesum_axis_value_not_in_input_rank_range(%arg0: !spirv.arm.tensor<20x24x22xi32>) -> (!spirv.arm.tensor<20x24x22xi32>) {
  // expected-error @+1 {{op failed to verify that axis attribute value should be lower than rank(input)}}
  %0 = spirv.Tosa.ReduceSum axis = 3, %arg0 : !spirv.arm.tensor<20x24x22xi32> -> !spirv.arm.tensor<20x24x22xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<20x24x22xi32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Concat
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @concat_must_have_at_least_one_input() -> (!spirv.arm.tensor<4x12xi8>) {
  // expected-error @+1 {{op failed to verify that variadic input1 must has at least 1 elements}}
  %0 = "spirv.Tosa.Concat"() <{axis = 0 : i32}> : () -> !spirv.arm.tensor<4x12xi8>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<4x12xi8>
}

spirv.ARM.Graph @concat_input_output_element_types_not_matching(%arg0: !spirv.arm.tensor<4x5xi8>, %arg1: !spirv.arm.tensor<4x7xi8>) -> (!spirv.arm.tensor<4x12xi16>) {
  // expected-error @+1 {{op failed to verify that all elements of variadic input1 must have same element type}}
  %0 = spirv.Tosa.Concat axis = 1, %arg0, %arg1 : !spirv.arm.tensor<4x5xi8>, !spirv.arm.tensor<4x7xi8> -> !spirv.arm.tensor<4x12xi16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<4x12xi16>
}

spirv.ARM.Graph @concat_input_output_ranks_not_matching(%arg0: !spirv.arm.tensor<4x5xi8>, %arg1: !spirv.arm.tensor<4x7xi8>) -> (!spirv.arm.tensor<4x12x1xi8>) {
  // expected-error @+1 {{op failed to verify that all elements of variadic input1 must have same element type}}
  %0 = spirv.Tosa.Concat axis = 1, %arg0, %arg1 : !spirv.arm.tensor<4x5xi8>, !spirv.arm.tensor<4x7xi8> -> !spirv.arm.tensor<4x12x1xi8>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<4x12x1xi8>
}

spirv.ARM.Graph @concat_axis_value_not_in_output_rank_range(%arg0: !spirv.arm.tensor<4x5xi8>, %arg1: !spirv.arm.tensor<4x7xi8>) -> (!spirv.arm.tensor<4x12xi8>) {
  // expected-error @+1 {{op failed to verify that axis attribute value should be lower than rank(output)}}
  %0 = spirv.Tosa.Concat axis = 2, %arg0, %arg1 : !spirv.arm.tensor<4x5xi8>, !spirv.arm.tensor<4x7xi8> -> !spirv.arm.tensor<4x12xi8>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<4x12xi8>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Pad
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @pad_input_pad_const_output_element_types_not_matching(%arg0: !spirv.arm.tensor<4x7xi8>, %arg1: !spirv.arm.tensor<4xi32>, %arg2: !spirv.arm.tensor<1xi16>) -> (!spirv.arm.tensor<5x8xi8>) {
  // expected-error @+1 {{op failed to verify that all of {input1, pad_const, output} have same element type}}
  %0 = spirv.Tosa.Pad %arg0, %arg1, %arg2 : !spirv.arm.tensor<4x7xi8>, !spirv.arm.tensor<4xi32>, !spirv.arm.tensor<1xi16> -> !spirv.arm.tensor<5x8xi8>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<5x8xi8>
}

spirv.ARM.Graph @pad_input_output_ranks_not_matching(%arg0: !spirv.arm.tensor<4x7xi8>, %arg1: !spirv.arm.tensor<4xi32>, %arg2: !spirv.arm.tensor<1xi8>) -> (!spirv.arm.tensor<1x5x8xi8>) {
  // expected-error @+1 {{op failed to verify that all of {input1, output} have same rank}}
  %0 = spirv.Tosa.Pad %arg0, %arg1, %arg2 : !spirv.arm.tensor<4x7xi8>, !spirv.arm.tensor<4xi32>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x5x8xi8>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<1x5x8xi8>
}

spirv.ARM.Graph @pad_padding_element_count_not_twice_input_rank(%arg0: !spirv.arm.tensor<4x7xi8>, %arg1: !spirv.arm.tensor<6xi32>, %arg2: !spirv.arm.tensor<1xi8>) -> (!spirv.arm.tensor<5x8xi8>) {
  // expected-error @+1 {{op failed to verify that the number of elements of padding must be rank(input1) * 2}}
  %0 = spirv.Tosa.Pad %arg0, %arg1, %arg2 : !spirv.arm.tensor<4x7xi8>, !spirv.arm.tensor<6xi32>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<5x8xi8>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<5x8xi8>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Reshape
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @reshape_input_output_element_types_not_matching(%arg0: !spirv.arm.tensor<2x3x4xi8>, %arg1: !spirv.arm.tensor<2xi32>) -> (!spirv.arm.tensor<6x4xi16>) {
  // expected-error @+1 {{op failed to verify that all of {input1, output} have same element type}}
  %0 = spirv.Tosa.Reshape %arg0, %arg1 : !spirv.arm.tensor<2x3x4xi8>, !spirv.arm.tensor<2xi32> -> !spirv.arm.tensor<6x4xi16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x4xi16>
}

spirv.ARM.Graph @reshape_input_output_element_counts_not_matching(%arg0: !spirv.arm.tensor<2x3x4xi8>, %arg1: !spirv.arm.tensor<2xi32>) -> (!spirv.arm.tensor<5x4xi8>) {
  // expected-error @+1 {{op failed to verify that all of {input1, output} have same element count}}
  %0 = spirv.Tosa.Reshape %arg0, %arg1 : !spirv.arm.tensor<2x3x4xi8>, !spirv.arm.tensor<2xi32> -> !spirv.arm.tensor<5x4xi8>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<5x4xi8>
}

spirv.ARM.Graph @reshape_shape_element_count_not_output_rank(%arg0: !spirv.arm.tensor<2x3x4xi8>, %arg1: !spirv.arm.tensor<4xi32>) -> (!spirv.arm.tensor<6x4xi8>) {
  // expected-error @+1 {{op failed to verify that the number of elements of shape must be rank(output)}}
  %0 = spirv.Tosa.Reshape %arg0, %arg1 : !spirv.arm.tensor<2x3x4xi8>, !spirv.arm.tensor<4xi32> -> !spirv.arm.tensor<6x4xi8>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x4xi8>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Reverse
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @reverse_input_output_types_not_matching(%arg0: !spirv.arm.tensor<2x3x4xi8>) -> (!spirv.arm.tensor<2x3x4xi16>) {
  // expected-error @+1 {{op failed to verify that all of {input1, output} have same type}}
  %0 = spirv.Tosa.Reverse axis = 1, %arg0 : !spirv.arm.tensor<2x3x4xi8> -> !spirv.arm.tensor<2x3x4xi16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<2x3x4xi16>
}

spirv.ARM.Graph @reverse_axis_value_not_in_input_rank_range(%arg0: !spirv.arm.tensor<2x3x4xi8>) -> (!spirv.arm.tensor<2x3x4xi8>) {
  // expected-error @+1 {{op failed to verify that axis attribute value should be lower than rank(input1)}}
  %0 = spirv.Tosa.Reverse axis = 3, %arg0 : !spirv.arm.tensor<2x3x4xi8> -> !spirv.arm.tensor<2x3x4xi8>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<2x3x4xi8>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Slice
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @slice_input_output_element_types_not_matching(%arg0: !spirv.arm.tensor<4x5x6xi8>) -> (!spirv.arm.tensor<2x3x4xi16>) {
  %start = spirv.Constant dense<[0, 1, 2]> : !spirv.arm.tensor<3xi32>
  %size = spirv.Constant dense<[2, 3, 4]> : !spirv.arm.tensor<3xi32>
  // expected-error @+1 {{op failed to verify that all of {input1, output} have same element type}}
  %0 = spirv.Tosa.Slice %arg0, %start, %size : !spirv.arm.tensor<4x5x6xi8>, !spirv.arm.tensor<3xi32>, !spirv.arm.tensor<3xi32> -> !spirv.arm.tensor<2x3x4xi16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<2x3x4xi16>
}

spirv.ARM.Graph @slice_start_element_count_not_input_rank(%arg0: !spirv.arm.tensor<4x5x6xi8>) -> (!spirv.arm.tensor<2x3x4xi8>) {
  %start = spirv.Constant dense<[0, 1]> : !spirv.arm.tensor<2xi32>
  %size = spirv.Constant dense<[2, 3, 4]> : !spirv.arm.tensor<3xi32>
  // expected-error @+1 {{op failed to verify that the number of elements of start must be rank(input1)}}
  %0 = spirv.Tosa.Slice %arg0, %start, %size : !spirv.arm.tensor<4x5x6xi8>, !spirv.arm.tensor<2xi32>, !spirv.arm.tensor<3xi32> -> !spirv.arm.tensor<2x3x4xi8>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<2x3x4xi8>
}

spirv.ARM.Graph @slice_size_element_count_not_input_rank(%arg0: !spirv.arm.tensor<4x5x6xi8>) -> (!spirv.arm.tensor<2x3x4xi8>) {
  %start = spirv.Constant dense<[0, 1, 2]> : !spirv.arm.tensor<3xi32>
  %size = spirv.Constant dense<[2, 3]> : !spirv.arm.tensor<2xi32>
  // expected-error @+1 {{op failed to verify that the number of elements of size must be rank(input1)}}
  %0 = spirv.Tosa.Slice %arg0, %start, %size : !spirv.arm.tensor<4x5x6xi8>, !spirv.arm.tensor<3xi32>, !spirv.arm.tensor<2xi32> -> !spirv.arm.tensor<2x3x4xi8>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<2x3x4xi8>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Tile
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @tile_input_output_element_types_not_matching(%arg0: !spirv.arm.tensor<2x3x4xi8>) -> (!spirv.arm.tensor<4x3x8xi16>) {
  %multiples = spirv.Constant dense<[2, 1, 2]> : !spirv.arm.tensor<3xi32>
  // expected-error @+1 {{op failed to verify that all of {input1, output} have same element type}}
  %0 = spirv.Tosa.Tile %arg0, %multiples : !spirv.arm.tensor<2x3x4xi8>, !spirv.arm.tensor<3xi32> -> !spirv.arm.tensor<4x3x8xi16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<4x3x8xi16>
}

spirv.ARM.Graph @tile_input_output_ranks_not_matching(%arg0: !spirv.arm.tensor<2x3x4xi8>) -> (!spirv.arm.tensor<2x12xi8>) {
  %multiples = spirv.Constant dense<[1, 1, 1]> : !spirv.arm.tensor<3xi32>
  // expected-error @+1 {{op failed to verify that all of {input1, output} have same rank}}
  %0 = spirv.Tosa.Tile %arg0, %multiples : !spirv.arm.tensor<2x3x4xi8>, !spirv.arm.tensor<3xi32> -> !spirv.arm.tensor<2x12xi8>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<2x12xi8>
}

spirv.ARM.Graph @tile_multiples_element_count_not_input_rank(%arg0: !spirv.arm.tensor<2x3x4xi8>) -> (!spirv.arm.tensor<4x3x8xi8>) {
  %multiples = spirv.Constant dense<[2, 1]> : !spirv.arm.tensor<2xi32>
  // expected-error @+1 {{op failed to verify that the number of elements of multiples must be rank(input1)}}
  %0 = spirv.Tosa.Tile %arg0, %multiples : !spirv.arm.tensor<2x3x4xi8>, !spirv.arm.tensor<2xi32> -> !spirv.arm.tensor<4x3x8xi8>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<4x3x8xi8>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Transpose
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @transpose_input_output_element_types_not_matching(%arg0: !spirv.arm.tensor<2x3x4xi8>) -> (!spirv.arm.tensor<4x2x3xi16>) {
  // expected-error @+1 {{op failed to verify that all of {input1, output} have same element type}}
  %0 = spirv.Tosa.Transpose perms = [2, 0, 1], %arg0 : !spirv.arm.tensor<2x3x4xi8> -> !spirv.arm.tensor<4x2x3xi16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<4x2x3xi16>
}

spirv.ARM.Graph @transpose_input_output_ranks_not_matching(%arg0: !spirv.arm.tensor<2x3x4xi8>) -> (!spirv.arm.tensor<24xi8>) {
  // expected-error @+1 {{op failed to verify that all of {input1, output} have same rank}}
  %0 = spirv.Tosa.Transpose perms = [2, 0, 1], %arg0 : !spirv.arm.tensor<2x3x4xi8> -> !spirv.arm.tensor<24xi8>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<24xi8>
}

spirv.ARM.Graph @transpose_input_output_element_counts_not_matching(%arg0: !spirv.arm.tensor<2x3x4xi8>) -> (!spirv.arm.tensor<3x3x3xi8>) {
  // expected-error @+1 {{op failed to verify that all of {input1, output} have same element count}}
  %0 = spirv.Tosa.Transpose perms = [2, 0, 1], %arg0 : !spirv.arm.tensor<2x3x4xi8> -> !spirv.arm.tensor<3x3x3xi8>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<3x3x3xi8>
}

spirv.ARM.Graph @transpose_perms_element_count_not_input_rank(%arg0: !spirv.arm.tensor<2x3x4xi8>) -> (!spirv.arm.tensor<4x2x3xi8>) {
  // expected-error @+1 {{op failed to verify that the number of elements of perms must be rank(input1)}}
  %0 = spirv.Tosa.Transpose perms = [1, 0], %arg0 : !spirv.arm.tensor<2x3x4xi8> -> !spirv.arm.tensor<4x2x3xi8>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<4x2x3xi8>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Gather
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @gather_values_output_element_types_not_matching(%arg0: !spirv.arm.tensor<31x11x45xi32>, %arg1: !spirv.arm.tensor<31x15xi32>) -> (!spirv.arm.tensor<31x15x45xi16>) {
  // expected-error @+1 {{op failed to verify that all of {values, output} have same element type}}
  %0 = spirv.Tosa.Gather %arg0, %arg1 : !spirv.arm.tensor<31x11x45xi32>, !spirv.arm.tensor<31x15xi32> -> !spirv.arm.tensor<31x15x45xi16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<31x15x45xi16>
}

spirv.ARM.Graph @gather_shapes_not_matching(%arg0: !spirv.arm.tensor<31x11x45xi32>, %arg1: !spirv.arm.tensor<31x15xi32>) -> (!spirv.arm.tensor<30x15x44xi32>) {
  // expected-error @+1 {{op failed to verify that shapes of values, indices, and output must satisfy [N,K,C], [N,W], [N,W,C]}}
  %0 = spirv.Tosa.Gather %arg0, %arg1 : !spirv.arm.tensor<31x11x45xi32>, !spirv.arm.tensor<31x15xi32> -> !spirv.arm.tensor<30x15x44xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<30x15x44xi32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Scatter
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @scatter_values_in_input_values_out_element_types_not_matching(%arg0: !spirv.arm.tensor<34x28x54xi32>, %arg1: !spirv.arm.tensor<34x18xi32>, %arg2: !spirv.arm.tensor<34x18x54xi16>) -> (!spirv.arm.tensor<34x28x54xi32>) {
  // expected-error @+1 {{op failed to verify that all of {values_in, input, values_out} have same element type}}
  %0 = spirv.Tosa.Scatter %arg0, %arg1, %arg2 : !spirv.arm.tensor<34x28x54xi32>, !spirv.arm.tensor<34x18xi32>, !spirv.arm.tensor<34x18x54xi16> -> !spirv.arm.tensor<34x28x54xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<34x28x54xi32>
}

spirv.ARM.Graph @scatter_values_in_values_out_types_not_matching(%arg0: !spirv.arm.tensor<34x28x54xi32>, %arg1: !spirv.arm.tensor<34x18xi32>, %arg2: !spirv.arm.tensor<34x18x54xi32>) -> (!spirv.arm.tensor<35x28x54xi32>) {
  // expected-error @+1 {{op failed to verify that all of {values_in, values_out} have same type}}
  %0 = spirv.Tosa.Scatter %arg0, %arg1, %arg2 : !spirv.arm.tensor<34x28x54xi32>, !spirv.arm.tensor<34x18xi32>, !spirv.arm.tensor<34x18x54xi32> -> !spirv.arm.tensor<35x28x54xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<35x28x54xi32>
}

spirv.ARM.Graph @scatter_shapes_not_matching(%arg0: !spirv.arm.tensor<34x28x54xi32>, %arg1: !spirv.arm.tensor<34x18xi32>, %arg2: !spirv.arm.tensor<34x18x55xi32>) -> (!spirv.arm.tensor<34x28x54xi32>) {
  // expected-error @+1 {{op failed to verify that shapes of values_in, indices, and input must satisfy [N,K,C], [N,W], [N,W,C]}}
  %0 = spirv.Tosa.Scatter %arg0, %arg1, %arg2 : !spirv.arm.tensor<34x28x54xi32>, !spirv.arm.tensor<34x18xi32>, !spirv.arm.tensor<34x18x55xi32> -> !spirv.arm.tensor<34x28x54xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<34x28x54xi32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Resize
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @resize_i8_input_output_element_type_must_be_i8_or_i32(%arg0: !spirv.arm.tensor<1x1x31x55xi8>) -> (!spirv.arm.tensor<1x1x278x55xf16>) {
  %1 = spirv.Constant dense<[16, 1, 9, 1]> : !spirv.arm.tensor<4xi32>
  %2 = spirv.Constant dense<0> : !spirv.arm.tensor<2xi32>
  %3 = spirv.Constant dense<[0, 7]> : !spirv.arm.tensor<2xi32>
  // expected-error @+1 {{op failed to verify that if input has type 8-bit signless integer then output must have a type in [8-bit signless integer,32-bit signless integer]}}
  %4 = spirv.Tosa.Resize mode = <NearestNeighbor>, %arg0, %1, %2, %3 : !spirv.arm.tensor<1x1x31x55xi8>, !spirv.arm.tensor<4xi32>, !spirv.arm.tensor<2xi32>, !spirv.arm.tensor<2xi32> -> !spirv.arm.tensor<1x1x278x55xf16>
  spirv.ARM.GraphOutputs %4 : !spirv.arm.tensor<1x1x278x55xf16>
}

spirv.ARM.Graph @resize_i16_input_output_element_type_must_be_i16_or_i64(%arg0: !spirv.arm.tensor<1x1x31x55xi16>) -> (!spirv.arm.tensor<1x1x278x55xi32>) {
  %1 = spirv.Constant dense<[16, 1, 9, 1]> : !spirv.arm.tensor<4xi32>
  %2 = spirv.Constant dense<0> : !spirv.arm.tensor<2xi32>
  %3 = spirv.Constant dense<[0, 7]> : !spirv.arm.tensor<2xi32>
  // expected-error @+1 {{op failed to verify that if input has type 16-bit signless integer then output must have a type in [16-bit signless integer,64-bit signless integer]}}
  %4 = spirv.Tosa.Resize mode = <NearestNeighbor>, %arg0, %1, %2, %3 : !spirv.arm.tensor<1x1x31x55xi16>, !spirv.arm.tensor<4xi32>, !spirv.arm.tensor<2xi32>, !spirv.arm.tensor<2xi32> -> !spirv.arm.tensor<1x1x278x55xi32>
  spirv.ARM.GraphOutputs %4 : !spirv.arm.tensor<1x1x278x55xi32>
}

spirv.ARM.Graph @resize_f16_input_output_element_type_must_be_f16(%arg0: !spirv.arm.tensor<1x48x33x63xf16>) -> (!spirv.arm.tensor<1x753x297x63xf32>) {
  %1 = spirv.Constant dense<[16, 1, 9, 1]> : !spirv.arm.tensor<4xi32>
  %2 = spirv.Constant dense<0> : !spirv.arm.tensor<2xi32>
  %3 = spirv.Constant dense<[0, 8]> : !spirv.arm.tensor<2xi32>
  // expected-error @+1 {{op failed to verify that if input has type 16-bit float then output must have a type in [16-bit float]}}
  %4 = spirv.Tosa.Resize mode = <Bilinear>, %arg0, %1, %2, %3 : !spirv.arm.tensor<1x48x33x63xf16>, !spirv.arm.tensor<4xi32>, !spirv.arm.tensor<2xi32>, !spirv.arm.tensor<2xi32> -> !spirv.arm.tensor<1x753x297x63xf32>
  spirv.ARM.GraphOutputs %4 : !spirv.arm.tensor<1x753x297x63xf32>
}

spirv.ARM.Graph @resize_f32_input_output_element_type_must_be_f32(%arg0: !spirv.arm.tensor<1x48x33x63xf32>) -> (!spirv.arm.tensor<1x753x297x63xf16>) {
  %1 = spirv.Constant dense<[16, 1, 9, 1]> : !spirv.arm.tensor<4xi32>
  %2 = spirv.Constant dense<0> : !spirv.arm.tensor<2xi32>
  %3 = spirv.Constant dense<[0, 8]> : !spirv.arm.tensor<2xi32>
  // expected-error @+1 {{op failed to verify that if input has type 32-bit float then output must have a type in [32-bit float]}}
  %4 = spirv.Tosa.Resize mode = <Bilinear>, %arg0, %1, %2, %3 : !spirv.arm.tensor<1x48x33x63xf32>, !spirv.arm.tensor<4xi32>, !spirv.arm.tensor<2xi32>, !spirv.arm.tensor<2xi32> -> !spirv.arm.tensor<1x753x297x63xf16>
  spirv.ARM.GraphOutputs %4 : !spirv.arm.tensor<1x753x297x63xf16>
}

spirv.ARM.Graph @resize_bf16_input_output_element_type_must_be_bf16(%arg0: !spirv.arm.tensor<1x48x33x63xbf16>) -> (!spirv.arm.tensor<1x753x297x63xf32>) {
  %1 = spirv.Constant dense<[16, 1, 9, 1]> : !spirv.arm.tensor<4xi32>
  %2 = spirv.Constant dense<0> : !spirv.arm.tensor<2xi32>
  %3 = spirv.Constant dense<[0, 8]> : !spirv.arm.tensor<2xi32>
  // expected-error @+1 {{op failed to verify that if input has type bfloat16 type then output must have a type in [bfloat16 type]}}
  %4 = spirv.Tosa.Resize mode = <Bilinear>, %arg0, %1, %2, %3 : !spirv.arm.tensor<1x48x33x63xbf16>, !spirv.arm.tensor<4xi32>, !spirv.arm.tensor<2xi32>, !spirv.arm.tensor<2xi32> -> !spirv.arm.tensor<1x753x297x63xf32>
  spirv.ARM.GraphOutputs %4 : !spirv.arm.tensor<1x753x297x63xf32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Cast
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @cast_input_output_shapes_not_matching(%arg0: !spirv.arm.tensor<2x3x4xi8>) -> (!spirv.arm.tensor<2x3x5xi32>) {
  // expected-error @+1 {{op failed to verify that all of {input, output} have same shape}}
  %0 = spirv.Tosa.Cast %arg0 : !spirv.arm.tensor<2x3x4xi8> -> !spirv.arm.tensor<2x3x5xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<2x3x5xi32>
}

spirv.ARM.Graph @cast_f16_to_bf16_not_supported(%arg0: !spirv.arm.tensor<2x3x4xf16>) -> (!spirv.arm.tensor<2x3x4xbf16>) {
  // expected-error @+1 {{op failed to verify that if input has type 16-bit float then output must have a type in [32-bit float,16-bit signless integer,32-bit signless integer,8-bit signless integer]}}
  %0 = spirv.Tosa.Cast %arg0 : !spirv.arm.tensor<2x3x4xf16> -> !spirv.arm.tensor<2x3x4xbf16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<2x3x4xbf16>
}

spirv.ARM.Graph @cast_f16_to_f16_not_supported(%arg0: !spirv.arm.tensor<2x3x4xf16>) -> (!spirv.arm.tensor<2x3x4xf16>) {
  // expected-error @+1 {{op failed to verify that if input has type 16-bit float then output must have a type in [32-bit float,16-bit signless integer,32-bit signless integer,8-bit signless integer]}}
  %0 = spirv.Tosa.Cast %arg0 : !spirv.arm.tensor<2x3x4xf16> -> !spirv.arm.tensor<2x3x4xf16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<2x3x4xf16>
}

spirv.ARM.Graph @cast_f32_to_f32_not_supported(%arg0: !spirv.arm.tensor<2x3x4xf32>) -> (!spirv.arm.tensor<2x3x4xf32>) {
  // expected-error @+1 {{op failed to verify that if input has type 32-bit float then output must have a type in [16-bit float,16-bit signless integer,32-bit signless integer,8-bit signless integer,bfloat16 type]}}
  %0 = spirv.Tosa.Cast %arg0 : !spirv.arm.tensor<2x3x4xf32> -> !spirv.arm.tensor<2x3x4xf32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<2x3x4xf32>
}

spirv.ARM.Graph @cast_i8_to_i8_not_supported(%arg0: !spirv.arm.tensor<2x3x4xi8>) -> (!spirv.arm.tensor<2x3x4xi8>) {
  // expected-error @+1 {{op failed to verify that if input has type 8-bit signless integer then output must have a type in [16-bit float,32-bit float,16-bit signless integer,32-bit signless integer,bool,bfloat16 type]}}
  %0 = spirv.Tosa.Cast %arg0 : !spirv.arm.tensor<2x3x4xi8> -> !spirv.arm.tensor<2x3x4xi8>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<2x3x4xi8>
}

spirv.ARM.Graph @cast_i16_to_i16_not_supported(%arg0: !spirv.arm.tensor<2x3x4xi16>) -> (!spirv.arm.tensor<2x3x4xi16>) {
  // expected-error @+1 {{op failed to verify that if input has type 16-bit signless integer then output must have a type in [16-bit float,32-bit float,32-bit signless integer,8-bit signless integer,bool,bfloat16 type]}}
  %0 = spirv.Tosa.Cast %arg0 : !spirv.arm.tensor<2x3x4xi16> -> !spirv.arm.tensor<2x3x4xi16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<2x3x4xi16>
}

spirv.ARM.Graph @cast_i32_to_i32_not_supported(%arg0: !spirv.arm.tensor<2x3x4xi32>) -> (!spirv.arm.tensor<2x3x4xi32>) {
  // expected-error @+1 {{op failed to verify that if input has type 32-bit signless integer then output must have a type in [16-bit float,32-bit float,16-bit signless integer,8-bit signless integer,bool,bfloat16 type]}}
  %0 = spirv.Tosa.Cast %arg0 : !spirv.arm.tensor<2x3x4xi32> -> !spirv.arm.tensor<2x3x4xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<2x3x4xi32>
}

spirv.ARM.Graph @cast_bool_to_f32_not_supported(%arg0: !spirv.arm.tensor<2x3x4xi1>) -> (!spirv.arm.tensor<2x3x4xf32>) {
  // expected-error @+1 {{op failed to verify that if input has type bool then output must have a type in [16-bit signless integer,32-bit signless integer,8-bit signless integer]}}
  %0 = spirv.Tosa.Cast %arg0 : !spirv.arm.tensor<2x3x4xi1> -> !spirv.arm.tensor<2x3x4xf32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<2x3x4xf32>
}

spirv.ARM.Graph @cast_bool_to_f16_not_supported(%arg0: !spirv.arm.tensor<2x3x4xi1>) -> (!spirv.arm.tensor<2x3x4xf16>) {
  // expected-error @+1 {{op failed to verify that if input has type bool then output must have a type in [16-bit signless integer,32-bit signless integer,8-bit signless integer]}}
  %0 = spirv.Tosa.Cast %arg0 : !spirv.arm.tensor<2x3x4xi1> -> !spirv.arm.tensor<2x3x4xf16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<2x3x4xf16>
}

spirv.ARM.Graph @cast_bool_to_bf16_not_supported(%arg0: !spirv.arm.tensor<2x3x4xi1>) -> (!spirv.arm.tensor<2x3x4xbf16>) {
  // expected-error @+1 {{op failed to verify that if input has type bool then output must have a type in [16-bit signless integer,32-bit signless integer,8-bit signless integer]}}
  %0 = spirv.Tosa.Cast %arg0 : !spirv.arm.tensor<2x3x4xi1> -> !spirv.arm.tensor<2x3x4xbf16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<2x3x4xbf16>
}

spirv.ARM.Graph @cast_bf16_to_f16_not_supported(%arg0: !spirv.arm.tensor<2x3x4xbf16>) -> (!spirv.arm.tensor<2x3x4xf16>) {
  // expected-error @+1 {{op failed to verify that if input has type bfloat16 type then output must have a type in [32-bit float,16-bit signless integer,32-bit signless integer,8-bit signless integer]}}
  %0 = spirv.Tosa.Cast %arg0 : !spirv.arm.tensor<2x3x4xbf16> -> !spirv.arm.tensor<2x3x4xf16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<2x3x4xf16>
}

spirv.ARM.Graph @cast_bf16_to_bf16_not_supported(%arg0: !spirv.arm.tensor<2x3x4xbf16>) -> (!spirv.arm.tensor<2x3x4xbf16>) {
  // expected-error @+1 {{op failed to verify that if input has type bfloat16 type then output must have a type in [32-bit float,16-bit signless integer,32-bit signless integer,8-bit signless integer]}}
  %0 = spirv.Tosa.Cast %arg0 : !spirv.arm.tensor<2x3x4xbf16> -> !spirv.arm.tensor<2x3x4xbf16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<2x3x4xbf16>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Rescale
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @rescale_input_output_shapes_not_matching(%arg0: !spirv.arm.tensor<2x3x4xi16>) -> (!spirv.arm.tensor<2x3x5xi16>) {
  %1 = spirv.Constant dense<1> : !spirv.arm.tensor<1xi16>
  %2 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi8>
  %3 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi16>
  %4 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi16>
  // expected-error @+1 {{op failed to verify that all of {input, output} have same shape}}
  %5 = spirv.Tosa.Rescale scale32 = false, rounding_mode = <SingleRound>, per_channel = false, input_unsigned = false, output_unsigned = false, %arg0, %1, %2, %3, %4 : !spirv.arm.tensor<2x3x4xi16>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi16> -> !spirv.arm.tensor<2x3x5xi16>
  spirv.ARM.GraphOutputs %5 : !spirv.arm.tensor<2x3x5xi16>
}

spirv.ARM.Graph @rescale_input_and_input_zp_element_types_not_matching(%arg0: !spirv.arm.tensor<2x3x4xi16>) -> (!spirv.arm.tensor<2x3x4xi16>) {
  %1 = spirv.Constant dense<1> : !spirv.arm.tensor<1xi16>
  %2 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi8>
  %3 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi8>
  %4 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi16>
  // expected-error @+1 {{op failed to verify that all of {input, input_zp} have same element type}}
  %5 = spirv.Tosa.Rescale scale32 = false, rounding_mode = <SingleRound>, per_channel = false, input_unsigned = false, output_unsigned = false, %arg0, %1, %2, %3, %4 : !spirv.arm.tensor<2x3x4xi16>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi16> -> !spirv.arm.tensor<2x3x4xi16>
  spirv.ARM.GraphOutputs %5 : !spirv.arm.tensor<2x3x4xi16>
}

spirv.ARM.Graph @rescale_output_and_output_zp_element_types_not_matching(%arg0: !spirv.arm.tensor<2x3x4xi16>) -> (!spirv.arm.tensor<2x3x4xi16>) {
  %1 = spirv.Constant dense<1> : !spirv.arm.tensor<1xi16>
  %2 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi8>
  %3 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi16>
  %4 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi8>
  // expected-error @+1 {{op failed to verify that all of {output, output_zp} have same element type}}
  %5 = spirv.Tosa.Rescale scale32 = false, rounding_mode = <SingleRound>, per_channel = false, input_unsigned = false, output_unsigned = false, %arg0, %1, %2, %3, %4 : !spirv.arm.tensor<2x3x4xi16>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<2x3x4xi16>
  spirv.ARM.GraphOutputs %5 : !spirv.arm.tensor<2x3x4xi16>
}

spirv.ARM.Graph @rescale_scale32_true_requires_i32_multiplier(%arg0: !spirv.arm.tensor<2x3x4xi16>) -> (!spirv.arm.tensor<2x3x4xi16>) {
  %1 = spirv.Constant dense<1> : !spirv.arm.tensor<1xi16>
  %2 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi8>
  %3 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi16>
  %4 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi16>
  // expected-error @+1 {{op failed to verify that multiplier must have element type i32 when scale32 is true, otherwise i16}}
  %5 = spirv.Tosa.Rescale scale32 = true, rounding_mode = <SingleRound>, per_channel = false, input_unsigned = false, output_unsigned = false, %arg0, %1, %2, %3, %4 : !spirv.arm.tensor<2x3x4xi16>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi16> -> !spirv.arm.tensor<2x3x4xi16>
  spirv.ARM.GraphOutputs %5 : !spirv.arm.tensor<2x3x4xi16>
}

spirv.ARM.Graph @rescale_scale32_false_requires_i16_multiplier(%arg0: !spirv.arm.tensor<2x3x4xi16>) -> (!spirv.arm.tensor<2x3x4xi16>) {
  %1 = spirv.Constant dense<1> : !spirv.arm.tensor<1xi32>
  %2 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi8>
  %3 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi16>
  %4 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi16>
  // expected-error @+1 {{op failed to verify that multiplier must have element type i32 when scale32 is true, otherwise i16}}
  %5 = spirv.Tosa.Rescale scale32 = false, rounding_mode = <SingleRound>, per_channel = false, input_unsigned = false, output_unsigned = false, %arg0, %1, %2, %3, %4 : !spirv.arm.tensor<2x3x4xi16>, !spirv.arm.tensor<1xi32>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi16> -> !spirv.arm.tensor<2x3x4xi16>
  spirv.ARM.GraphOutputs %5 : !spirv.arm.tensor<2x3x4xi16>
}

spirv.ARM.Graph @rescale_per_channel_true_requires_multiplier_length_last_dimension(%arg0: !spirv.arm.tensor<2x3x4xi16>) -> (!spirv.arm.tensor<2x3x4xi16>) {
  %1 = spirv.Constant dense<[1]> : !spirv.arm.tensor<1xi16>
  %2 = spirv.Constant dense<[0, 0]> : !spirv.arm.tensor<2xi8>
  %3 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi16>
  %4 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi16>
  // expected-error @+1 {{op failed to verify that multiplier must have length input_shape[rank(input) - 1] when per_channel is true, otherwise length 1}}
  %5 = spirv.Tosa.Rescale scale32 = false, rounding_mode = <SingleRound>, per_channel = true, input_unsigned = false, output_unsigned = false, %arg0, %1, %2, %3, %4 : !spirv.arm.tensor<2x3x4xi16>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<2xi8>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi16> -> !spirv.arm.tensor<2x3x4xi16>
  spirv.ARM.GraphOutputs %5 : !spirv.arm.tensor<2x3x4xi16>
}

spirv.ARM.Graph @rescale_per_channel_true_requires_shift_length_last_dimension(%arg0: !spirv.arm.tensor<2x3x4xi16>) -> (!spirv.arm.tensor<2x3x4xi16>) {
  %1 = spirv.Constant dense<[1, 1, 1, 1]> : !spirv.arm.tensor<4xi16>
  %2 = spirv.Constant dense<[0]> : !spirv.arm.tensor<1xi8>
  %3 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi16>
  %4 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi16>
  // expected-error @+1 {{op failed to verify that shift must have length input_shape[rank(input) - 1] when per_channel is true, otherwise length 1}}
  %5 = spirv.Tosa.Rescale scale32 = false, rounding_mode = <SingleRound>, per_channel = true, input_unsigned = false, output_unsigned = false, %arg0, %1, %2, %3, %4 : !spirv.arm.tensor<2x3x4xi16>, !spirv.arm.tensor<4xi16>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi16> -> !spirv.arm.tensor<2x3x4xi16>
  spirv.ARM.GraphOutputs %5 : !spirv.arm.tensor<2x3x4xi16>
}

spirv.ARM.Graph @rescale_per_channel_false_requires_multiplier_length_one(%arg0: !spirv.arm.tensor<2x3x4xi16>) -> (!spirv.arm.tensor<2x3x4xi16>) {
  %1 = spirv.Constant dense<[1, 1]> : !spirv.arm.tensor<2xi16>
  %2 = spirv.Constant dense<[0]> : !spirv.arm.tensor<1xi8>
  %3 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi16>
  %4 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi16>
  // expected-error @+1 {{op failed to verify that multiplier must have length input_shape[rank(input) - 1] when per_channel is true, otherwise length 1}}
  %5 = spirv.Tosa.Rescale scale32 = false, rounding_mode = <SingleRound>, per_channel = false, input_unsigned = false, output_unsigned = false, %arg0, %1, %2, %3, %4 : !spirv.arm.tensor<2x3x4xi16>, !spirv.arm.tensor<2xi16>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi16> -> !spirv.arm.tensor<2x3x4xi16>
  spirv.ARM.GraphOutputs %5 : !spirv.arm.tensor<2x3x4xi16>
}

spirv.ARM.Graph @rescale_per_channel_false_requires_shift_length_one(%arg0: !spirv.arm.tensor<2x3x4xi16>) -> (!spirv.arm.tensor<2x3x4xi16>) {
  %1 = spirv.Constant dense<[1]> : !spirv.arm.tensor<1xi16>
  %2 = spirv.Constant dense<[0, 0]> : !spirv.arm.tensor<2xi8>
  %3 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi16>
  %4 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi16>
  // expected-error @+1 {{op failed to verify that shift must have length input_shape[rank(input) - 1] when per_channel is true, otherwise length 1}}
  %5 = spirv.Tosa.Rescale scale32 = false, rounding_mode = <SingleRound>, per_channel = false, input_unsigned = false, output_unsigned = false, %arg0, %1, %2, %3, %4 : !spirv.arm.tensor<2x3x4xi16>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<2xi8>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi16> -> !spirv.arm.tensor<2x3x4xi16>
  spirv.ARM.GraphOutputs %5 : !spirv.arm.tensor<2x3x4xi16>
}

spirv.ARM.Graph @rescale_i8_input_requires_i8_i16_or_i32_output(%arg0: !spirv.arm.tensor<2x3x4xi8>) -> (!spirv.arm.tensor<2x3x4xi64>) {
  %1 = spirv.Constant dense<1> : !spirv.arm.tensor<1xi16>
  %2 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi8>
  %3 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi8>
  %4 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi64>
  // expected-error @+1 {{op failed to verify that if input has type 8-bit signless integer then output must have a type in [8-bit signless integer,16-bit signless integer,32-bit signless integer]}}
  %5 = spirv.Tosa.Rescale scale32 = false, rounding_mode = <SingleRound>, per_channel = false, input_unsigned = false, output_unsigned = false, %arg0, %1, %2, %3, %4 : !spirv.arm.tensor<2x3x4xi8>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi64> -> !spirv.arm.tensor<2x3x4xi64>
  spirv.ARM.GraphOutputs %5 : !spirv.arm.tensor<2x3x4xi64>
}

spirv.ARM.Graph @rescale_i16_input_requires_i8_i16_or_i32_output(%arg0: !spirv.arm.tensor<2x3x4xi16>) -> (!spirv.arm.tensor<2x3x4xi64>) {
  %1 = spirv.Constant dense<1> : !spirv.arm.tensor<1xi16>
  %2 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi8>
  %3 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi16>
  %4 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi64>
  // expected-error @+1 {{op failed to verify that if input has type 16-bit signless integer then output must have a type in [8-bit signless integer,16-bit signless integer,32-bit signless integer]}}
  %5 = spirv.Tosa.Rescale scale32 = false, rounding_mode = <SingleRound>, per_channel = false, input_unsigned = false, output_unsigned = false, %arg0, %1, %2, %3, %4 : !spirv.arm.tensor<2x3x4xi16>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi64> -> !spirv.arm.tensor<2x3x4xi64>
  spirv.ARM.GraphOutputs %5 : !spirv.arm.tensor<2x3x4xi64>
}

spirv.ARM.Graph @rescale_i32_input_requires_i8_i16_or_i32_output(%arg0: !spirv.arm.tensor<2x3x4xi32>) -> (!spirv.arm.tensor<2x3x4xi64>) {
  %1 = spirv.Constant dense<1> : !spirv.arm.tensor<1xi16>
  %2 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi8>
  %3 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi32>
  %4 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi64>
  // expected-error @+1 {{op failed to verify that if input has type 32-bit signless integer then output must have a type in [8-bit signless integer,16-bit signless integer,32-bit signless integer]}}
  %5 = spirv.Tosa.Rescale scale32 = false, rounding_mode = <SingleRound>, per_channel = false, input_unsigned = false, output_unsigned = false, %arg0, %1, %2, %3, %4 : !spirv.arm.tensor<2x3x4xi32>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi32>, !spirv.arm.tensor<1xi64> -> !spirv.arm.tensor<2x3x4xi64>
  spirv.ARM.GraphOutputs %5 : !spirv.arm.tensor<2x3x4xi64>
}

spirv.ARM.Graph @rescale_i64_input_requires_i8_i16_or_i32_output(%arg0: !spirv.arm.tensor<2x3x4xi64>) -> (!spirv.arm.tensor<2x3x4xi64>) {
  %1 = spirv.Constant dense<1> : !spirv.arm.tensor<1xi16>
  %2 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi8>
  %3 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi64>
  %4 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi64>
  // expected-error @+1 {{op failed to verify that if input has type 64-bit signless integer then output must have a type in [8-bit signless integer,16-bit signless integer,32-bit signless integer]}}
  %5 = spirv.Tosa.Rescale scale32 = false, rounding_mode = <SingleRound>, per_channel = false, input_unsigned = false, output_unsigned = false, %arg0, %1, %2, %3, %4 : !spirv.arm.tensor<2x3x4xi64>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi64>, !spirv.arm.tensor<1xi64> -> !spirv.arm.tensor<2x3x4xi64>
  spirv.ARM.GraphOutputs %5 : !spirv.arm.tensor<2x3x4xi64>
}

spirv.ARM.Graph @rescale_input_unsigned_true_requires_i8_or_i16_input(%arg0: !spirv.arm.tensor<2x3x4xi32>) -> (!spirv.arm.tensor<2x3x4xi16>) {
  %1 = spirv.Constant dense<1> : !spirv.arm.tensor<1xi16>
  %2 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi8>
  %3 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi32>
  %4 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi16>
  // expected-error @+1 {{op failed to verify that if input_unsigned is true then input must have a type in [8-bit signless integer,16-bit signless integer]}}
  %5 = spirv.Tosa.Rescale scale32 = false, rounding_mode = <SingleRound>, per_channel = false, input_unsigned = true, output_unsigned = false, %arg0, %1, %2, %3, %4 : !spirv.arm.tensor<2x3x4xi32>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi32>, !spirv.arm.tensor<1xi16> -> !spirv.arm.tensor<2x3x4xi16>
  spirv.ARM.GraphOutputs %5 : !spirv.arm.tensor<2x3x4xi16>
}

spirv.ARM.Graph @rescale_output_unsigned_true_requires_i8_or_i16_input(%arg0: !spirv.arm.tensor<2x3x4xi32>) -> (!spirv.arm.tensor<2x3x4xi16>) {
  %1 = spirv.Constant dense<1> : !spirv.arm.tensor<1xi16>
  %2 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi8>
  %3 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi32>
  %4 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi16>
  // expected-error @+1 {{op failed to verify that if output_unsigned is true then input must have a type in [8-bit signless integer,16-bit signless integer]}}
  %5 = spirv.Tosa.Rescale scale32 = false, rounding_mode = <SingleRound>, per_channel = false, input_unsigned = false, output_unsigned = true, %arg0, %1, %2, %3, %4 : !spirv.arm.tensor<2x3x4xi32>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi32>, !spirv.arm.tensor<1xi16> -> !spirv.arm.tensor<2x3x4xi16>
  spirv.ARM.GraphOutputs %5 : !spirv.arm.tensor<2x3x4xi16>
}

spirv.ARM.Graph @rescale_input_unsigned_true_requires_i8_or_i16_output(%arg0: !spirv.arm.tensor<2x3x4xi16>) -> (!spirv.arm.tensor<2x3x4xi32>) {
  %1 = spirv.Constant dense<1> : !spirv.arm.tensor<1xi16>
  %2 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi8>
  %3 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi16>
  %4 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi32>
  // expected-error @+1 {{op failed to verify that if input_unsigned is true then output must have a type in [8-bit signless integer,16-bit signless integer]}}
  %5 = spirv.Tosa.Rescale scale32 = false, rounding_mode = <SingleRound>, per_channel = false, input_unsigned = true, output_unsigned = false, %arg0, %1, %2, %3, %4 : !spirv.arm.tensor<2x3x4xi16>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi32> -> !spirv.arm.tensor<2x3x4xi32>
  spirv.ARM.GraphOutputs %5 : !spirv.arm.tensor<2x3x4xi32>
}

spirv.ARM.Graph @rescale_output_unsigned_true_requires_i8_or_i16_output(%arg0: !spirv.arm.tensor<2x3x4xi16>) -> (!spirv.arm.tensor<2x3x4xi32>) {
  %1 = spirv.Constant dense<1> : !spirv.arm.tensor<1xi16>
  %2 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi8>
  %3 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi16>
  %4 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi32>
  // expected-error @+1 {{op failed to verify that if output_unsigned is true then output must have a type in [8-bit signless integer,16-bit signless integer]}}
  %5 = spirv.Tosa.Rescale scale32 = false, rounding_mode = <SingleRound>, per_channel = false, input_unsigned = false, output_unsigned = true, %arg0, %1, %2, %3, %4 : !spirv.arm.tensor<2x3x4xi16>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi32> -> !spirv.arm.tensor<2x3x4xi32>
  spirv.ARM.GraphOutputs %5 : !spirv.arm.tensor<2x3x4xi32>
}
