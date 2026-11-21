// RUN: mlir-opt %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spirv.TOSA.Rescale
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @rescale_per_channel_dynamic_input_last_dimension(%arg0: !spirv.arm.tensor<?x?x?xi16>) -> (!spirv.arm.tensor<?x?x?xi16>) {
  %1 = spirv.Constant dense<[1]> : !spirv.arm.tensor<1xi16>
  %2 = spirv.Constant dense<[0, 0]> : !spirv.arm.tensor<2xi8>
  %3 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi16>
  %4 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi16>
  // CHECK: {{%.*}} = spirv.Tosa.Rescale scale32 = false, rounding_mode = <SingleRound>, per_channel = true, input_unsigned = false, output_unsigned = false, %arg0, {{%.*}}, {{%.*}}, {{%.*}}, {{%.*}} : !spirv.arm.tensor<?x?x?xi16>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<2xi8>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi16> -> !spirv.arm.tensor<?x?x?xi16>
  %5 = spirv.Tosa.Rescale scale32 = false, rounding_mode = <SingleRound>, per_channel = true, input_unsigned = false, output_unsigned = false, %arg0, %1, %2, %3, %4 : !spirv.arm.tensor<?x?x?xi16>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<2xi8>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi16> -> !spirv.arm.tensor<?x?x?xi16>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<?x?x?xi16>
  spirv.ARM.GraphOutputs %5 : !spirv.arm.tensor<?x?x?xi16>
}

spirv.ARM.Graph @rescale_per_channel_dynamic_multiplier_and_shift_length(%arg0: !spirv.arm.tensor<2x3x4xi16>, %multiplier: !spirv.arm.tensor<?xi16>, %shift: !spirv.arm.tensor<?xi8>) -> (!spirv.arm.tensor<2x3x4xi16>) {
  %3 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi16>
  %4 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi16>
  // CHECK: {{%.*}} = spirv.Tosa.Rescale scale32 = false, rounding_mode = <SingleRound>, per_channel = true, input_unsigned = false, output_unsigned = false, %arg0, %arg1, %arg2, {{%.*}}, {{%.*}} : !spirv.arm.tensor<2x3x4xi16>, !spirv.arm.tensor<?xi16>, !spirv.arm.tensor<?xi8>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi16> -> !spirv.arm.tensor<2x3x4xi16>
  %5 = spirv.Tosa.Rescale scale32 = false, rounding_mode = <SingleRound>, per_channel = true, input_unsigned = false, output_unsigned = false, %arg0, %multiplier, %shift, %3, %4 : !spirv.arm.tensor<2x3x4xi16>, !spirv.arm.tensor<?xi16>, !spirv.arm.tensor<?xi8>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi16> -> !spirv.arm.tensor<2x3x4xi16>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<2x3x4xi16>
  spirv.ARM.GraphOutputs %5 : !spirv.arm.tensor<2x3x4xi16>
}
