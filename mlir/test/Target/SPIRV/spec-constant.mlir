// RUN: mlir-translate -no-implicit-module -test-spirv-roundtrip -split-input-file %s | FileCheck %s

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  // CHECK: spirv.SpecConstant @sc_true = true
  spirv.SpecConstant @sc_true = true
  // CHECK: spirv.SpecConstant @sc_false spec_id(1) = false
  spirv.SpecConstant @sc_false spec_id(1) = false

  // CHECK: spirv.SpecConstant @sc_int = -5 : i32
  spirv.SpecConstant @sc_int = -5 : i32

  // CHECK: spirv.SpecConstant @sc_float spec_id(5) = 1.000000e+00 : f32
  spirv.SpecConstant @sc_float spec_id(5) = 1. : f32

  // CHECK: spirv.SpecConstantComposite @scc (@sc_int, @sc_int) : !spirv.array<2 x i32>
  spirv.SpecConstantComposite @scc (@sc_int, @sc_int) : !spirv.array<2 x i32>

  // CHECK-LABEL: @use
  spirv.func @use() -> (i32) "None" {
    // We materialize a `spirv.mlir.referenceof` op at every use of a
    // specialization constant in the deserializer. So two ops here.
    // CHECK: %[[USE1:.*]] = spirv.mlir.referenceof @sc_int : i32
    // CHECK: %[[USE2:.*]] = spirv.mlir.referenceof @sc_int : i32
    // CHECK: spirv.IAdd %[[USE1]], %[[USE2]]

    %0 = spirv.mlir.referenceof @sc_int : i32
    %1 = spirv.IAdd %0, %0 : i32
    spirv.ReturnValue %1 : i32
  }

  // CHECK-LABEL: @use
  spirv.func @use_composite() -> (i32) "None" {
    // We materialize a `spirv.mlir.referenceof` op at every use of a
    // specialization constant in the deserializer. So two ops here.
    // CHECK: %[[USE1:.*]] = spirv.mlir.referenceof @scc : !spirv.array<2 x i32>
    // CHECK: %[[ITM0:.*]] = spirv.CompositeExtract %[[USE1]][0 : i32] : !spirv.array<2 x i32>
    // CHECK: %[[USE2:.*]] = spirv.mlir.referenceof @scc : !spirv.array<2 x i32>
    // CHECK: %[[ITM1:.*]] = spirv.CompositeExtract %[[USE2]][1 : i32] : !spirv.array<2 x i32>
    // CHECK: spirv.IAdd %[[ITM0]], %[[ITM1]]

    %0 = spirv.mlir.referenceof @scc : !spirv.array<2 x i32>
    %1 = spirv.CompositeExtract %0[0 : i32] : !spirv.array<2 x i32>
    %2 = spirv.CompositeExtract %0[1 : i32] : !spirv.array<2 x i32>
    %3 = spirv.IAdd %1, %2 : i32
    spirv.ReturnValue %3 : i32
  }
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {

  spirv.SpecConstant @sc_f32_1 = 1.5 : f32
  spirv.SpecConstant @sc_f32_2 = 2.5 : f32
  spirv.SpecConstant @sc_f32_3 = 3.5 : f32

  spirv.SpecConstant @sc_i32_1 = 1   : i32

  // CHECK: spirv.SpecConstantComposite @scc_array (@sc_f32_1, @sc_f32_2, @sc_f32_3) : !spirv.array<3 x f32>
  spirv.SpecConstantComposite @scc_array (@sc_f32_1, @sc_f32_2, @sc_f32_3) : !spirv.array<3 x f32>

  // CHECK: spirv.SpecConstantComposite @scc_struct (@sc_i32_1, @sc_f32_2, @sc_f32_3) : !spirv.struct<(i32, f32, f32)>
  spirv.SpecConstantComposite @scc_struct (@sc_i32_1, @sc_f32_2, @sc_f32_3) : !spirv.struct<(i32, f32, f32)>

  // CHECK: spirv.SpecConstantComposite @scc_vector (@sc_f32_1, @sc_f32_2, @sc_f32_3) : vector<3xf32>
  spirv.SpecConstantComposite @scc_vector (@sc_f32_1, @sc_f32_2, @sc_f32_3) : vector<3 x f32>
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {

  spirv.SpecConstant @sc_f32_1 = 1.5 : f32
  spirv.SpecConstant @sc_f32_2 = 2.5 : f32
  spirv.SpecConstant @sc_f32_3 = 3.5 : f32

  spirv.SpecConstant @sc_i32_1 = 1   : i32

  // CHECK: spirv.SpecConstantComposite @scc_array (@sc_f32_1, @sc_f32_2, @sc_f32_3) : !spirv.array<3 x f32>
  spirv.SpecConstantComposite @scc_array (@sc_f32_1, @sc_f32_2, @sc_f32_3) : !spirv.array<3 x f32>

  // CHECK: spirv.SpecConstantComposite @scc_struct (@sc_i32_1, @sc_f32_2, @sc_f32_3) : !spirv.struct<(i32, f32, f32)>
  spirv.SpecConstantComposite @scc_struct (@sc_i32_1, @sc_f32_2, @sc_f32_3) : !spirv.struct<(i32, f32, f32)>

  // CHECK: spirv.SpecConstantComposite @scc_vector (@sc_f32_1, @sc_f32_2, @sc_f32_3) : vector<3xf32>
  spirv.SpecConstantComposite @scc_vector (@sc_f32_1, @sc_f32_2, @sc_f32_3) : vector<3 x f32>
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {

  spirv.SpecConstant @sc_i32_1 = 1 : i32

  spirv.func @use_composite() -> (i32) "None" {
    // CHECK: [[USE1:%.*]] = spirv.mlir.referenceof @sc_i32_1 : i32
    // CHECK: [[USE2:%.*]] = spirv.Constant 0 : i32

    // CHECK: [[RES1:%.*]] = spirv.SpecConstantOperation wraps "spirv.ISub"([[USE1]], [[USE2]]) : (i32, i32) -> i32

    // CHECK: [[USE3:%.*]] = spirv.mlir.referenceof @sc_i32_1 : i32
    // CHECK: [[USE4:%.*]] = spirv.Constant 0 : i32

    // CHECK: [[RES2:%.*]] = spirv.SpecConstantOperation wraps "spirv.ISub"([[USE3]], [[USE4]]) : (i32, i32) -> i32

    %0 = spirv.mlir.referenceof @sc_i32_1 : i32
    %1 = spirv.Constant 0 : i32
    %2 = spirv.SpecConstantOperation wraps "spirv.ISub"(%0, %1) : (i32, i32) -> i32

    // CHECK: [[RES3:%.*]] = spirv.SpecConstantOperation wraps "spirv.IMul"([[RES1]], [[RES2]]) : (i32, i32) -> i32
    %3 = spirv.SpecConstantOperation wraps "spirv.IMul"(%2, %2) : (i32, i32) -> i32

    // Make sure deserialization continues from the right place after creating
    // the previous op.
    // CHECK: spirv.ReturnValue [[RES3]]
    spirv.ReturnValue %3 : i32
  }
}
