// RUN: mlir-translate -test-spirv-roundtrip -split-input-file %s | FileCheck %s

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  // CHECK: spv.SpecConstant @sc_true = true
  spv.SpecConstant @sc_true = true
  // CHECK: spv.SpecConstant @sc_false spec_id(1) = false
  spv.SpecConstant @sc_false spec_id(1) = false

  // CHECK: spv.SpecConstant @sc_int = -5 : i32
  spv.SpecConstant @sc_int = -5 : i32

  // CHECK: spv.SpecConstant @sc_float spec_id(5) = 1.000000e+00 : f32
  spv.SpecConstant @sc_float spec_id(5) = 1. : f32

  // CHECK: spv.SpecConstantComposite @scc (@sc_int, @sc_int) : !spv.array<2 x i32>
  spv.SpecConstantComposite @scc (@sc_int, @sc_int) : !spv.array<2 x i32>

  // CHECK-LABEL: @use
  spv.func @use() -> (i32) "None" {
    // We materialize a `spv.mlir.referenceof` op at every use of a
    // specialization constant in the deserializer. So two ops here.
    // CHECK: %[[USE1:.*]] = spv.mlir.referenceof @sc_int : i32
    // CHECK: %[[USE2:.*]] = spv.mlir.referenceof @sc_int : i32
    // CHECK: spv.IAdd %[[USE1]], %[[USE2]]

    %0 = spv.mlir.referenceof @sc_int : i32
    %1 = spv.IAdd %0, %0 : i32
    spv.ReturnValue %1 : i32
  }

  // CHECK-LABEL: @use
  spv.func @use_composite() -> (i32) "None" {
    // We materialize a `spv.mlir.referenceof` op at every use of a
    // specialization constant in the deserializer. So two ops here.
    // CHECK: %[[USE1:.*]] = spv.mlir.referenceof @scc : !spv.array<2 x i32>
    // CHECK: %[[ITM0:.*]] = spv.CompositeExtract %[[USE1]][0 : i32] : !spv.array<2 x i32>
    // CHECK: %[[USE2:.*]] = spv.mlir.referenceof @scc : !spv.array<2 x i32>
    // CHECK: %[[ITM1:.*]] = spv.CompositeExtract %[[USE2]][1 : i32] : !spv.array<2 x i32>
    // CHECK: spv.IAdd %[[ITM0]], %[[ITM1]]

    %0 = spv.mlir.referenceof @scc : !spv.array<2 x i32>
    %1 = spv.CompositeExtract %0[0 : i32] : !spv.array<2 x i32>
    %2 = spv.CompositeExtract %0[1 : i32] : !spv.array<2 x i32>
    %3 = spv.IAdd %1, %2 : i32
    spv.ReturnValue %3 : i32
  }
}

// -----

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {

  spv.SpecConstant @sc_f32_1 = 1.5 : f32
  spv.SpecConstant @sc_f32_2 = 2.5 : f32
  spv.SpecConstant @sc_f32_3 = 3.5 : f32

  spv.SpecConstant @sc_i32_1 = 1   : i32

  // CHECK: spv.SpecConstantComposite @scc_array (@sc_f32_1, @sc_f32_2, @sc_f32_3) : !spv.array<3 x f32>
  spv.SpecConstantComposite @scc_array (@sc_f32_1, @sc_f32_2, @sc_f32_3) : !spv.array<3 x f32>

  // CHECK: spv.SpecConstantComposite @scc_struct (@sc_i32_1, @sc_f32_2, @sc_f32_3) : !spv.struct<(i32, f32, f32)>
  spv.SpecConstantComposite @scc_struct (@sc_i32_1, @sc_f32_2, @sc_f32_3) : !spv.struct<(i32, f32, f32)>

  // CHECK: spv.SpecConstantComposite @scc_vector (@sc_f32_1, @sc_f32_2, @sc_f32_3) : vector<3xf32>
  spv.SpecConstantComposite @scc_vector (@sc_f32_1, @sc_f32_2, @sc_f32_3) : vector<3 x f32>
}

// -----

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {

  spv.SpecConstant @sc_f32_1 = 1.5 : f32
  spv.SpecConstant @sc_f32_2 = 2.5 : f32
  spv.SpecConstant @sc_f32_3 = 3.5 : f32

  spv.SpecConstant @sc_i32_1 = 1   : i32

  // CHECK: spv.SpecConstantComposite @scc_array (@sc_f32_1, @sc_f32_2, @sc_f32_3) : !spv.array<3 x f32>
  spv.SpecConstantComposite @scc_array (@sc_f32_1, @sc_f32_2, @sc_f32_3) : !spv.array<3 x f32>

  // CHECK: spv.SpecConstantComposite @scc_struct (@sc_i32_1, @sc_f32_2, @sc_f32_3) : !spv.struct<(i32, f32, f32)>
  spv.SpecConstantComposite @scc_struct (@sc_i32_1, @sc_f32_2, @sc_f32_3) : !spv.struct<(i32, f32, f32)>

  // CHECK: spv.SpecConstantComposite @scc_vector (@sc_f32_1, @sc_f32_2, @sc_f32_3) : vector<3xf32>
  spv.SpecConstantComposite @scc_vector (@sc_f32_1, @sc_f32_2, @sc_f32_3) : vector<3 x f32>
}

// -----

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {

  spv.SpecConstant @sc_i32_1 = 1 : i32

  spv.func @use_composite() -> (i32) "None" {
    // CHECK: [[USE1:%.*]] = spv.mlir.referenceof @sc_i32_1 : i32
    // CHECK: [[USE2:%.*]] = spv.constant 0 : i32

    // CHECK: [[RES1:%.*]] = spv.SpecConstantOperation wraps "spv.ISub"([[USE1]], [[USE2]]) : (i32, i32) -> i32

    // CHECK: [[USE3:%.*]] = spv.mlir.referenceof @sc_i32_1 : i32
    // CHECK: [[USE4:%.*]] = spv.constant 0 : i32

    // CHECK: [[RES2:%.*]] = spv.SpecConstantOperation wraps "spv.ISub"([[USE3]], [[USE4]]) : (i32, i32) -> i32

    %0 = spv.mlir.referenceof @sc_i32_1 : i32
    %1 = spv.constant 0 : i32
    %2 = spv.SpecConstantOperation wraps "spv.ISub"(%0, %1) : (i32, i32) -> i32

    // CHECK: [[RES3:%.*]] = spv.SpecConstantOperation wraps "spv.IMul"([[RES1]], [[RES2]]) : (i32, i32) -> i32
    %3 = spv.SpecConstantOperation wraps "spv.IMul"(%2, %2) : (i32, i32) -> i32

    // Make sure deserialization continues from the right place after creating
    // the previous op.
    // CHECK: spv.ReturnValue [[RES3]]
    spv.ReturnValue %3 : i32
  }
}
