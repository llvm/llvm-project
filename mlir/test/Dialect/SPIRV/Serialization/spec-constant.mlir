// RUN: mlir-translate -test-spirv-roundtrip -split-input-file %s | FileCheck %s

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  // CHECK: spv.specConstant @sc_true = true
  spv.specConstant @sc_true = true
  // CHECK: spv.specConstant @sc_false spec_id(1) = false
  spv.specConstant @sc_false spec_id(1) = false

  // CHECK: spv.specConstant @sc_int = -5 : i32
  spv.specConstant @sc_int = -5 : i32

  // CHECK: spv.specConstant @sc_float spec_id(5) = 1.000000e+00 : f32
  spv.specConstant @sc_float spec_id(5) = 1. : f32

  // CHECK: spv.specConstantComposite @scc (@sc_int, @sc_int) : !spv.array<2 x i32>
  spv.specConstantComposite @scc (@sc_int, @sc_int) : !spv.array<2 x i32>

  // CHECK-LABEL: @use
  spv.func @use() -> (i32) "None" {
    // We materialize a `spv._reference_of` op at every use of a
    // specialization constant in the deserializer. So two ops here.
    // CHECK: %[[USE1:.*]] = spv._reference_of @sc_int : i32
    // CHECK: %[[USE2:.*]] = spv._reference_of @sc_int : i32
    // CHECK: spv.IAdd %[[USE1]], %[[USE2]]

    %0 = spv._reference_of @sc_int : i32
    %1 = spv.IAdd %0, %0 : i32
    spv.ReturnValue %1 : i32
  }

  // CHECK-LABEL: @use
  spv.func @use_composite() -> (i32) "None" {
    // We materialize a `spv._reference_of` op at every use of a
    // specialization constant in the deserializer. So two ops here.
    // CHECK: %[[USE1:.*]] = spv._reference_of @scc : !spv.array<2 x i32>
    // CHECK: %[[ITM0:.*]] = spv.CompositeExtract %[[USE1]][0 : i32] : !spv.array<2 x i32>
    // CHECK: %[[USE2:.*]] = spv._reference_of @scc : !spv.array<2 x i32>
    // CHECK: %[[ITM1:.*]] = spv.CompositeExtract %[[USE2]][1 : i32] : !spv.array<2 x i32>
    // CHECK: spv.IAdd %[[ITM0]], %[[ITM1]]

    %0 = spv._reference_of @scc : !spv.array<2 x i32>
    %1 = spv.CompositeExtract %0[0 : i32] : !spv.array<2 x i32>
    %2 = spv.CompositeExtract %0[1 : i32] : !spv.array<2 x i32>
    %3 = spv.IAdd %1, %2 : i32
    spv.ReturnValue %3 : i32
  }
}

// -----

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {

  spv.specConstant @sc_f32_1 = 1.5 : f32
  spv.specConstant @sc_f32_2 = 2.5 : f32
  spv.specConstant @sc_f32_3 = 3.5 : f32

  spv.specConstant @sc_i32_1 = 1   : i32

  // CHECK: spv.specConstantComposite @scc_array (@sc_f32_1, @sc_f32_2, @sc_f32_3) : !spv.array<3 x f32>
  spv.specConstantComposite @scc_array (@sc_f32_1, @sc_f32_2, @sc_f32_3) : !spv.array<3 x f32>

  // CHECK: spv.specConstantComposite @scc_struct (@sc_i32_1, @sc_f32_2, @sc_f32_3) : !spv.struct<(i32, f32, f32)>
  spv.specConstantComposite @scc_struct (@sc_i32_1, @sc_f32_2, @sc_f32_3) : !spv.struct<(i32, f32, f32)>

  // CHECK: spv.specConstantComposite @scc_vector (@sc_f32_1, @sc_f32_2, @sc_f32_3) : vector<3xf32>
  spv.specConstantComposite @scc_vector (@sc_f32_1, @sc_f32_2, @sc_f32_3) : vector<3 x f32>
}

// -----

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {

  spv.specConstant @sc_f32_1 = 1.5 : f32
  spv.specConstant @sc_f32_2 = 2.5 : f32
  spv.specConstant @sc_f32_3 = 3.5 : f32

  spv.specConstant @sc_i32_1 = 1   : i32

  // CHECK: spv.specConstantComposite @scc_array (@sc_f32_1, @sc_f32_2, @sc_f32_3) : !spv.array<3 x f32>
  spv.specConstantComposite @scc_array (@sc_f32_1, @sc_f32_2, @sc_f32_3) : !spv.array<3 x f32>

  // CHECK: spv.specConstantComposite @scc_struct (@sc_i32_1, @sc_f32_2, @sc_f32_3) : !spv.struct<(i32, f32, f32)>
  spv.specConstantComposite @scc_struct (@sc_i32_1, @sc_f32_2, @sc_f32_3) : !spv.struct<(i32, f32, f32)>

  // CHECK: spv.specConstantComposite @scc_vector (@sc_f32_1, @sc_f32_2, @sc_f32_3) : vector<3xf32>
  spv.specConstantComposite @scc_vector (@sc_f32_1, @sc_f32_2, @sc_f32_3) : vector<3 x f32>
}
