// RUN: mlir-opt --spirv-promote-to-replicated-constants --split-input-file %s | FileCheck %s

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, ReplicatedCompositesEXT], [SPV_EXT_replicated_composites]> {
  spirv.func @splat_vector_of_i32() -> (vector<3xi32>) "None" {
    // CHECK: {{%.*}} = spirv.EXT.ConstantCompositeReplicate [2 : i32] : vector<3xi32>
    %0 = spirv.Constant dense<2> : vector<3xi32>
    spirv.ReturnValue %0 : vector<3xi32>
  }

  spirv.func @splat_array_of_i32() -> (!spirv.array<3 x i32>) "None" {
    // CHECK: {{%.*}} = spirv.EXT.ConstantCompositeReplicate [1 : i32] : !spirv.array<3 x i32>
    %0 = spirv.Constant [1 : i32, 1 : i32, 1 : i32] : !spirv.array<3 x i32>
    spirv.ReturnValue %0 : !spirv.array<3 x i32>
  }

  spirv.func @splat_array_of_splat_array_of_i32() -> (!spirv.array<2 x !spirv.array<3 x i32>>) "None" {
    // CHECK: {{%.*}} = spirv.EXT.ConstantCompositeReplicate [3 : i32] : !spirv.array<2 x !spirv.array<3 x i32>>
    %0 = spirv.Constant [[3 : i32, 3 : i32, 3 : i32], [3 : i32, 3 : i32, 3 : i32]] : !spirv.array<2 x !spirv.array<3 x i32>>
    spirv.ReturnValue %0 : !spirv.array<2 x !spirv.array<3 x i32>>
  }

  spirv.func @splat_array_of_non_splat_array_of_i32() -> (!spirv.array<2 x !spirv.array<3 x i32>>) "None" {
    // CHECK: {{%.*}} = spirv.EXT.ConstantCompositeReplicate {{\[}}[1 : i32, 2 : i32, 3 : i32]] : !spirv.array<2 x !spirv.array<3 x i32>>
    %0 = spirv.Constant [[1 : i32, 2 : i32, 3 : i32], [1 : i32, 2 : i32, 3 : i32]] : !spirv.array<2 x !spirv.array<3 x i32>>
    spirv.ReturnValue %0 : !spirv.array<2 x !spirv.array<3 x i32>>
  }

  spirv.func @splat_array_of_vectors_of_i32() -> (!spirv.array<2xvector<2xi32>>) "None" {
    // CHECK: {{%.*}} = spirv.EXT.ConstantCompositeReplicate [dense<[1, 2]> : vector<2xi32>] : !spirv.array<2 x vector<2xi32>>
    %0 = spirv.Constant [dense<[1, 2]> : vector<2xi32>, dense<[1, 2]> : vector<2xi32>] : !spirv.array<2 x vector<2xi32>>
    spirv.ReturnValue %0 : !spirv.array<2 x vector<2xi32>>
  }

  spirv.func @splat_array_of_splat_vectors_of_i32() -> (!spirv.array<2 x vector<2xi32>>) "None" {
    // CHECK: {{%.*}} = spirv.EXT.ConstantCompositeReplicate [2 : i32] : !spirv.array<2 x vector<2xi32>>
    %0 = spirv.Constant [dense<2> : vector<2xi32>, dense<2> : vector<2xi32>] : !spirv.array<2 x vector<2xi32>>
    spirv.ReturnValue %0 : !spirv.array<2 x vector<2xi32>>
  }

  spirv.func @splat_tensor_of_i32() -> (!spirv.array<2 x !spirv.array<3 x i32>>) "None" {
    // CHECK: {{%.*}} = spirv.EXT.ConstantCompositeReplicate [3 : i32] : !spirv.array<2 x !spirv.array<3 x i32>>
    %0 = spirv.Constant dense<3> : tensor<2x3xi32> : !spirv.array<2 x !spirv.array<3 x i32>>
    spirv.ReturnValue %0 : !spirv.array<2 x !spirv.array<3 x i32>>
  }

  spirv.func @splat_arm_tensor_of_i32() -> (!spirv.arm.tensor<2x3xi32>) "None" {
    // CHECK: {{%.*}} = spirv.EXT.ConstantCompositeReplicate [2 : i32] : !spirv.arm.tensor<2x3xi32>
    %0 = spirv.Constant dense<2> : !spirv.arm.tensor<2x3xi32>
    spirv.ReturnValue %0 : !spirv.arm.tensor<2x3xi32>
  }

  spirv.func @array_of_splat_array_of_non_splat_vectors_of_i32() -> (!spirv.array<1 x !spirv.array<2 x vector<2xi32>>>) "None" {
    // CHECK: {{%.*}} = spirv.EXT.ConstantCompositeReplicate [dense<[1, 2]> : vector<2xi32>] : !spirv.array<1 x !spirv.array<2 x vector<2xi32>>
    %0 = spirv.Constant [[dense<[1, 2]> : vector<2xi32>, dense<[1, 2]> : vector<2xi32>]] : !spirv.array<1 x !spirv.array<2 x vector<2xi32>>>
    spirv.ReturnValue %0 : !spirv.array<1 x !spirv.array<2 x vector<2xi32>>>
  }

  spirv.func @array_of_one_splat_array_of_vector_of_one_i32() -> !spirv.array<1 x !spirv.array<2 x vector<1xi32>>> "None" {
    // CHECK: {{%.*}} = spirv.EXT.ConstantCompositeReplicate [dense<1> : vector<1xi32>] : !spirv.array<1 x !spirv.array<2 x vector<1xi32>
    %cst = spirv.Constant [[dense<1> : vector<1xi32>], [dense<1> : vector<1xi32>]] : !spirv.array<1 x !spirv.array<2 x vector<1xi32>>>
    spirv.ReturnValue %cst : !spirv.array<1 x !spirv.array<2 x vector<1xi32>>>
  }

  spirv.func @splat_array_of_array_of_one_vector_of_one_i32() -> (!spirv.array<2 x !spirv.array<1 x vector<1xi32>>>) "None" {
    // CHECK: {{%.*}} = spirv.EXT.ConstantCompositeReplicate [dense<1> : vector<1xi32>] : !spirv.array<2 x !spirv.array<1 x vector<1xi32>>>
    %0 = spirv.Constant [[dense<1> : vector<1xi32>], [dense<1> : vector<1xi32>]] : !spirv.array<2 x !spirv.array<1 x vector<1xi32>>>
    spirv.ReturnValue %0 : !spirv.array<2 x !spirv.array<1 x vector<1xi32>>>
  }

  spirv.func @array_of_one_array_of_one_splat_vector_of_i32() -> (!spirv.array<1 x !spirv.array<1 x vector<2xi32>>>) "None" {
    // CHECK: {{%.*}} = spirv.EXT.ConstantCompositeReplicate [1 : i32] : !spirv.array<1 x !spirv.array<1 x vector<2xi32>>>
    %0 = spirv.Constant [[dense<1> : vector<2xi32>]] : !spirv.array<1 x !spirv.array<1 x vector<2xi32>>>
    spirv.ReturnValue %0 : !spirv.array<1 x !spirv.array<1 x vector<2xi32>>>
  }

  spirv.func @splat_array_of_splat_array_of_non_splat_array_of_i32() -> (!spirv.array<2 x !spirv.array<2 x !spirv.array<3 x i32>>>) "None" {
    // CHECK: {{%.*}} = spirv.EXT.ConstantCompositeReplicate {{\[}}[1 : i32, 2 : i32, 3 : i32]] : !spirv.array<2 x !spirv.array<2 x !spirv.array<3 x i32>>>
    %0 = spirv.Constant [[[1 : i32, 2 : i32, 3 : i32], [1 : i32, 2 : i32, 3 : i32]], [[1 : i32, 2 : i32, 3 : i32], [1 : i32, 2 : i32, 3 : i32]]] : !spirv.array<2 x !spirv.array<2 x !spirv.array<3 x i32>>>
    spirv.ReturnValue %0 : !spirv.array<2 x !spirv.array<2 x !spirv.array<3 x i32>>>
  }

  spirv.func @splat_vector_of_f32() -> (vector<3xf32>) "None" {
    // CHECK: {{%.*}} = spirv.EXT.ConstantCompositeReplicate [2.000000e+00 : f32] : vector<3xf32>
    %0 = spirv.Constant dense<2.0> : vector<3xf32>
    spirv.ReturnValue %0 : vector<3xf32>
  }

  spirv.func @splat_array_of_f32() -> (!spirv.array<3 x f32>) "None" {
    // CHECK: {{%.*}} = spirv.EXT.ConstantCompositeReplicate [1.000000e+00 : f32] : !spirv.array<3 x f32>
    %0 = spirv.Constant [1.0 : f32, 1.0 : f32, 1.0 : f32] : !spirv.array<3 x f32>
    spirv.ReturnValue %0 : !spirv.array<3 x f32>
  }

  spirv.func @splat_array_of_splat_array_of_f32() -> (!spirv.array<2 x !spirv.array<3 x f32>>) "None" {
    // CHECK: {{%.*}} = spirv.EXT.ConstantCompositeReplicate [3.000000e+00 : f32] : !spirv.array<2 x !spirv.array<3 x f32>>
    %0 = spirv.Constant [[3.0 : f32, 3.0 : f32, 3.0 : f32], [3.0 : f32, 3.0 : f32, 3.0 : f32]] : !spirv.array<2 x !spirv.array<3 x f32>>
    spirv.ReturnValue %0 : !spirv.array<2 x !spirv.array<3 x f32>>
  }

  spirv.func @splat_array_of_non_splat_array_of_f32() -> (!spirv.array<2 x !spirv.array<3 x f32>>) "None" {
    // CHECK: {{%.*}} = spirv.EXT.ConstantCompositeReplicate {{\[}}[1.000000e+00 : f32, 2.000000e+00 : f32, 3.000000e+00 : f32]] : !spirv.array<2 x !spirv.array<3 x f32>>
    %0 = spirv.Constant [[1.0 : f32, 2.0 : f32, 3.0 : f32], [1.0 : f32, 2.0 : f32, 3.0 : f32]] : !spirv.array<2 x !spirv.array<3 x f32>>
    spirv.ReturnValue %0 : !spirv.array<2 x !spirv.array<3 x f32>>
  }

  spirv.func @splat_array_of_vectors_of_f32() -> (!spirv.array<2xvector<2xf32>>) "None" {
    // CHECK: {{%.*}} = spirv.EXT.ConstantCompositeReplicate [dense<[1.000000e+00, 2.000000e+00]> : vector<2xf32>] : !spirv.array<2 x vector<2xf32>>
    %0 = spirv.Constant [dense<[1.0, 2.0]> : vector<2xf32>, dense<[1.0, 2.0]> : vector<2xf32>] : !spirv.array<2 x vector<2xf32>>
    spirv.ReturnValue %0 : !spirv.array<2 x vector<2xf32>>
  }

  spirv.func @splat_array_of_splat_vectors_of_f32() -> (!spirv.array<2 x vector<2xf32>>) "None" {
    // CHECK: {{%.*}} = spirv.EXT.ConstantCompositeReplicate [2.000000e+00 : f32] : !spirv.array<2 x vector<2xf32>>
    %0 = spirv.Constant [dense<2.0> : vector<2xf32>, dense<2.0> : vector<2xf32>] : !spirv.array<2 x vector<2xf32>>
    spirv.ReturnValue %0 : !spirv.array<2 x vector<2xf32>>
  }

  spirv.func @splat_tensor_of_f32() -> (!spirv.array<2 x !spirv.array<3 x f32>>) "None" {
    // CHECK: {{%.*}} = spirv.EXT.ConstantCompositeReplicate [3.000000e+00 : f32] : !spirv.array<2 x !spirv.array<3 x f32>>
    %0 = spirv.Constant dense<3.0> : tensor<2x3xf32> : !spirv.array<2 x !spirv.array<3 x f32>>
    spirv.ReturnValue %0 : !spirv.array<2 x !spirv.array<3 x f32>>
  }

  spirv.func @splat_arm_tensor_of_f32() -> (!spirv.arm.tensor<2x3xf32>) "None" {
    // CHECK: {{%.*}} = spirv.EXT.ConstantCompositeReplicate [2.000000e+00 : f32] : !spirv.arm.tensor<2x3xf32>
    %0 = spirv.Constant dense<2.0> : !spirv.arm.tensor<2x3xf32>
    spirv.ReturnValue %0 : !spirv.arm.tensor<2x3xf32>
  }

  spirv.func @array_of_splat_array_of_non_splat_vectors_of_f32() -> (!spirv.array<1 x !spirv.array<2 x vector<2xf32>>>) "None" {
    // CHECK: {{%.*}} = spirv.EXT.ConstantCompositeReplicate [dense<[1.000000e+00, 2.000000e+00]> : vector<2xf32>] : !spirv.array<1 x !spirv.array<2 x vector<2xf32>>
    %0 = spirv.Constant [[dense<[1.0, 2.0]> : vector<2xf32>, dense<[1.0, 2.0]> : vector<2xf32>]] : !spirv.array<1 x !spirv.array<2 x vector<2xf32>>>
    spirv.ReturnValue %0 : !spirv.array<1 x !spirv.array<2 x vector<2xf32>>>
  }

  spirv.func @array_of_one_splat_array_of_vector_of_one_f32() -> !spirv.array<1 x !spirv.array<2 x vector<1xf32>>> "None" {
    // CHECK: {{%.*}} = spirv.EXT.ConstantCompositeReplicate [dense<1.000000e+00> : vector<1xf32>] : !spirv.array<1 x !spirv.array<2 x vector<1xf32>
    %cst = spirv.Constant [[dense<1.0> : vector<1xf32>], [dense<1.0> : vector<1xf32>]] : !spirv.array<1 x !spirv.array<2 x vector<1xf32>>>
    spirv.ReturnValue %cst : !spirv.array<1 x !spirv.array<2 x vector<1xf32>>>
  }

  spirv.func @splat_array_of_array_of_one_vector_of_one_f32() -> (!spirv.array<2 x !spirv.array<1 x vector<1xf32>>>) "None" {
    // CHECK: {{%.*}} = spirv.EXT.ConstantCompositeReplicate [dense<1.000000e+00> : vector<1xf32>] : !spirv.array<2 x !spirv.array<1 x vector<1xf32>>>
    %0 = spirv.Constant [[dense<1.0> : vector<1xf32>], [dense<1.0> : vector<1xf32>]] : !spirv.array<2 x !spirv.array<1 x vector<1xf32>>>
    spirv.ReturnValue %0 : !spirv.array<2 x !spirv.array<1 x vector<1xf32>>>
  }

  spirv.func @array_of_one_array_of_one_splat_vector_of_f32() -> (!spirv.array<1 x !spirv.array<1 x vector<2xf32>>>) "None" {
    // CHECK: {{%.*}} = spirv.EXT.ConstantCompositeReplicate [1.000000e+00 : f32] : !spirv.array<1 x !spirv.array<1 x vector<2xf32>>>
    %0 = spirv.Constant [[dense<1.0> : vector<2xf32>]] : !spirv.array<1 x !spirv.array<1 x vector<2xf32>>>
    spirv.ReturnValue %0 : !spirv.array<1 x !spirv.array<1 x vector<2xf32>>>
  }

  spirv.func @splat_array_of_splat_array_of_non_splat_array_of_f32() -> (!spirv.array<2 x !spirv.array<2 x !spirv.array<3 x f32>>>) "None" {
    // CHECK: {{%.*}} = spirv.EXT.ConstantCompositeReplicate {{\[}}[1.000000e+00 : f32, 2.000000e+00 : f32, 3.000000e+00 : f32]] : !spirv.array<2 x !spirv.array<2 x !spirv.array<3 x f32>>>
    %0 = spirv.Constant [[[1.0 : f32, 2.0 : f32, 3.0 : f32], [1.0 : f32, 2.0 : f32, 3.0 : f32]], [[1.0 : f32, 2.0 : f32, 3.0 : f32], [1.0 : f32, 2.0 : f32, 3.0 : f32]]] : !spirv.array<2 x !spirv.array<2 x !spirv.array<3 x f32>>>
    spirv.ReturnValue %0 : !spirv.array<2 x !spirv.array<2 x !spirv.array<3 x f32>>>
  }

  spirv.func @array_of_one_i32() -> (!spirv.array<1 x i32>) "None" {
    // CHECK-NOT: spirv.EXT.ConstantCompositeReplicate
    %0 = spirv.Constant [1 : i32] : !spirv.array<1 x i32>
    spirv.ReturnValue %0 : !spirv.array<1 x i32>
  }

  spirv.func @arm_tensor_of_one_i32() -> (!spirv.arm.tensor<1xi32>) "None" {
    // CHECK-NOT: spirv.EXT.ConstantCompositeReplicate
    %0 = spirv.Constant dense<1> : !spirv.arm.tensor<1xi32>
    spirv.ReturnValue %0 : !spirv.arm.tensor<1xi32>
  }

  spirv.func @non_splat_vector_of_i32() -> (vector<3xi32>) "None" {
    // CHECK-NOT: spirv.EXT.ConstantCompositeReplicate
    %0 = spirv.Constant dense<[0, 1, 2]> : vector<3xi32>
    spirv.ReturnValue %0 : vector<3xi32>
  }

  spirv.func @non_splat_array_of_vectors_of_i32() -> (!spirv.array<2xvector<2xi32>>) "None" {
    // CHECK-NOT: spirv.EXT.ConstantCompositeReplicate
    %0 = spirv.Constant [dense<[1, 2]> : vector<2xi32>, dense<[1, 3]> : vector<2xi32>] : !spirv.array<2 x vector<2xi32>>
    spirv.ReturnValue %0 : !spirv.array<2 x vector<2xi32>>
  }

  spirv.func @array_of_one_f32() -> (!spirv.array<1 x f32>) "None" {
    // CHECK-NOT: spirv.EXT.ConstantCompositeReplicate
    %0 = spirv.Constant [1.0 : f32] : !spirv.array<1 x f32>
    spirv.ReturnValue %0 : !spirv.array<1 x f32>
  }

  spirv.func @arm_tensor_of_one_f32() -> (!spirv.arm.tensor<1xf32>) "None" {
    // CHECK-NOT: spirv.EXT.ConstantCompositeReplicate
    %0 = spirv.Constant dense<1.0> : !spirv.arm.tensor<1xf32>
    spirv.ReturnValue %0 : !spirv.arm.tensor<1xf32>
  }

  spirv.func @non_splat_vector_of_f32() -> (vector<3xf32>) "None" {
    // CHECK-NOT: spirv.EXT.ConstantCompositeReplicate
    %0 = spirv.Constant dense<[0.0, 1.0, 2.0]> : vector<3xf32>
    spirv.ReturnValue %0 : vector<3xf32>
  }

  spirv.func @non_splat_array_of_vectors_of_f32() -> (!spirv.array<2xvector<2xf32>>) "None" {
    // CHECK-NOT: spirv.EXT.ConstantCompositeReplicate
    %0 = spirv.Constant [dense<[1.0, 2.0]> : vector<2xf32>, dense<[1.0, 3.0]> : vector<2xf32>] : !spirv.array<2 x vector<2xf32>>
    spirv.ReturnValue %0 : !spirv.array<2 x vector<2xf32>>
  }

  spirv.func @array_of_one_array_of_one_non_splat_vector_of_i32() -> (!spirv.array<1 x !spirv.array<1 x vector<2xi32>>>) "None" {
    // CHECK-NOT spirv.EXT.ConstantCompositeReplicate
    %0 = spirv.Constant [[dense<[1, 2]> : vector<2xi32>]] : !spirv.array<1 x !spirv.array<1 x vector<2xi32>>>
    spirv.ReturnValue %0 : !spirv.array<1 x !spirv.array<1 x vector<2xi32>>>
  }
  
  spirv.func @array_of_one_array_of_one_vector_of_one_i32() -> (!spirv.array<1 x !spirv.array<1 x vector<1xi32>>>) "None" {
    // CHECK-NOT spirv.EXT.ConstantCompositeReplicate
    %0 = spirv.Constant [[dense<1> : vector<1xi32>]] : !spirv.array<1 x !spirv.array<1 x vector<1xi32>>>
    spirv.ReturnValue %0 : !spirv.array<1 x !spirv.array<1 x vector<1xi32>>>
  }
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, ReplicatedCompositesEXT], [SPV_EXT_replicated_composites]> {

  spirv.SpecConstant @sc_i32_1 = 1 : i32

  // CHECK: spirv.EXT.SpecConstantCompositeReplicate @scc_splat_array_of_i32 (@sc_i32_1) : !spirv.array<3 x i32>
  spirv.SpecConstantComposite @scc_splat_array_of_i32 (@sc_i32_1, @sc_i32_1, @sc_i32_1) : !spirv.array<3 x i32>

  // CHECK: spirv.EXT.SpecConstantCompositeReplicate @scc_splat_struct_of_i32 (@sc_i32_1) : !spirv.struct<(i32, i32, i32)>
  spirv.SpecConstantComposite @scc_splat_struct_of_i32 (@sc_i32_1, @sc_i32_1, @sc_i32_1) : !spirv.struct<(i32, i32, i32)>

  // CHECK: spirv.EXT.SpecConstantCompositeReplicate @scc_splat_vector_of_i32 (@sc_i32_1) : vector<3xi32>
  spirv.SpecConstantComposite @scc_splat_vector_of_i32 (@sc_i32_1, @sc_i32_1, @sc_i32_1) : vector<3 x i32>

  // CHECK: spirv.EXT.SpecConstantCompositeReplicate @scc_splat_arm_tensor_of_i32 (@sc_i32_1) : !spirv.arm.tensor<3xi32>
  spirv.SpecConstantComposite @scc_splat_arm_tensor_of_i32 (@sc_i32_1, @sc_i32_1, @sc_i32_1) : !spirv.arm.tensor<3xi32>

  spirv.SpecConstant @sc_f32_1 = 1.0 : f32

  // CHECK: spirv.EXT.SpecConstantCompositeReplicate @scc_splat_array_of_f32 (@sc_f32_1) : !spirv.array<3 x f32>
  spirv.SpecConstantComposite @scc_splat_array_of_f32 (@sc_f32_1, @sc_f32_1, @sc_f32_1) : !spirv.array<3 x f32>

  // CHECK: spirv.EXT.SpecConstantCompositeReplicate @scc_splat_struct_of_f32 (@sc_f32_1) : !spirv.struct<(f32, f32, f32)>
  spirv.SpecConstantComposite @scc_splat_struct_of_f32 (@sc_f32_1, @sc_f32_1, @sc_f32_1) : !spirv.struct<(f32, f32, f32)>

  // CHECK: spirv.EXT.SpecConstantCompositeReplicate @scc_splat_vector_of_f32 (@sc_f32_1) : vector<3xf32>
  spirv.SpecConstantComposite @scc_splat_vector_of_f32 (@sc_f32_1, @sc_f32_1, @sc_f32_1) : vector<3 x f32>

  // CHECK: spirv.EXT.SpecConstantCompositeReplicate @scc_splat_arm_tensor_of_f32 (@sc_f32_1) : !spirv.arm.tensor<3xf32>
  spirv.SpecConstantComposite @scc_splat_arm_tensor_of_f32 (@sc_f32_1, @sc_f32_1, @sc_f32_1) : !spirv.arm.tensor<3xf32>

  spirv.SpecConstant @sc_i32_2 = 2 : i32

  // CHECK-NOT: spirv.EXT.SpecConstantCompositeReplicate
  spirv.SpecConstantComposite @scc_array_of_one_i32 (@sc_i32_1) : !spirv.array<1 x i32>

  // CHECK-NOT: spirv.EXT.SpecConstantCompositeReplicate
  spirv.SpecConstantComposite @scc_arm_tensor_of_one_i32 (@sc_i32_1) : !spirv.arm.tensor<1xi32>

  // CHECK-NOT: spirv.EXT.SpecConstantCompositeReplicate
  spirv.SpecConstantComposite @scc_non_splat_vector_of_i32 (@sc_i32_1, @sc_i32_1, @sc_i32_2) : vector<3 x i32>

  // CHECK-NOT: spirv.EXT.SpecConstantCompositeReplicate
  spirv.SpecConstantComposite @scc_non_splat_arm_tensor_of_i32 (@sc_i32_2, @sc_i32_1, @sc_i32_1) : !spirv.arm.tensor<3xi32>

  spirv.SpecConstant @sc_f32_2 = 2.0 : f32

  // CHECK-NOT: spirv.EXT.SpecConstantCompositeReplicate
  spirv.SpecConstantComposite @scc_array_of_one_f32 (@sc_f32_1) : !spirv.array<1 x f32>

  // CHECK-NOT: spirv.EXT.SpecConstantCompositeReplicate
  spirv.SpecConstantComposite @scc_arm_tensor_of_one_f32 (@sc_f32_1) : !spirv.arm.tensor<1xf32>

  // CHECK-NOT: spirv.EXT.SpecConstantCompositeReplicate
  spirv.SpecConstantComposite @scc_non_splat_vector_of_f32 (@sc_f32_1, @sc_f32_1, @sc_f32_2) : vector<3 x f32>

  // CHECK-NOT: spirv.EXT.SpecConstantCompositeReplicate
  spirv.SpecConstantComposite @scc_non_splat_arm_tensor_of_f32 (@sc_f32_2, @sc_f32_1, @sc_f32_1) : !spirv.arm.tensor<3xf32>

  // CHECK-NOT: spirv.EXT.SpecConstantCompositeReplicate
  spirv.SpecConstantComposite @scc_struct_of_i32_and_f32 (@sc_i32_1, @sc_i32_1, @sc_f32_1) : !spirv.struct<(i32, i32, f32)>
}
