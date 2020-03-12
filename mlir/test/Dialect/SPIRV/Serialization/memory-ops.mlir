// RUN: mlir-translate -test-spirv-roundtrip -split-input-file %s | FileCheck %s

// CHECK:           spv.func {{@.*}}([[ARG1:%.*]]: !spv.ptr<f32, Input>, [[ARG2:%.*]]: !spv.ptr<f32, Output>) "None" {
// CHECK-NEXT:        [[VALUE:%.*]] = spv.Load "Input" [[ARG1]] : f32
// CHECK-NEXT:        spv.Store "Output" [[ARG2]], [[VALUE]] : f32

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  spv.func @load_store(%arg0 : !spv.ptr<f32, Input>, %arg1 : !spv.ptr<f32, Output>) "None" {
    %1 = spv.Load "Input" %arg0 : f32
    spv.Store "Output" %arg1, %1 : f32
    spv.Return
  }
}

// -----

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  spv.func @access_chain(%arg0 : !spv.ptr<!spv.array<4x!spv.array<4xf32>>, Function>, %arg1 : i32, %arg2 : i32) "None" {
    // CHECK: {{%.*}} = spv.AccessChain {{%.*}}[{{%.*}}] : !spv.ptr<!spv.array<4 x !spv.array<4 x f32>>, Function>
    // CHECK-NEXT: {{%.*}} = spv.AccessChain {{%.*}}[{{%.*}}, {{%.*}}] : !spv.ptr<!spv.array<4 x !spv.array<4 x f32>>, Function>
    %1 = spv.AccessChain %arg0[%arg1] : !spv.ptr<!spv.array<4x!spv.array<4xf32>>, Function>
    %2 = spv.AccessChain %arg0[%arg1, %arg2] : !spv.ptr<!spv.array<4x!spv.array<4xf32>>, Function>
    spv.Return
  }
}

// -----

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  spv.func @load_store_zero_rank_float(%arg0: !spv.ptr<!spv.struct<!spv.array<1 x f32 [4]> [0]>, StorageBuffer>, %arg1: !spv.ptr<!spv.struct<!spv.array<1 x f32 [4]> [0]>, StorageBuffer>) "None" {
    // CHECK: [[LOAD_PTR:%.*]] = spv.AccessChain {{%.*}}[{{%.*}}, {{%.*}}] : !spv.ptr<!spv.struct<!spv.array<1 x f32 [4]> [0]>
    // CHECK-NEXT: [[VAL:%.*]] = spv.Load "StorageBuffer" [[LOAD_PTR]] : f32
    %0 = spv.constant 0 : i32
    %1 = spv.AccessChain %arg0[%0, %0] : !spv.ptr<!spv.struct<!spv.array<1 x f32 [4]> [0]>, StorageBuffer>
    %2 = spv.Load "StorageBuffer" %1 : f32

    // CHECK: [[STORE_PTR:%.*]] = spv.AccessChain {{%.*}}[{{%.*}}, {{%.*}}] : !spv.ptr<!spv.struct<!spv.array<1 x f32 [4]> [0]>
    // CHECK-NEXT: spv.Store "StorageBuffer" [[STORE_PTR]], [[VAL]] : f32
    %3 = spv.constant 0 : i32
    %4 = spv.AccessChain %arg1[%3, %3] : !spv.ptr<!spv.struct<!spv.array<1 x f32 [4]> [0]>, StorageBuffer>
    spv.Store "StorageBuffer" %4, %2 : f32
    spv.Return
  }

  spv.func @load_store_zero_rank_int(%arg0: !spv.ptr<!spv.struct<!spv.array<1 x i32 [4]> [0]>, StorageBuffer>, %arg1: !spv.ptr<!spv.struct<!spv.array<1 x i32 [4]> [0]>, StorageBuffer>) "None" {
    // CHECK: [[LOAD_PTR:%.*]] = spv.AccessChain {{%.*}}[{{%.*}}, {{%.*}}] : !spv.ptr<!spv.struct<!spv.array<1 x i32 [4]> [0]>
    // CHECK-NEXT: [[VAL:%.*]] = spv.Load "StorageBuffer" [[LOAD_PTR]] : i32
    %0 = spv.constant 0 : i32
    %1 = spv.AccessChain %arg0[%0, %0] : !spv.ptr<!spv.struct<!spv.array<1 x i32 [4]> [0]>, StorageBuffer>
    %2 = spv.Load "StorageBuffer" %1 : i32

    // CHECK: [[STORE_PTR:%.*]] = spv.AccessChain {{%.*}}[{{%.*}}, {{%.*}}] : !spv.ptr<!spv.struct<!spv.array<1 x i32 [4]> [0]>
    // CHECK-NEXT: spv.Store "StorageBuffer" [[STORE_PTR]], [[VAL]] : i32
    %3 = spv.constant 0 : i32
    %4 = spv.AccessChain %arg1[%3, %3] : !spv.ptr<!spv.struct<!spv.array<1 x i32 [4]> [0]>, StorageBuffer>
    spv.Store "StorageBuffer" %4, %2 : i32
    spv.Return
  }
}
