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
    %1 = spv.AccessChain %arg0[%arg1] : !spv.ptr<!spv.array<4x!spv.array<4xf32>>, Function>, i32
    %2 = spv.AccessChain %arg0[%arg1, %arg2] : !spv.ptr<!spv.array<4x!spv.array<4xf32>>, Function>, i32, i32
    spv.Return
  }
}

// -----

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  spv.func @load_store_zero_rank_float(%arg0: !spv.ptr<!spv.struct<(!spv.array<1 x f32, stride=4> [0])>, StorageBuffer>, %arg1: !spv.ptr<!spv.struct<(!spv.array<1 x f32, stride=4> [0])>, StorageBuffer>) "None" {
    // CHECK: [[LOAD_PTR:%.*]] = spv.AccessChain {{%.*}}[{{%.*}}, {{%.*}}] : !spv.ptr<!spv.struct<(!spv.array<1 x f32, stride=4> [0])>
    // CHECK-NEXT: [[VAL:%.*]] = spv.Load "StorageBuffer" [[LOAD_PTR]] : f32
    %0 = spv.Constant 0 : i32
    %1 = spv.AccessChain %arg0[%0, %0] : !spv.ptr<!spv.struct<(!spv.array<1 x f32, stride=4> [0])>, StorageBuffer>, i32, i32
    %2 = spv.Load "StorageBuffer" %1 : f32

    // CHECK: [[STORE_PTR:%.*]] = spv.AccessChain {{%.*}}[{{%.*}}, {{%.*}}] : !spv.ptr<!spv.struct<(!spv.array<1 x f32, stride=4> [0])>
    // CHECK-NEXT: spv.Store "StorageBuffer" [[STORE_PTR]], [[VAL]] : f32
    %3 = spv.Constant 0 : i32
    %4 = spv.AccessChain %arg1[%3, %3] : !spv.ptr<!spv.struct<(!spv.array<1 x f32, stride=4> [0])>, StorageBuffer>, i32, i32
    spv.Store "StorageBuffer" %4, %2 : f32
    spv.Return
  }

  spv.func @load_store_zero_rank_int(%arg0: !spv.ptr<!spv.struct<(!spv.array<1 x i32, stride=4> [0])>, StorageBuffer>, %arg1: !spv.ptr<!spv.struct<(!spv.array<1 x i32, stride=4> [0])>, StorageBuffer>) "None" {
    // CHECK: [[LOAD_PTR:%.*]] = spv.AccessChain {{%.*}}[{{%.*}}, {{%.*}}] : !spv.ptr<!spv.struct<(!spv.array<1 x i32, stride=4> [0])>
    // CHECK-NEXT: [[VAL:%.*]] = spv.Load "StorageBuffer" [[LOAD_PTR]] : i32
    %0 = spv.Constant 0 : i32
    %1 = spv.AccessChain %arg0[%0, %0] : !spv.ptr<!spv.struct<(!spv.array<1 x i32, stride=4> [0])>, StorageBuffer>, i32, i32
    %2 = spv.Load "StorageBuffer" %1 : i32

    // CHECK: [[STORE_PTR:%.*]] = spv.AccessChain {{%.*}}[{{%.*}}, {{%.*}}] : !spv.ptr<!spv.struct<(!spv.array<1 x i32, stride=4> [0])>
    // CHECK-NEXT: spv.Store "StorageBuffer" [[STORE_PTR]], [[VAL]] : i32
    %3 = spv.Constant 0 : i32
    %4 = spv.AccessChain %arg1[%3, %3] : !spv.ptr<!spv.struct<(!spv.array<1 x i32, stride=4> [0])>, StorageBuffer>, i32, i32
    spv.Store "StorageBuffer" %4, %2 : i32
    spv.Return
  }
}

// -----

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  spv.func @copy_memory_simple() "None" {
    %0 = spv.Variable : !spv.ptr<f32, Function>
    %1 = spv.Variable : !spv.ptr<f32, Function>
    // CHECK: spv.CopyMemory "Function" %{{.*}}, "Function" %{{.*}} : f32
    spv.CopyMemory "Function" %0, "Function" %1 : f32
    spv.Return
  }
}

// -----

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  spv.func @copy_memory_different_storage_classes(%in : !spv.ptr<!spv.array<4xf32>, Input>, %out : !spv.ptr<!spv.array<4xf32>, Output>) "None" {
    // CHECK: spv.CopyMemory "Output" %{{.*}}, "Input" %{{.*}} : !spv.array<4 x f32>
    spv.CopyMemory "Output" %out, "Input" %in : !spv.array<4xf32>
    spv.Return
  }
}


// -----

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  spv.func @copy_memory_with_access_operands() "None" {
    %0 = spv.Variable : !spv.ptr<f32, Function>
    %1 = spv.Variable : !spv.ptr<f32, Function>
    // CHECK: spv.CopyMemory "Function" %{{.*}}, "Function" %{{.*}} ["Aligned", 4] : f32
    spv.CopyMemory "Function" %0, "Function" %1 ["Aligned", 4] : f32

    // CHECK: spv.CopyMemory "Function" %{{.*}}, "Function" %{{.*}} ["Volatile"] : f32
    spv.CopyMemory "Function" %0, "Function" %1 ["Volatile"] : f32

    // CHECK: spv.CopyMemory "Function" %{{.*}}, "Function" %{{.*}} ["Volatile"], ["Volatile"] : f32
    spv.CopyMemory "Function" %0, "Function" %1 ["Volatile"], ["Volatile"] : f32

    // CHECK: spv.CopyMemory "Function" %{{.*}}, "Function" %{{.*}} ["Aligned", 4], ["Volatile"] : f32
    spv.CopyMemory "Function" %0, "Function" %1 ["Aligned", 4], ["Volatile"] : f32

    // CHECK: spv.CopyMemory "Function" %{{.*}}, "Function" %{{.*}} ["Volatile"], ["Aligned", 4] : f32
    spv.CopyMemory "Function" %0, "Function" %1 ["Volatile"], ["Aligned", 4] : f32

    // CHECK: spv.CopyMemory "Function" %{{.*}}, "Function" %{{.*}} ["Aligned", 8], ["Aligned", 4] : f32
    spv.CopyMemory "Function" %0, "Function" %1 ["Aligned", 8], ["Aligned", 4] : f32

    spv.Return
  }
}

