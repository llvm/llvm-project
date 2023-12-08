// RUN: mlir-translate -no-implicit-module -test-spirv-roundtrip -split-input-file %s | FileCheck %s


spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  // CHECK-LABEL: spirv.func @load_store
  //  CHECK-SAME: ([[ARG1:%.*]]: !spirv.ptr<f32, Input>, [[ARG2:%.*]]: !spirv.ptr<f32, Output>)
  spirv.func @load_store(%arg0 : !spirv.ptr<f32, Input>, %arg1 : !spirv.ptr<f32, Output>) "None" {
    // CHECK-NEXT: [[VALUE:%.*]] = spirv.Load "Input" [[ARG1]] : f32
    %1 = spirv.Load "Input" %arg0 : f32
    // CHECK-NEXT: spirv.Store "Output" [[ARG2]], [[VALUE]] : f32
    spirv.Store "Output" %arg1, %1 : f32
    spirv.Return
  }

  // CHECK-LABEL: spirv.func @load_store_memory_operands
  spirv.func @load_store_memory_operands(%arg0 : !spirv.ptr<f32, Input>, %arg1 : !spirv.ptr<f32, Output>) "None" {
    // CHECK: spirv.Load "Input" %{{.+}} ["Volatile|Aligned", 4] : f32
    %1 = spirv.Load "Input" %arg0 ["Volatile|Aligned", 4]: f32
    // CHECK: spirv.Store "Output" %{{.+}}, %{{.+}} ["Volatile|Aligned", 4] : f32
    spirv.Store "Output" %arg1, %1 ["Volatile|Aligned", 4]: f32
    spirv.Return
  }
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  spirv.func @access_chain(%arg0 : !spirv.ptr<!spirv.array<4x!spirv.array<4xf32>>, Function>, %arg1 : i32, %arg2 : i32) "None" {
    // CHECK: {{%.*}} = spirv.AccessChain {{%.*}}[{{%.*}}] : !spirv.ptr<!spirv.array<4 x !spirv.array<4 x f32>>, Function>
    // CHECK-NEXT: {{%.*}} = spirv.AccessChain {{%.*}}[{{%.*}}, {{%.*}}] : !spirv.ptr<!spirv.array<4 x !spirv.array<4 x f32>>, Function>
    %1 = spirv.AccessChain %arg0[%arg1] : !spirv.ptr<!spirv.array<4x!spirv.array<4xf32>>, Function>, i32
    %2 = spirv.AccessChain %arg0[%arg1, %arg2] : !spirv.ptr<!spirv.array<4x!spirv.array<4xf32>>, Function>, i32, i32
    spirv.Return
  }
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  spirv.func @load_store_zero_rank_float(%arg0: !spirv.ptr<!spirv.struct<(!spirv.array<1 x f32, stride=4> [0])>, StorageBuffer>, %arg1: !spirv.ptr<!spirv.struct<(!spirv.array<1 x f32, stride=4> [0])>, StorageBuffer>) "None" {
    // CHECK: [[LOAD_PTR:%.*]] = spirv.AccessChain {{%.*}}[{{%.*}}, {{%.*}}] : !spirv.ptr<!spirv.struct<(!spirv.array<1 x f32, stride=4> [0])>
    // CHECK-NEXT: [[VAL:%.*]] = spirv.Load "StorageBuffer" [[LOAD_PTR]] : f32
    %0 = spirv.Constant 0 : i32
    %1 = spirv.AccessChain %arg0[%0, %0] : !spirv.ptr<!spirv.struct<(!spirv.array<1 x f32, stride=4> [0])>, StorageBuffer>, i32, i32
    %2 = spirv.Load "StorageBuffer" %1 : f32

    // CHECK: [[STORE_PTR:%.*]] = spirv.AccessChain {{%.*}}[{{%.*}}, {{%.*}}] : !spirv.ptr<!spirv.struct<(!spirv.array<1 x f32, stride=4> [0])>
    // CHECK-NEXT: spirv.Store "StorageBuffer" [[STORE_PTR]], [[VAL]] : f32
    %3 = spirv.Constant 0 : i32
    %4 = spirv.AccessChain %arg1[%3, %3] : !spirv.ptr<!spirv.struct<(!spirv.array<1 x f32, stride=4> [0])>, StorageBuffer>, i32, i32
    spirv.Store "StorageBuffer" %4, %2 : f32
    spirv.Return
  }

  spirv.func @load_store_zero_rank_int(%arg0: !spirv.ptr<!spirv.struct<(!spirv.array<1 x i32, stride=4> [0])>, StorageBuffer>, %arg1: !spirv.ptr<!spirv.struct<(!spirv.array<1 x i32, stride=4> [0])>, StorageBuffer>) "None" {
    // CHECK: [[LOAD_PTR:%.*]] = spirv.AccessChain {{%.*}}[{{%.*}}, {{%.*}}] : !spirv.ptr<!spirv.struct<(!spirv.array<1 x i32, stride=4> [0])>
    // CHECK-NEXT: [[VAL:%.*]] = spirv.Load "StorageBuffer" [[LOAD_PTR]] : i32
    %0 = spirv.Constant 0 : i32
    %1 = spirv.AccessChain %arg0[%0, %0] : !spirv.ptr<!spirv.struct<(!spirv.array<1 x i32, stride=4> [0])>, StorageBuffer>, i32, i32
    %2 = spirv.Load "StorageBuffer" %1 : i32

    // CHECK: [[STORE_PTR:%.*]] = spirv.AccessChain {{%.*}}[{{%.*}}, {{%.*}}] : !spirv.ptr<!spirv.struct<(!spirv.array<1 x i32, stride=4> [0])>
    // CHECK-NEXT: spirv.Store "StorageBuffer" [[STORE_PTR]], [[VAL]] : i32
    %3 = spirv.Constant 0 : i32
    %4 = spirv.AccessChain %arg1[%3, %3] : !spirv.ptr<!spirv.struct<(!spirv.array<1 x i32, stride=4> [0])>, StorageBuffer>, i32, i32
    spirv.Store "StorageBuffer" %4, %2 : i32
    spirv.Return
  }
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  spirv.func @copy_memory_simple() "None" {
    %0 = spirv.Variable : !spirv.ptr<f32, Function>
    %1 = spirv.Variable : !spirv.ptr<f32, Function>
    // CHECK: spirv.CopyMemory "Function" %{{.*}}, "Function" %{{.*}} : f32
    spirv.CopyMemory "Function" %0, "Function" %1 : f32
    spirv.Return
  }
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  spirv.func @copy_memory_different_storage_classes(%in : !spirv.ptr<!spirv.array<4xf32>, Input>, %out : !spirv.ptr<!spirv.array<4xf32>, Output>) "None" {
    // CHECK: spirv.CopyMemory "Output" %{{.*}}, "Input" %{{.*}} : !spirv.array<4 x f32>
    spirv.CopyMemory "Output" %out, "Input" %in : !spirv.array<4xf32>
    spirv.Return
  }
}


// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  spirv.func @copy_memory_with_access_operands() "None" {
    %0 = spirv.Variable : !spirv.ptr<f32, Function>
    %1 = spirv.Variable : !spirv.ptr<f32, Function>
    // CHECK: spirv.CopyMemory "Function" %{{.*}}, "Function" %{{.*}} ["Aligned", 4] : f32
    spirv.CopyMemory "Function" %0, "Function" %1 ["Aligned", 4] : f32

    // CHECK: spirv.CopyMemory "Function" %{{.*}}, "Function" %{{.*}} ["Volatile"] : f32
    spirv.CopyMemory "Function" %0, "Function" %1 ["Volatile"] : f32

    // CHECK: spirv.CopyMemory "Function" %{{.*}}, "Function" %{{.*}} ["Volatile"], ["Volatile"] : f32
    spirv.CopyMemory "Function" %0, "Function" %1 ["Volatile"], ["Volatile"] : f32

    // CHECK: spirv.CopyMemory "Function" %{{.*}}, "Function" %{{.*}} ["Aligned", 4], ["Volatile"] : f32
    spirv.CopyMemory "Function" %0, "Function" %1 ["Aligned", 4], ["Volatile"] : f32

    // CHECK: spirv.CopyMemory "Function" %{{.*}}, "Function" %{{.*}} ["Volatile"], ["Aligned", 4] : f32
    spirv.CopyMemory "Function" %0, "Function" %1 ["Volatile"], ["Aligned", 4] : f32

    // CHECK: spirv.CopyMemory "Function" %{{.*}}, "Function" %{{.*}} ["Aligned", 8], ["Aligned", 4] : f32
    spirv.CopyMemory "Function" %0, "Function" %1 ["Aligned", 8], ["Aligned", 4] : f32

    spirv.Return
  }
}

