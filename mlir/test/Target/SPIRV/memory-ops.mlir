// RUN: mlir-translate -no-implicit-module -test-spirv-roundtrip -split-input-file %s | FileCheck %s

// RUN: %if spirv-tools %{ rm -rf %t %}
// RUN: %if spirv-tools %{ mkdir %t %}
// RUN: %if spirv-tools %{ mlir-translate --no-implicit-module --serialize-spirv --split-input-file --spirv-save-validation-files-with-prefix=%t/module %s %}
// RUN: %if spirv-tools %{ spirv-val %t %}

// Alignment decorations on function-local variables survive round-tripping,
// including unsigned literals whose high bit is set.
spirv.module Physical64 OpenCL requires #spirv.vce<v1.0, [Addresses, Kernel, Linkage], []> {
  // CHECK-LABEL: spirv.func @aligned_variables()
  spirv.func @aligned_variables() "None" {
    // CHECK-NEXT: %{{.*}} = spirv.Variable {alignment = 16 : i32} : !spirv.ptr<f32, Function>
    // CHECK-NEXT: %{{.*}} = spirv.Variable {alignment = -2147483648 : i32} : !spirv.ptr<f32, Function>
    %aligned = spirv.Variable {alignment = 16 : i32} : !spirv.ptr<f32, Function>
    %high_bit = spirv.Variable {alignment = 2147483648 : i32} : !spirv.ptr<f32, Function>
    spirv.Return
  }
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage], []> {
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
    // CHECK: spirv.Load "Input" %{{.+}} ["Volatile|Aligned", 4] {relaxed_precision} : f32
    %1 = spirv.Load "Input" %arg0 ["Volatile|Aligned", 4] {relaxed_precision} : f32
    // CHECK: spirv.Store "Output" %{{.+}}, %{{.+}} ["Volatile|Aligned", 4] : f32
    spirv.Store "Output" %arg1, %1 ["Volatile|Aligned", 4]: f32
    spirv.Return
  }
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage], []> {
  spirv.func @access_chain(%arg0 : !spirv.ptr<!spirv.array<4x!spirv.array<4xf32>>, Function>, %arg1 : i32, %arg2 : i32) "None" {
    // CHECK: {{%.*}} = spirv.AccessChain {{%.*}}[{{%.*}}] : !spirv.ptr<!spirv.array<4 x !spirv.array<4 x f32>>, Function>
    // CHECK-NEXT: {{%.*}} = spirv.AccessChain {{%.*}}[{{%.*}}, {{%.*}}] : !spirv.ptr<!spirv.array<4 x !spirv.array<4 x f32>>, Function>
    %1 = spirv.AccessChain %arg0[%arg1] : !spirv.ptr<!spirv.array<4x!spirv.array<4xf32>>, Function>, i32 -> !spirv.ptr<!spirv.array<4xf32>, Function>
    %2 = spirv.AccessChain %arg0[%arg1, %arg2] : !spirv.ptr<!spirv.array<4x!spirv.array<4xf32>>, Function>, i32, i32 -> !spirv.ptr<f32, Function>
    spirv.Return
  }
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage], []> {
  spirv.func @inbounds_access_chain(%arg0 : !spirv.ptr<!spirv.array<4xf32>, Function>, %arg1 : i32) "None" {
    // CHECK: {{%.*}} = spirv.InBoundsAccessChain {{%.*}}[{{%.*}}] : !spirv.ptr<!spirv.array<4 x f32>, Function>
    %0 = spirv.InBoundsAccessChain %arg0[%arg1] : !spirv.ptr<!spirv.array<4xf32>, Function>, i32 -> !spirv.ptr<f32, Function>
    %1 = spirv.Load "Function" %0 : f32
    spirv.Return
  }
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage], [SPV_KHR_storage_buffer_storage_class]> {
  spirv.func @load_store_zero_rank_float(%arg0: !spirv.ptr<!spirv.struct<(!spirv.array<1 x f32, stride=4> [0]), Block>, StorageBuffer>, %arg1: !spirv.ptr<!spirv.struct<(!spirv.array<1 x f32, stride=4> [0]), Block>, StorageBuffer>) "None" {
    // CHECK: [[LOAD_PTR:%.*]] = spirv.AccessChain {{%.*}}[{{%.*}}, {{%.*}}] : !spirv.ptr<!spirv.struct<(!spirv.array<1 x f32, stride=4> [0]), Block>, StorageBuffer>
    // CHECK-NEXT: [[VAL:%.*]] = spirv.Load "StorageBuffer" [[LOAD_PTR]] : f32
    %0 = spirv.Constant 0 : i32
    %1 = spirv.AccessChain %arg0[%0, %0] : !spirv.ptr<!spirv.struct<(!spirv.array<1 x f32, stride=4> [0]), Block>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
    %2 = spirv.Load "StorageBuffer" %1 : f32

    // CHECK: [[STORE_PTR:%.*]] = spirv.AccessChain {{%.*}}[{{%.*}}, {{%.*}}] : !spirv.ptr<!spirv.struct<(!spirv.array<1 x f32, stride=4> [0]), Block>, StorageBuffer>
    // CHECK-NEXT: spirv.Store "StorageBuffer" [[STORE_PTR]], [[VAL]] : f32
    %3 = spirv.Constant 0 : i32
    %4 = spirv.AccessChain %arg1[%3, %3] : !spirv.ptr<!spirv.struct<(!spirv.array<1 x f32, stride=4> [0]), Block>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
    spirv.Store "StorageBuffer" %4, %2 : f32
    spirv.Return
  }

  spirv.func @load_store_zero_rank_int(%arg0: !spirv.ptr<!spirv.struct<(!spirv.array<1 x i32, stride=4> [0]), Block>, StorageBuffer>, %arg1: !spirv.ptr<!spirv.struct<(!spirv.array<1 x i32, stride=4> [0]), Block>, StorageBuffer>) "None" {
    // CHECK: [[LOAD_PTR:%.*]] = spirv.AccessChain {{%.*}}[{{%.*}}, {{%.*}}] : !spirv.ptr<!spirv.struct<(!spirv.array<1 x i32, stride=4> [0]), Block>, StorageBuffer>
    // CHECK-NEXT: [[VAL:%.*]] = spirv.Load "StorageBuffer" [[LOAD_PTR]] : i32
    %0 = spirv.Constant 0 : i32
    %1 = spirv.AccessChain %arg0[%0, %0] : !spirv.ptr<!spirv.struct<(!spirv.array<1 x i32, stride=4> [0]), Block>, StorageBuffer>, i32, i32 -> !spirv.ptr<i32, StorageBuffer>
    %2 = spirv.Load "StorageBuffer" %1 : i32

    // CHECK: [[STORE_PTR:%.*]] = spirv.AccessChain {{%.*}}[{{%.*}}, {{%.*}}] : !spirv.ptr<!spirv.struct<(!spirv.array<1 x i32, stride=4> [0]), Block>, StorageBuffer>
    // CHECK-NEXT: spirv.Store "StorageBuffer" [[STORE_PTR]], [[VAL]] : i32
    %3 = spirv.Constant 0 : i32
    %4 = spirv.AccessChain %arg1[%3, %3] : !spirv.ptr<!spirv.struct<(!spirv.array<1 x i32, stride=4> [0]), Block>, StorageBuffer>, i32, i32 -> !spirv.ptr<i32, StorageBuffer>
    spirv.Store "StorageBuffer" %4, %2 : i32
    spirv.Return
  }
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage], []> {
  spirv.func @copy_memory_simple() "None" {
    %0 = spirv.Variable : !spirv.ptr<f32, Function>
    %1 = spirv.Variable : !spirv.ptr<f32, Function>
    // CHECK: spirv.CopyMemory "Function" %{{.*}}, "Function" %{{.*}} : f32
    spirv.CopyMemory "Function" %0, "Function" %1 : f32
    spirv.Return
  }
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage], []> {
  spirv.func @copy_memory_different_storage_classes(%in : !spirv.ptr<!spirv.array<4xf32>, Input>, %out : !spirv.ptr<!spirv.array<4xf32>, Output>) "None" {
    // CHECK: spirv.CopyMemory "Output" %{{.*}}, "Input" %{{.*}} : !spirv.array<4 x f32>
    spirv.CopyMemory "Output" %out, "Input" %in : !spirv.array<4xf32>
    spirv.Return
  }
}


// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.4, [Shader, Linkage], []> {
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

// -----

spirv.module Physical64 OpenCL requires #spirv.vce<v1.0, [Kernel, Addresses, Int64, Linkage], []> {
  spirv.func @copy_memory_sized(%size: i64) "None" {
    %0 = spirv.Variable : !spirv.ptr<f32, Function>
    %1 = spirv.Variable : !spirv.ptr<i32, Function>
    %2 = spirv.Constant 4 : i32

    // CHECK: spirv.CopyMemorySized %{{.*}}, %{{.*}}, %{{.*}} : !spirv.ptr<f32, Function>, !spirv.ptr<i32, Function>, i32
    spirv.CopyMemorySized %0, %1, %2 : !spirv.ptr<f32, Function>, !spirv.ptr<i32, Function>, i32

    // CHECK: spirv.CopyMemorySized %{{.*}}, %{{.*}}, %{{.*}} : !spirv.ptr<f32, Function>, !spirv.ptr<i32, Function>, i64
    spirv.CopyMemorySized %0, %1, %size : !spirv.ptr<f32, Function>, !spirv.ptr<i32, Function>, i64

    // CHECK: spirv.CopyMemorySized %{{.*}}, %{{.*}}, %{{.*}} ["Volatile"] : !spirv.ptr<f32, Function>, !spirv.ptr<i32, Function>, i32
    spirv.CopyMemorySized %0, %1, %2 ["Volatile"] : !spirv.ptr<f32, Function>, !spirv.ptr<i32, Function>, i32

    // CHECK: spirv.CopyMemorySized %{{.*}}, %{{.*}}, %{{.*}} ["Aligned", 4] : !spirv.ptr<f32, Function>, !spirv.ptr<i32, Function>, i32
    spirv.CopyMemorySized %0, %1, %2 ["Aligned", 4] : !spirv.ptr<f32, Function>, !spirv.ptr<i32, Function>, i32

    // CHECK: spirv.CopyMemorySized %{{.*}}, %{{.*}}, %{{.*}} ["Volatile|Aligned", 4] : !spirv.ptr<f32, Function>, !spirv.ptr<i32, Function>, i32
    spirv.CopyMemorySized %0, %1, %2 ["Volatile|Aligned", 4] : !spirv.ptr<f32, Function>, !spirv.ptr<i32, Function>, i32

    spirv.Return
  }
}

// -----

spirv.module Physical64 OpenCL requires #spirv.vce<v1.4, [Kernel, Addresses, Linkage], []> {
  spirv.func @copy_memory_sized_with_two_access_operands() "None" {
    %0 = spirv.Variable : !spirv.ptr<f32, Function>
    %1 = spirv.Variable : !spirv.ptr<i32, Function>
    %2 = spirv.Constant 4 : i32

    // CHECK: spirv.CopyMemorySized %{{.*}}, %{{.*}}, %{{.*}} ["Volatile"], ["Volatile"] : !spirv.ptr<f32, Function>, !spirv.ptr<i32, Function>, i32
    spirv.CopyMemorySized %0, %1, %2 ["Volatile"], ["Volatile"] : !spirv.ptr<f32, Function>, !spirv.ptr<i32, Function>, i32

    // CHECK: spirv.CopyMemorySized %{{.*}}, %{{.*}}, %{{.*}} ["Aligned", 4], ["Volatile"] : !spirv.ptr<f32, Function>, !spirv.ptr<i32, Function>, i32
    spirv.CopyMemorySized %0, %1, %2 ["Aligned", 4], ["Volatile"] : !spirv.ptr<f32, Function>, !spirv.ptr<i32, Function>, i32

    // CHECK: spirv.CopyMemorySized %{{.*}}, %{{.*}}, %{{.*}} ["Volatile"], ["Aligned", 4] : !spirv.ptr<f32, Function>, !spirv.ptr<i32, Function>, i32
    spirv.CopyMemorySized %0, %1, %2 ["Volatile"], ["Aligned", 4] : !spirv.ptr<f32, Function>, !spirv.ptr<i32, Function>, i32

    // CHECK: spirv.CopyMemorySized %{{.*}}, %{{.*}}, %{{.*}} ["Aligned", 8], ["Aligned", 4] : !spirv.ptr<f32, Function>, !spirv.ptr<i32, Function>, i32
    spirv.CopyMemorySized %0, %1, %2 ["Aligned", 8], ["Aligned", 4] : !spirv.ptr<f32, Function>, !spirv.ptr<i32, Function>, i32

    // CHECK: spirv.CopyMemorySized %{{.*}}, %{{.*}}, %{{.*}} ["Volatile|Aligned", 8], ["Volatile|Aligned", 4]
    spirv.CopyMemorySized %0, %1, %2 ["Volatile|Aligned", 8], ["Volatile|Aligned", 4] : !spirv.ptr<f32, Function>, !spirv.ptr<i32, Function>, i32

    // Source-only masks acquire an explicit None target mask in text and binary.
    // CHECK: spirv.CopyMemorySized %{{.*}}, %{{.*}}, %{{.*}} ["None"], ["Volatile"] : !spirv.ptr<f32, Function>, !spirv.ptr<i32, Function>, i32
    "spirv.CopyMemorySized"(%0, %1, %2) {source_memory_access = #spirv.memory_access<Volatile>} : (!spirv.ptr<f32, Function>, !spirv.ptr<i32, Function>, i32) -> ()
    // CHECK: spirv.CopyMemorySized %{{.*}}, %{{.*}}, %{{.*}} ["None"], ["Volatile|Aligned", 4] : !spirv.ptr<f32, Function>, !spirv.ptr<i32, Function>, i32
    "spirv.CopyMemorySized"(%0, %1, %2) {source_memory_access = #spirv.memory_access<Volatile|Aligned>, source_alignment = 4 : i32} : (!spirv.ptr<f32, Function>, !spirv.ptr<i32, Function>, i32) -> ()
    // CHECK: spirv.CopyMemorySized %{{.*}}, %{{.*}}, %{{.*}} ["None"], ["None"] : !spirv.ptr<f32, Function>, !spirv.ptr<i32, Function>, i32
    "spirv.CopyMemorySized"(%0, %1, %2) {source_memory_access = #spirv.memory_access<None>} : (!spirv.ptr<f32, Function>, !spirv.ptr<i32, Function>, i32) -> ()

    spirv.Return
  }
}

// -----

spirv.module Physical64 OpenCL requires #spirv.vce<v1.4, [Kernel, Addresses, Linkage], []> {
  spirv.func @copy_memory_combined_masks(%dst: !spirv.ptr<i32, Function>, %src: !spirv.ptr<i32, Function>) "None" {
    // CHECK: spirv.CopyMemory "Function" %{{.*}}, "Function" %{{.*}} ["Volatile|Aligned", 4], ["Volatile|Aligned", 8] : i32
    spirv.CopyMemory "Function" %dst, "Function" %src ["Volatile|Aligned", 4], ["Volatile|Aligned", 8] : i32
    // CHECK: spirv.CopyMemory "Function" %{{.*}}, "Function" %{{.*}} ["None"], ["Volatile"] : i32
    "spirv.CopyMemory"(%dst, %src) {source_memory_access = #spirv.memory_access<Volatile>} : (!spirv.ptr<i32, Function>, !spirv.ptr<i32, Function>) -> ()
    // CHECK: spirv.CopyMemory "Function" %{{.*}}, "Function" %{{.*}} ["None"], ["Volatile|Aligned", 4] : i32
    "spirv.CopyMemory"(%dst, %src) {source_memory_access = #spirv.memory_access<Volatile|Aligned>, source_alignment = 4 : i32} : (!spirv.ptr<i32, Function>, !spirv.ptr<i32, Function>) -> ()
    // CHECK: spirv.CopyMemory "Function" %{{.*}}, "Function" %{{.*}} ["None"], ["None"] : i32
    "spirv.CopyMemory"(%dst, %src) {source_memory_access = #spirv.memory_access<None>} : (!spirv.ptr<i32, Function>, !spirv.ptr<i32, Function>) -> ()
    spirv.Return
  }
}
