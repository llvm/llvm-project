// RUN: mlir-translate -no-implicit-module -split-input-file -test-spirv-roundtrip %s | FileCheck %s

// RUN: %if spirv-tools %{ rm -rf %t %}
// RUN: %if spirv-tools %{ mkdir %t %}
// RUN: %if spirv-tools %{ mlir-translate --no-implicit-module --serialize-spirv --split-input-file --spirv-save-validation-files-with-prefix=%t/module %s %}
// RUN: %if spirv-tools %{ spirv-val %t %}

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage], [SPV_KHR_storage_buffer_storage_class]> {
  spirv.func @foo() -> () "None" {
    // CHECK: %[[UNDEF:.*]] = spirv.Undef : f32
    // CHECK-NEXT: spirv.FAdd %[[UNDEF]], %[[UNDEF]] : f32
    %0 = spirv.Undef : f32
    %1 = spirv.Undef : f32
    %2 = spirv.FAdd %0, %1 : f32
    // CHECK: {{%.*}} = spirv.Undef : vector<4xi32>
    %3 = spirv.Undef : vector<4xi32>
    %4 = spirv.CompositeExtract %3[1 : i32] : vector<4xi32>
    // CHECK: {{%.*}} = spirv.Undef : !spirv.array<4 x !spirv.array<4 x i32>>
    %5 = spirv.Undef : !spirv.array<4x!spirv.array<4xi32>>
    %6 = spirv.CompositeExtract %5[1 : i32, 2 : i32] : !spirv.array<4x!spirv.array<4xi32>>
    // CHECK: {{%.*}} = spirv.Undef : !spirv.ptr<!spirv.struct<(f32), Block>, StorageBuffer>
    %7 = spirv.Undef : !spirv.ptr<!spirv.struct<(f32), Block>, StorageBuffer>
    %8 = spirv.Constant 0 : i32
    %9 = spirv.AccessChain %7[%8] : !spirv.ptr<!spirv.struct<(f32), Block>, StorageBuffer>, i32 -> !spirv.ptr<f32, StorageBuffer>
    spirv.Return
  }

  // CHECK-LABEL: @undef_once_per_block
  spirv.func @undef_once_per_block() -> f32 "None" {
    // CHECK: %[[U1:.*]] = spirv.Undef : f32
    // CHECK-NEXT: %[[ADD:.*]] = spirv.FAdd %[[U1]], %[[U1]]
    // CHECK-NEXT: spirv.Branch ^[[BB:.*]]
    // CHECK-NEXT: ^[[BB]]:
    // CHECK-NEXT: %[[U2:.*]] = spirv.Undef : f32
    // CHECK-NEXT: spirv.FAdd %[[ADD]], %[[U2]]
    %0 = spirv.Undef : f32
    %1 = spirv.FAdd %0, %0 : f32
    spirv.Branch ^bb1
  ^bb1:
    %2 = spirv.FAdd %1, %0 : f32
    spirv.ReturnValue %2 : f32
  }
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage], []> {
  // CHECK: spirv.func {{@.*}}
  spirv.func @ignore_unused_undef() -> () "None" {
    // CHECK-NEXT: spirv.Return
    %0 = spirv.Undef : f32
    spirv.Return
  }
}
