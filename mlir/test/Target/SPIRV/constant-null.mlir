// RUN: mlir-translate --no-implicit-module --test-spirv-roundtrip %s | FileCheck %s
// RUN: %if spirv-tools %{ mlir-translate --no-implicit-module --serialize-spirv %s | spirv-val %}

spirv.module Physical64 OpenCL requires #spirv.vce<v1.0, [Kernel, Addresses, Linkage], []> {
  // CHECK-LABEL: @null_const
  spirv.func @null_const() "None" {
    // CHECK: spirv.Constant #spirv.null : !spirv.ptr<f32, CrossWorkgroup>
    %0 = spirv.Constant #spirv.null : !spirv.ptr<f32, CrossWorkgroup>
    // CHECK: spirv.Constant #spirv.null : !spirv.ptr<i32, Function>
    %1 = spirv.Constant #spirv.null : !spirv.ptr<i32, Function>
    // CHECK: spirv.Constant #spirv.null : !spirv.array<4 x i32>
    %2 = spirv.Constant #spirv.null : !spirv.array<4 x i32>
    // CHECK: spirv.Constant #spirv.null : !spirv.struct<S, (i32, !spirv.ptr<f32, CrossWorkgroup>)>
    %3 = spirv.Constant #spirv.null : !spirv.struct<S, (i32, !spirv.ptr<f32, CrossWorkgroup>)>
    // CHECK: spirv.Constant [#spirv.null : !spirv.ptr<f32, CrossWorkgroup>, 1 : i32] : !spirv.struct<(!spirv.ptr<f32, CrossWorkgroup>, i32)>
    %4 = spirv.Constant [#spirv.null : !spirv.ptr<f32, CrossWorkgroup>, 1 : i32] : !spirv.struct<(!spirv.ptr<f32, CrossWorkgroup>, i32)>
    %5 = spirv.Variable init(%0) : !spirv.ptr<!spirv.ptr<f32, CrossWorkgroup>, Function>
    %6 = spirv.Variable init(%1) : !spirv.ptr<!spirv.ptr<i32, Function>, Function>
    %7 = spirv.Variable init(%2) : !spirv.ptr<!spirv.array<4 x i32>, Function>
    %8 = spirv.Variable init(%3) : !spirv.ptr<!spirv.struct<S, (i32, !spirv.ptr<f32, CrossWorkgroup>)>, Function>
    %9 = spirv.Variable init(%4) : !spirv.ptr<!spirv.struct<(!spirv.ptr<f32, CrossWorkgroup>, i32)>, Function>
    spirv.Return
  }
}
