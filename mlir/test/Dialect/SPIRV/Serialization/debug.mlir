// RUN: mlir-translate -test-spirv-roundtrip-debug -mlir-print-debuginfo %s | FileCheck %s

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  // CHECK: loc({{".*debug.mlir"}}:5:3)
  spv.globalVariable @var0 bind(0, 1) : !spv.ptr<f32, Input>
  spv.func @arithmetic(%arg0 : vector<4xf32>, %arg1 : vector<4xf32>) "None" {
    // CHECK: loc({{".*debug.mlir"}}:8:10)
    %0 = spv.FAdd %arg0, %arg1 : vector<4xf32>
    // CHECK: loc({{".*debug.mlir"}}:10:10)
    %1 = spv.FNegate %arg0 : vector<4xf32>
    spv.Return
  }

  spv.func @atomic(%ptr: !spv.ptr<i32, Workgroup>, %value: i32, %comparator: i32) "None" {
    // CHECK: loc({{".*debug.mlir"}}:16:10)
    %1 = spv.AtomicAnd "Device" "None" %ptr, %value : !spv.ptr<i32, Workgroup>
    spv.Return
  }

  spv.func @bitwiser(%arg0 : i32, %arg1 : i32) "None" {
    // CHECK: loc({{".*debug.mlir"}}:22:10)
    %0 = spv.BitwiseAnd %arg0, %arg1 : i32
    spv.Return
  }

  spv.func @convert(%arg0 : f32) "None" {
    // CHECK: loc({{".*debug.mlir"}}:28:10)
    %0 = spv.ConvertFToU %arg0 : f32 to i32
    spv.Return
  }

  spv.func @composite(%arg0 : !spv.struct<f32, !spv.struct<!spv.array<4xf32>, f32>>, %arg1: !spv.array<4xf32>, %arg2 : f32, %arg3 : f32) "None" {
    // CHECK: loc({{".*debug.mlir"}}:34:10)
    %0 = spv.CompositeInsert %arg1, %arg0[1 : i32, 0 : i32] : !spv.array<4xf32> into !spv.struct<f32, !spv.struct<!spv.array<4xf32>, f32>>
    // CHECK: loc({{".*debug.mlir"}}:36:10)
    %1 = spv.CompositeConstruct %arg2, %arg3 : vector<2xf32>
    spv.Return
  }

  spv.func @group_non_uniform(%val: f32) "None" {
    // CHECK: loc({{".*debug.mlir"}}:42:10)
    %0 = spv.GroupNonUniformFAdd "Workgroup" "Reduce" %val : f32
    spv.Return
  }

  spv.func @local_var() "None" {
    %zero = spv.constant 0: i32
    // CHECK: loc({{".*debug.mlir"}}:49:12)
    %var = spv.Variable init(%zero) : !spv.ptr<i32, Function>
    spv.Return
  }

  spv.func @logical(%arg0: i32, %arg1: i32) "None" {
    // CHECK: loc({{".*debug.mlir"}}:55:10)
    %0 = spv.IEqual %arg0, %arg1 : i32
    spv.Return
  }

  spv.func @memory_accesses(%arg0 : !spv.ptr<!spv.array<4x!spv.array<4xf32>>, StorageBuffer>, %arg1 : i32, %arg2 : i32) "None" {
    // CHECK: loc({{".*debug.mlir"}}:61:10)
    %2 = spv.AccessChain %arg0[%arg1, %arg2] : !spv.ptr<!spv.array<4x!spv.array<4xf32>>, StorageBuffer>
    // CHECK: loc({{".*debug.mlir"}}:63:10)
    %3 = spv.Load "StorageBuffer" %2 : f32
    // CHECK: loc({{.*debug.mlir"}}:65:5)
    spv.Store "StorageBuffer" %2, %3 : f32
    // CHECK: loc({{".*debug.mlir"}}:67:5)
    spv.Return
  }
}
