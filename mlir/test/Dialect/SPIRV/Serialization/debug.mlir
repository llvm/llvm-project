// RUN: mlir-translate -test-spirv-roundtrip-debug -mlir-print-debuginfo %s | FileCheck %s

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  spv.func @arithmetic(%arg0 : vector<4xf32>, %arg1 : vector<4xf32>) "None" {
    // CHECK: loc({{".*debug.mlir"}}:6:10)
    %0 = spv.FAdd %arg0, %arg1 : vector<4xf32>
    // CHECK: loc({{".*debug.mlir"}}:8:10)
    %1 = spv.FNegate %arg0 : vector<4xf32>
    spv.Return
  }

  spv.func @atomic(%ptr: !spv.ptr<i32, Workgroup>, %value: i32, %comparator: i32) "None" {
    // CHECK: loc({{".*debug.mlir"}}:14:10)
    %1 = spv.AtomicAnd "Device" "None" %ptr, %value : !spv.ptr<i32, Workgroup>
    spv.Return
  }

  spv.func @bitwiser(%arg0 : i32, %arg1 : i32) "None" {
    // CHECK: loc({{".*debug.mlir"}}:20:10)
    %0 = spv.BitwiseAnd %arg0, %arg1 : i32
    spv.Return
  }

  spv.func @convert(%arg0 : f32) "None" {
    // CHECK: loc({{".*debug.mlir"}}:26:10)
    %0 = spv.ConvertFToU %arg0 : f32 to i32
    spv.Return
  }

  spv.func @composite(%arg0 : !spv.struct<f32, !spv.struct<!spv.array<4xf32>, f32>>, %arg1: !spv.array<4xf32>, %arg2 : f32, %arg3 : f32) "None" {
    // CHECK: loc({{".*debug.mlir"}}:32:10)
    %0 = spv.CompositeInsert %arg1, %arg0[1 : i32, 0 : i32] : !spv.array<4xf32> into !spv.struct<f32, !spv.struct<!spv.array<4xf32>, f32>>
    // CHECK: loc({{".*debug.mlir"}}:34:10)
    %1 = spv.CompositeConstruct %arg2, %arg3 : vector<2xf32>
    spv.Return
  }

  spv.func @group_non_uniform(%val: f32) "None" {
    // CHECK: loc({{".*debug.mlir"}}:40:10)
    %0 = spv.GroupNonUniformFAdd "Workgroup" "Reduce" %val : f32
    spv.Return
  }

  spv.func @logical(%arg0: i32, %arg1: i32) "None" {
    // CHECK: loc({{".*debug.mlir"}}:46:10)
    %0 = spv.IEqual %arg0, %arg1 : i32
    spv.Return
  }

  spv.func @memory_accesses(%arg0 : !spv.ptr<!spv.array<4x!spv.array<4xf32>>, StorageBuffer>, %arg1 : i32, %arg2 : i32) "None" {
    // CHECK: loc({{".*debug.mlir"}}:52:10)
    %2 = spv.AccessChain %arg0[%arg1, %arg2] : !spv.ptr<!spv.array<4x!spv.array<4xf32>>, StorageBuffer>
    // CHECK: loc({{".*debug.mlir"}}:54:10)
    %3 = spv.Load "StorageBuffer" %2 : f32
    // CHECK: loc({{.*debug.mlir"}}:56:5)
    spv.Store "StorageBuffer" %2, %3 : f32
    // CHECK: loc({{".*debug.mlir"}}:58:5)
    spv.Return
  }
}
