// RUN: mlir-translate -no-implicit-module -test-spirv-roundtrip-debug -mlir-print-debuginfo -mlir-print-local-scope %s | FileCheck %s

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  // CHECK: loc({{".*debug.mlir"}}:5:3)
  spirv.GlobalVariable @var0 bind(0, 1) : !spirv.ptr<f32, Input>
  spirv.func @arithmetic(%arg0 : vector<4xf32>, %arg1 : vector<4xf32>) "None" {
    // CHECK: loc({{".*debug.mlir"}}:8:10)
    %0 = spirv.FAdd %arg0, %arg1 : vector<4xf32>
    // CHECK: loc({{".*debug.mlir"}}:10:10)
    %1 = spirv.FNegate %arg0 : vector<4xf32>
    spirv.Return
  }

  spirv.func @atomic(%ptr: !spirv.ptr<i32, Workgroup>, %value: i32, %comparator: i32) "None" {
    // CHECK: loc({{".*debug.mlir"}}:16:10)
    %1 = spirv.AtomicAnd "Device" "None" %ptr, %value : !spirv.ptr<i32, Workgroup>
    spirv.Return
  }

  spirv.func @bitwiser(%arg0 : i32, %arg1 : i32) "None" {
    // CHECK: loc({{".*debug.mlir"}}:22:10)
    %0 = spirv.BitwiseAnd %arg0, %arg1 : i32
    spirv.Return
  }

  spirv.func @convert(%arg0 : f32) "None" {
    // CHECK: loc({{".*debug.mlir"}}:28:10)
    %0 = spirv.ConvertFToU %arg0 : f32 to i32
    spirv.Return
  }

  spirv.func @composite(%arg0 : !spirv.struct<(f32, !spirv.struct<(!spirv.array<4xf32>, f32)>)>, %arg1: !spirv.array<4xf32>, %arg2 : f32, %arg3 : f32) "None" {
    // CHECK: loc({{".*debug.mlir"}}:34:10)
    %0 = spirv.CompositeInsert %arg1, %arg0[1 : i32, 0 : i32] : !spirv.array<4xf32> into !spirv.struct<(f32, !spirv.struct<(!spirv.array<4xf32>, f32)>)>
    // CHECK: loc({{".*debug.mlir"}}:36:10)
    %1 = spirv.CompositeConstruct %arg2, %arg3 : (f32, f32) -> vector<2xf32>
    spirv.Return
  }

  spirv.func @group_non_uniform(%val: f32) "None" {
    // CHECK: loc({{".*debug.mlir"}}:42:10)
    %0 = spirv.GroupNonUniformFAdd "Workgroup" "Reduce" %val : f32
    spirv.Return
  }

  spirv.func @local_var() "None" {
    %zero = spirv.Constant 0: i32
    // CHECK: loc({{".*debug.mlir"}}:49:12)
    %var = spirv.Variable init(%zero) : !spirv.ptr<i32, Function>
    spirv.Return
  }

  spirv.func @logical(%arg0: i32, %arg1: i32) "None" {
    // CHECK: loc({{".*debug.mlir"}}:55:10)
    %0 = spirv.IEqual %arg0, %arg1 : i32
    spirv.Return
  }

  spirv.func @memory_accesses(%arg0 : !spirv.ptr<!spirv.array<4x!spirv.array<4xf32>>, StorageBuffer>, %arg1 : i32, %arg2 : i32) "None" {
    // CHECK: loc({{".*debug.mlir"}}:61:10)
    %2 = spirv.AccessChain %arg0[%arg1, %arg2] : !spirv.ptr<!spirv.array<4x!spirv.array<4xf32>>, StorageBuffer>, i32, i32
    // CHECK: loc({{".*debug.mlir"}}:63:10)
    %3 = spirv.Load "StorageBuffer" %2 : f32
    // CHECK: loc({{.*debug.mlir"}}:65:5)
    spirv.Store "StorageBuffer" %2, %3 : f32
    // CHECK: loc({{".*debug.mlir"}}:67:5)
    spirv.Return
  }

  spirv.func @loop(%count : i32) -> () "None" {
    %zero = spirv.Constant 0: i32
    %one = spirv.Constant 1: i32
    %ivar = spirv.Variable init(%zero) : !spirv.ptr<i32, Function>
    %jvar = spirv.Variable init(%zero) : !spirv.ptr<i32, Function>
    spirv.mlir.loop {
      // CHECK: loc({{".*debug.mlir"}}:75:5)
      spirv.Branch ^header
    ^header:
      %ival0 = spirv.Load "Function" %ivar : i32
      %icmp = spirv.SLessThan %ival0, %count : i32
      // CHECK: loc({{".*debug.mlir"}}:75:5)
      spirv.BranchConditional %icmp, ^body, ^merge
    ^body:
      spirv.Store "Function" %jvar, %zero : i32
      spirv.mlir.loop {
        // CHECK: loc({{".*debug.mlir"}}:85:7)
        spirv.Branch ^header
      ^header:
        %jval0 = spirv.Load "Function" %jvar : i32
        %jcmp = spirv.SLessThan %jval0, %count : i32
        // CHECK: loc({{".*debug.mlir"}}:85:7)
        spirv.BranchConditional %jcmp, ^body, ^merge
      ^body:
        // CHECK: loc({{".*debug.mlir"}}:95:9)
        spirv.Branch ^continue
      ^continue:
        %jval1 = spirv.Load "Function" %jvar : i32
        %add = spirv.IAdd %jval1, %one : i32
        spirv.Store "Function" %jvar, %add : i32
        // CHECK: loc({{".*debug.mlir"}}:101:9)
        spirv.Branch ^header
      ^merge:
        // CHECK: loc({{".*debug.mlir"}}:85:7)
        spirv.mlir.merge
        // CHECK: loc({{".*debug.mlir"}}:85:7)
      }
      // CHECK: loc({{".*debug.mlir"}}:108:7)
      spirv.Branch ^continue
    ^continue:
      %ival1 = spirv.Load "Function" %ivar : i32
      %add = spirv.IAdd %ival1, %one : i32
      spirv.Store "Function" %ivar, %add : i32
      // CHECK: loc({{".*debug.mlir"}}:114:7)
      spirv.Branch ^header
    ^merge:
      // CHECK: loc({{".*debug.mlir"}}:75:5)
      spirv.mlir.merge
    // CHECK: loc({{".*debug.mlir"}}:75:5)
    }
    spirv.Return
  }

  spirv.func @selection(%cond: i1) -> () "None" {
    %zero = spirv.Constant 0: i32
    %one = spirv.Constant 1: i32
    %two = spirv.Constant 2: i32
    %var = spirv.Variable init(%zero) : !spirv.ptr<i32, Function>
    spirv.mlir.selection {
      // CHECK: loc({{".*debug.mlir"}}:128:5)
      spirv.BranchConditional %cond [5, 10], ^then, ^else
    ^then:
      spirv.Store "Function" %var, %one : i32
      // CHECK: loc({{".*debug.mlir"}}:134:7)
      spirv.Branch ^merge
    ^else:
      spirv.Store "Function" %var, %two : i32
      // CHECK: loc({{".*debug.mlir"}}:138:7)
      spirv.Branch ^merge
    ^merge:
      // CHECK: loc({{".*debug.mlir"}}:128:5)
      spirv.mlir.merge
    // CHECK: loc({{".*debug.mlir"}}:128:5)
    }
    spirv.Return
  }
}
