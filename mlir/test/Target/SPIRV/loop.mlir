// RUN: mlir-translate -split-input-file -test-spirv-roundtrip %s | FileCheck %s

// Single loop

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  // for (int i = 0; i < count; ++i) {}
// CHECK-LABEL: @loop
  spirv.func @loop(%count : i32) -> () "None" {
    %zero = spirv.Constant 0: i32
    %one = spirv.Constant 1: i32
    %var = spirv.Variable init(%zero) : !spirv.ptr<i32, Function>

// CHECK:        spirv.Branch ^bb1
// CHECK-NEXT: ^bb1:
// CHECK-NEXT:   spirv.mlir.loop
    spirv.mlir.loop {
// CHECK-NEXT:     spirv.Branch ^bb1
      spirv.Branch ^header

// CHECK-NEXT:   ^bb1:
    ^header:
// CHECK-NEXT:     spirv.Load
      %val0 = spirv.Load "Function" %var : i32
// CHECK-NEXT:     spirv.SLessThan
      %cmp = spirv.SLessThan %val0, %count : i32
// CHECK-NEXT:     spirv.BranchConditional %{{.*}} [1, 1], ^bb2, ^bb4
      spirv.BranchConditional %cmp [1, 1], ^body, ^merge

// CHECK-NEXT:   ^bb2:
    ^body:
      // Do nothing
// CHECK-NEXT:     spirv.Branch ^bb3
      spirv.Branch ^continue

// CHECK-NEXT:   ^bb3:
    ^continue:
// CHECK-NEXT:     spirv.Load
      %val1 = spirv.Load "Function" %var : i32
// CHECK-NEXT:     spirv.Constant 1
// CHECK-NEXT:     spirv.IAdd
      %add = spirv.IAdd %val1, %one : i32
// CHECK-NEXT:     spirv.Store
      spirv.Store "Function" %var, %add : i32
// CHECK-NEXT:     spirv.Branch ^bb1
      spirv.Branch ^header

// CHECK-NEXT:   ^bb4:
// CHECK-NEXT:     spirv.mlir.merge
    ^merge:
      spirv.mlir.merge
    }
    spirv.Return
  }

  spirv.func @main() -> () "None" {
    spirv.Return
  }
  spirv.EntryPoint "GLCompute" @main
}

// -----

// Single loop with block arguments

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  spirv.GlobalVariable @GV1 bind(0, 0) : !spirv.ptr<!spirv.struct<(!spirv.array<10 x f32, stride=4> [0])>, StorageBuffer>
  spirv.GlobalVariable @GV2 bind(0, 1) : !spirv.ptr<!spirv.struct<(!spirv.array<10 x f32, stride=4> [0])>, StorageBuffer>
// CHECK-LABEL: @loop_kernel
  spirv.func @loop_kernel() "None" {
    %0 = spirv.mlir.addressof @GV1 : !spirv.ptr<!spirv.struct<(!spirv.array<10 x f32, stride=4> [0])>, StorageBuffer>
    %1 = spirv.Constant 0 : i32
    %2 = spirv.AccessChain %0[%1] : !spirv.ptr<!spirv.struct<(!spirv.array<10 x f32, stride=4> [0])>, StorageBuffer>, i32
    %3 = spirv.mlir.addressof @GV2 : !spirv.ptr<!spirv.struct<(!spirv.array<10 x f32, stride=4> [0])>, StorageBuffer>
    %5 = spirv.AccessChain %3[%1] : !spirv.ptr<!spirv.struct<(!spirv.array<10 x f32, stride=4> [0])>, StorageBuffer>, i32
    %6 = spirv.Constant 4 : i32
    %7 = spirv.Constant 42 : i32
    %8 = spirv.Constant 2 : i32
// CHECK:        spirv.Branch ^bb1(%{{.*}} : i32)
// CHECK-NEXT: ^bb1(%[[OUTARG:.*]]: i32):
// CHECK-NEXT:   spirv.mlir.loop {
    spirv.mlir.loop {
// CHECK-NEXT:     spirv.Branch ^bb1(%[[OUTARG]] : i32)
      spirv.Branch ^header(%6 : i32)
// CHECK-NEXT:   ^bb1(%[[HEADARG:.*]]: i32):
    ^header(%9: i32):
      %10 = spirv.SLessThan %9, %7 : i32
// CHECK:          spirv.BranchConditional %{{.*}}, ^bb2, ^bb3
      spirv.BranchConditional %10, ^body, ^merge
// CHECK-NEXT:   ^bb2:     // pred: ^bb1
    ^body:
      %11 = spirv.AccessChain %2[%9] : !spirv.ptr<!spirv.array<10 x f32, stride=4>, StorageBuffer>, i32
      %12 = spirv.Load "StorageBuffer" %11 : f32
      %13 = spirv.AccessChain %5[%9] : !spirv.ptr<!spirv.array<10 x f32, stride=4>, StorageBuffer>, i32
      spirv.Store "StorageBuffer" %13, %12 : f32
// CHECK:          %[[ADD:.*]] = spirv.IAdd
      %14 = spirv.IAdd %9, %8 : i32
// CHECK-NEXT:     spirv.Branch ^bb1(%[[ADD]] : i32)
      spirv.Branch ^header(%14 : i32)
// CHECK-NEXT:   ^bb3:
    ^merge:
// CHECK-NEXT:     spirv.mlir.merge
      spirv.mlir.merge
    }
    spirv.Return
  }
  spirv.EntryPoint "GLCompute" @loop_kernel
  spirv.ExecutionMode @loop_kernel "LocalSize", 1, 1, 1
}

// -----

// Nested loop

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  // for (int i = 0; i < count; ++i) {
  //   for (int j = 0; j < count; ++j) { }
  // }
// CHECK-LABEL: @loop
  spirv.func @loop(%count : i32) -> () "None" {
    %zero = spirv.Constant 0: i32
    %one = spirv.Constant 1: i32
    %ivar = spirv.Variable init(%zero) : !spirv.ptr<i32, Function>
    %jvar = spirv.Variable init(%zero) : !spirv.ptr<i32, Function>

// CHECK:        spirv.Branch ^bb1
// CHECK-NEXT: ^bb1:
// CHECK-NEXT:   spirv.mlir.loop control(Unroll)
    spirv.mlir.loop control(Unroll) {
// CHECK-NEXT:     spirv.Branch ^bb1
      spirv.Branch ^header

// CHECK-NEXT:   ^bb1:
    ^header:
// CHECK-NEXT:     spirv.Load
      %ival0 = spirv.Load "Function" %ivar : i32
// CHECK-NEXT:     spirv.SLessThan
      %icmp = spirv.SLessThan %ival0, %count : i32
// CHECK-NEXT:     spirv.BranchConditional %{{.*}}, ^bb2, ^bb5
      spirv.BranchConditional %icmp, ^body, ^merge

// CHECK-NEXT:   ^bb2:
    ^body:
// CHECK-NEXT:     spirv.Constant 0
// CHECK-NEXT: 		 spirv.Store
      spirv.Store "Function" %jvar, %zero : i32
// CHECK-NEXT:     spirv.Branch ^bb3
// CHECK-NEXT:   ^bb3:
// CHECK-NEXT:     spirv.mlir.loop control(DontUnroll)
      spirv.mlir.loop control(DontUnroll) {
// CHECK-NEXT:       spirv.Branch ^bb1
        spirv.Branch ^header

// CHECK-NEXT:     ^bb1:
      ^header:
// CHECK-NEXT:       spirv.Load
        %jval0 = spirv.Load "Function" %jvar : i32
// CHECK-NEXT:       spirv.SLessThan
        %jcmp = spirv.SLessThan %jval0, %count : i32
// CHECK-NEXT:       spirv.BranchConditional %{{.*}}, ^bb2, ^bb4
        spirv.BranchConditional %jcmp, ^body, ^merge

// CHECK-NEXT:     ^bb2:
      ^body:
        // Do nothing
// CHECK-NEXT:       spirv.Branch ^bb3
        spirv.Branch ^continue

// CHECK-NEXT:     ^bb3:
      ^continue:
// CHECK-NEXT:       spirv.Load
        %jval1 = spirv.Load "Function" %jvar : i32
// CHECK-NEXT:       spirv.Constant 1
// CHECK-NEXT:       spirv.IAdd
        %add = spirv.IAdd %jval1, %one : i32
// CHECK-NEXT:       spirv.Store
        spirv.Store "Function" %jvar, %add : i32
// CHECK-NEXT:       spirv.Branch ^bb1
        spirv.Branch ^header

// CHECK-NEXT:     ^bb4:
      ^merge:
// CHECK-NEXT:       spirv.mlir.merge
        spirv.mlir.merge
      } // end inner loop

// CHECK:          spirv.Branch ^bb4
      spirv.Branch ^continue

// CHECK-NEXT:   ^bb4:
    ^continue:
// CHECK-NEXT:     spirv.Load
      %ival1 = spirv.Load "Function" %ivar : i32
// CHECK-NEXT:     spirv.Constant 1
// CHECK-NEXT:     spirv.IAdd
      %add = spirv.IAdd %ival1, %one : i32
// CHECK-NEXT:     spirv.Store
      spirv.Store "Function" %ivar, %add : i32
// CHECK-NEXT:     spirv.Branch ^bb1
      spirv.Branch ^header

// CHECK-NEXT:   ^bb5:
// CHECK-NEXT:     spirv.mlir.merge
    ^merge:
      spirv.mlir.merge
    } // end outer loop
    spirv.Return
  }

  spirv.func @main() -> () "None" {
    spirv.Return
  }
  spirv.EntryPoint "GLCompute" @main
}


// -----

// Loop with selection in its header

spirv.module Physical64 OpenCL requires #spirv.vce<v1.0, [Kernel, Linkage, Addresses, Int64], []> {
// CHECK-LABEL:   @kernel
// CHECK-SAME:    (%[[INPUT0:.+]]: i64)
  spirv.func @kernel(%input: i64) "None" {
// CHECK-NEXT:     %[[VAR:.+]] = spirv.Variable : !spirv.ptr<i1, Function>
// CHECK-NEXT:     spirv.Branch ^[[BB0:.+]](%[[INPUT0]] : i64)
// CHECK-NEXT:   ^[[BB0]](%[[INPUT1:.+]]: i64):
    %cst0_i64 = spirv.Constant 0 : i64
    %true = spirv.Constant true
    %false = spirv.Constant false
// CHECK-NEXT:     spirv.mlir.loop {
    spirv.mlir.loop {
// CHECK-NEXT:       spirv.Branch ^[[LOOP_HEADER:.+]](%[[INPUT1]] : i64)
      spirv.Branch ^loop_header(%input : i64)
// CHECK-NEXT:     ^[[LOOP_HEADER]](%[[ARG1:.+]]: i64):
    ^loop_header(%arg1: i64):
// CHECK-NEXT:       spirv.Branch ^[[LOOP_BODY:.+]]
// CHECK-NEXT:     ^[[LOOP_BODY]]:
// CHECK-NEXT:         %[[C0:.+]] = spirv.Constant 0 : i64
      %gt = spirv.SGreaterThan %arg1, %cst0_i64 : i64
// CHECK-NEXT:         %[[GT:.+]] = spirv.SGreaterThan %[[ARG1]], %[[C0]] : i64
// CHECK-NEXT:         spirv.Branch ^[[BB1:.+]]
// CHECK-NEXT:     ^[[BB1]]:
      %var = spirv.Variable : !spirv.ptr<i1, Function>
// CHECK-NEXT:       spirv.mlir.selection {
      spirv.mlir.selection {
// CHECK-NEXT:         spirv.BranchConditional %[[GT]], ^[[THEN:.+]], ^[[ELSE:.+]]
        spirv.BranchConditional %gt, ^then, ^else
// CHECK-NEXT:       ^[[THEN]]:
      ^then:
// CHECK-NEXT:         %true = spirv.Constant true
// CHECK-NEXT:         spirv.Store "Function" %[[VAR]], %true : i1
        spirv.Store "Function" %var, %true : i1
// CHECK-NEXT:         spirv.Branch ^[[SELECTION_MERGE:.+]]
        spirv.Branch ^selection_merge
// CHECK-NEXT:       ^[[ELSE]]:
      ^else:
// CHECK-NEXT:         %false = spirv.Constant false
// CHECK-NEXT:         spirv.Store "Function" %[[VAR]], %false : i1
        spirv.Store "Function" %var, %false : i1
// CHECK-NEXT:         spirv.Branch ^[[SELECTION_MERGE]]
        spirv.Branch ^selection_merge
// CHECK-NEXT:       ^[[SELECTION_MERGE]]:
      ^selection_merge:
// CHECK-NEXT:         spirv.mlir.merge
        spirv.mlir.merge
// CHECK-NEXT:       }
      }
// CHECK-NEXT:       %[[LOAD:.+]] = spirv.Load "Function" %[[VAR]] : i1
      %load = spirv.Load "Function" %var : i1
// CHECK-NEXT:       spirv.BranchConditional %[[LOAD]], ^[[CONTINUE:.+]](%[[ARG1]] : i64), ^[[LOOP_MERGE:.+]]
      spirv.BranchConditional %load, ^continue(%arg1 : i64), ^loop_merge
// CHECK-NEXT:     ^[[CONTINUE]](%[[ARG2:.+]]: i64):
    ^continue(%arg2: i64):
// CHECK-NEXT:       %[[C0:.+]] = spirv.Constant 0 : i64
// CHECK-NEXT:       %[[LT:.+]] = spirv.SLessThan %[[ARG2]], %[[C0]] : i64
      %lt = spirv.SLessThan %arg2, %cst0_i64 : i64
// CHECK-NEXT:       spirv.Store "Function" %[[VAR]], %[[LT]] : i1
      spirv.Store "Function" %var, %lt : i1
// CHECK-NEXT:       spirv.Branch ^[[LOOP_HEADER]](%[[ARG2]] : i64)
      spirv.Branch ^loop_header(%arg2 : i64)
// CHECK-NEXT:     ^[[LOOP_MERGE]]:
    ^loop_merge:
// CHECK-NEXT:       spirv.mlir.merge
      spirv.mlir.merge
// CHECK-NEXT:     }
    }
// CHECK-NEXT:     spirv.Return
    spirv.Return
  }
}
