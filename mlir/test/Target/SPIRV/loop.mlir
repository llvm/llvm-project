// RUN: mlir-translate -split-input-file -test-spirv-roundtrip %s | FileCheck %s

// Single loop

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  // for (int i = 0; i < count; ++i) {}
// CHECK-LABEL: @loop
  spv.func @loop(%count : i32) -> () "None" {
    %zero = spv.Constant 0: i32
    %one = spv.Constant 1: i32
    %var = spv.Variable init(%zero) : !spv.ptr<i32, Function>

// CHECK:        spv.Branch ^bb1
// CHECK-NEXT: ^bb1:
// CHECK-NEXT:   spv.mlir.loop
    spv.mlir.loop {
// CHECK-NEXT:     spv.Branch ^bb1
      spv.Branch ^header

// CHECK-NEXT:   ^bb1:
    ^header:
// CHECK-NEXT:     spv.Load
      %val0 = spv.Load "Function" %var : i32
// CHECK-NEXT:     spv.SLessThan
      %cmp = spv.SLessThan %val0, %count : i32
// CHECK-NEXT:     spv.BranchConditional %{{.*}} [1, 1], ^bb2, ^bb4
      spv.BranchConditional %cmp [1, 1], ^body, ^merge

// CHECK-NEXT:   ^bb2:
    ^body:
      // Do nothing
// CHECK-NEXT:     spv.Branch ^bb3
      spv.Branch ^continue

// CHECK-NEXT:   ^bb3:
    ^continue:
// CHECK-NEXT:     spv.Load
      %val1 = spv.Load "Function" %var : i32
// CHECK-NEXT:     spv.Constant 1
// CHECK-NEXT:     spv.IAdd
      %add = spv.IAdd %val1, %one : i32
// CHECK-NEXT:     spv.Store
      spv.Store "Function" %var, %add : i32
// CHECK-NEXT:     spv.Branch ^bb1
      spv.Branch ^header

// CHECK-NEXT:   ^bb4:
// CHECK-NEXT:     spv.mlir.merge
    ^merge:
      spv.mlir.merge
    }
    spv.Return
  }

  spv.func @main() -> () "None" {
    spv.Return
  }
  spv.EntryPoint "GLCompute" @main
}

// -----

// Single loop with block arguments

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  spv.GlobalVariable @GV1 bind(0, 0) : !spv.ptr<!spv.struct<(!spv.array<10 x f32, stride=4> [0])>, StorageBuffer>
  spv.GlobalVariable @GV2 bind(0, 1) : !spv.ptr<!spv.struct<(!spv.array<10 x f32, stride=4> [0])>, StorageBuffer>
// CHECK-LABEL: @loop_kernel
  spv.func @loop_kernel() "None" {
    %0 = spv.mlir.addressof @GV1 : !spv.ptr<!spv.struct<(!spv.array<10 x f32, stride=4> [0])>, StorageBuffer>
    %1 = spv.Constant 0 : i32
    %2 = spv.AccessChain %0[%1] : !spv.ptr<!spv.struct<(!spv.array<10 x f32, stride=4> [0])>, StorageBuffer>, i32
    %3 = spv.mlir.addressof @GV2 : !spv.ptr<!spv.struct<(!spv.array<10 x f32, stride=4> [0])>, StorageBuffer>
    %5 = spv.AccessChain %3[%1] : !spv.ptr<!spv.struct<(!spv.array<10 x f32, stride=4> [0])>, StorageBuffer>, i32
    %6 = spv.Constant 4 : i32
    %7 = spv.Constant 42 : i32
    %8 = spv.Constant 2 : i32
// CHECK:        spv.Branch ^bb1(%{{.*}} : i32)
// CHECK-NEXT: ^bb1(%[[OUTARG:.*]]: i32):
// CHECK-NEXT:   spv.mlir.loop {
    spv.mlir.loop {
// CHECK-NEXT:     spv.Branch ^bb1(%[[OUTARG]] : i32)
      spv.Branch ^header(%6 : i32)
// CHECK-NEXT:   ^bb1(%[[HEADARG:.*]]: i32):
    ^header(%9: i32):
      %10 = spv.SLessThan %9, %7 : i32
// CHECK:          spv.BranchConditional %{{.*}}, ^bb2, ^bb3
      spv.BranchConditional %10, ^body, ^merge
// CHECK-NEXT:   ^bb2:     // pred: ^bb1
    ^body:
      %11 = spv.AccessChain %2[%9] : !spv.ptr<!spv.array<10 x f32, stride=4>, StorageBuffer>, i32
      %12 = spv.Load "StorageBuffer" %11 : f32
      %13 = spv.AccessChain %5[%9] : !spv.ptr<!spv.array<10 x f32, stride=4>, StorageBuffer>, i32
      spv.Store "StorageBuffer" %13, %12 : f32
// CHECK:          %[[ADD:.*]] = spv.IAdd
      %14 = spv.IAdd %9, %8 : i32
// CHECK-NEXT:     spv.Branch ^bb1(%[[ADD]] : i32)
      spv.Branch ^header(%14 : i32)
// CHECK-NEXT:   ^bb3:
    ^merge:
// CHECK-NEXT:     spv.mlir.merge
      spv.mlir.merge
    }
    spv.Return
  }
  spv.EntryPoint "GLCompute" @loop_kernel
  spv.ExecutionMode @loop_kernel "LocalSize", 1, 1, 1
}

// -----

// Nested loop

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  // for (int i = 0; i < count; ++i) {
  //   for (int j = 0; j < count; ++j) { }
  // }
// CHECK-LABEL: @loop
  spv.func @loop(%count : i32) -> () "None" {
    %zero = spv.Constant 0: i32
    %one = spv.Constant 1: i32
    %ivar = spv.Variable init(%zero) : !spv.ptr<i32, Function>
    %jvar = spv.Variable init(%zero) : !spv.ptr<i32, Function>

// CHECK:        spv.Branch ^bb1
// CHECK-NEXT: ^bb1:
// CHECK-NEXT:   spv.mlir.loop control(Unroll)
    spv.mlir.loop control(Unroll) {
// CHECK-NEXT:     spv.Branch ^bb1
      spv.Branch ^header

// CHECK-NEXT:   ^bb1:
    ^header:
// CHECK-NEXT:     spv.Load
      %ival0 = spv.Load "Function" %ivar : i32
// CHECK-NEXT:     spv.SLessThan
      %icmp = spv.SLessThan %ival0, %count : i32
// CHECK-NEXT:     spv.BranchConditional %{{.*}}, ^bb2, ^bb5
      spv.BranchConditional %icmp, ^body, ^merge

// CHECK-NEXT:   ^bb2:
    ^body:
// CHECK-NEXT:     spv.Constant 0
// CHECK-NEXT: 		 spv.Store
      spv.Store "Function" %jvar, %zero : i32
// CHECK-NEXT:     spv.Branch ^bb3
// CHECK-NEXT:   ^bb3:
// CHECK-NEXT:     spv.mlir.loop control(DontUnroll)
      spv.mlir.loop control(DontUnroll) {
// CHECK-NEXT:       spv.Branch ^bb1
        spv.Branch ^header

// CHECK-NEXT:     ^bb1:
      ^header:
// CHECK-NEXT:       spv.Load
        %jval0 = spv.Load "Function" %jvar : i32
// CHECK-NEXT:       spv.SLessThan
        %jcmp = spv.SLessThan %jval0, %count : i32
// CHECK-NEXT:       spv.BranchConditional %{{.*}}, ^bb2, ^bb4
        spv.BranchConditional %jcmp, ^body, ^merge

// CHECK-NEXT:     ^bb2:
      ^body:
        // Do nothing
// CHECK-NEXT:       spv.Branch ^bb3
        spv.Branch ^continue

// CHECK-NEXT:     ^bb3:
      ^continue:
// CHECK-NEXT:       spv.Load
        %jval1 = spv.Load "Function" %jvar : i32
// CHECK-NEXT:       spv.Constant 1
// CHECK-NEXT:       spv.IAdd
        %add = spv.IAdd %jval1, %one : i32
// CHECK-NEXT:       spv.Store
        spv.Store "Function" %jvar, %add : i32
// CHECK-NEXT:       spv.Branch ^bb1
        spv.Branch ^header

// CHECK-NEXT:     ^bb4:
      ^merge:
// CHECK-NEXT:       spv.mlir.merge
        spv.mlir.merge
      } // end inner loop

// CHECK:          spv.Branch ^bb4
      spv.Branch ^continue

// CHECK-NEXT:   ^bb4:
    ^continue:
// CHECK-NEXT:     spv.Load
      %ival1 = spv.Load "Function" %ivar : i32
// CHECK-NEXT:     spv.Constant 1
// CHECK-NEXT:     spv.IAdd
      %add = spv.IAdd %ival1, %one : i32
// CHECK-NEXT:     spv.Store
      spv.Store "Function" %ivar, %add : i32
// CHECK-NEXT:     spv.Branch ^bb1
      spv.Branch ^header

// CHECK-NEXT:   ^bb5:
// CHECK-NEXT:     spv.mlir.merge
    ^merge:
      spv.mlir.merge
    } // end outer loop
    spv.Return
  }

  spv.func @main() -> () "None" {
    spv.Return
  }
  spv.EntryPoint "GLCompute" @main
}


// -----

// Loop with selection in its header

spv.module Physical64 OpenCL requires #spv.vce<v1.0, [Kernel, Linkage, Addresses, Int64], []> {
// CHECK-LABEL:   @kernel
// CHECK-SAME:    (%[[INPUT0:.+]]: i64)
  spv.func @kernel(%input: i64) "None" {
// CHECK-NEXT:     %[[VAR:.+]] = spv.Variable : !spv.ptr<i1, Function>
// CHECK-NEXT:     spv.Branch ^[[BB0:.+]](%[[INPUT0]] : i64)
// CHECK-NEXT:   ^[[BB0]](%[[INPUT1:.+]]: i64):
    %cst0_i64 = spv.Constant 0 : i64
    %true = spv.Constant true
    %false = spv.Constant false
// CHECK-NEXT:     spv.mlir.loop {
    spv.mlir.loop {
// CHECK-NEXT:       spv.Branch ^[[LOOP_HEADER:.+]](%[[INPUT1]] : i64)
      spv.Branch ^loop_header(%input : i64)
// CHECK-NEXT:     ^[[LOOP_HEADER]](%[[ARG1:.+]]: i64):
    ^loop_header(%arg1: i64):
// CHECK-NEXT:       spv.Branch ^[[LOOP_BODY:.+]]
// CHECK-NEXT:     ^[[LOOP_BODY]]:
// CHECK-NEXT:         %[[C0:.+]] = spv.Constant 0 : i64
      %gt = spv.SGreaterThan %arg1, %cst0_i64 : i64
// CHECK-NEXT:         %[[GT:.+]] = spv.SGreaterThan %[[ARG1]], %[[C0]] : i64
// CHECK-NEXT:         spv.Branch ^[[BB1:.+]]
// CHECK-NEXT:     ^[[BB1]]:
      %var = spv.Variable : !spv.ptr<i1, Function>
// CHECK-NEXT:       spv.mlir.selection {
      spv.mlir.selection {
// CHECK-NEXT:         spv.BranchConditional %[[GT]], ^[[THEN:.+]], ^[[ELSE:.+]]
        spv.BranchConditional %gt, ^then, ^else
// CHECK-NEXT:       ^[[THEN]]:
      ^then:
// CHECK-NEXT:         %true = spv.Constant true
// CHECK-NEXT:         spv.Store "Function" %[[VAR]], %true : i1
        spv.Store "Function" %var, %true : i1
// CHECK-NEXT:         spv.Branch ^[[SELECTION_MERGE:.+]]
        spv.Branch ^selection_merge
// CHECK-NEXT:       ^[[ELSE]]:
      ^else:
// CHECK-NEXT:         %false = spv.Constant false
// CHECK-NEXT:         spv.Store "Function" %[[VAR]], %false : i1
        spv.Store "Function" %var, %false : i1
// CHECK-NEXT:         spv.Branch ^[[SELECTION_MERGE]]
        spv.Branch ^selection_merge
// CHECK-NEXT:       ^[[SELECTION_MERGE]]:
      ^selection_merge:
// CHECK-NEXT:         spv.mlir.merge
        spv.mlir.merge
// CHECK-NEXT:       }
      }
// CHECK-NEXT:       %[[LOAD:.+]] = spv.Load "Function" %[[VAR]] : i1
      %load = spv.Load "Function" %var : i1
// CHECK-NEXT:       spv.BranchConditional %[[LOAD]], ^[[CONTINUE:.+]](%[[ARG1]] : i64), ^[[LOOP_MERGE:.+]]
      spv.BranchConditional %load, ^continue(%arg1 : i64), ^loop_merge
// CHECK-NEXT:     ^[[CONTINUE]](%[[ARG2:.+]]: i64):
    ^continue(%arg2: i64):
// CHECK-NEXT:       %[[C0:.+]] = spv.Constant 0 : i64
// CHECK-NEXT:       %[[LT:.+]] = spv.SLessThan %[[ARG2]], %[[C0]] : i64
      %lt = spv.SLessThan %arg2, %cst0_i64 : i64
// CHECK-NEXT:       spv.Store "Function" %[[VAR]], %[[LT]] : i1
      spv.Store "Function" %var, %lt : i1
// CHECK-NEXT:       spv.Branch ^[[LOOP_HEADER]](%[[ARG2]] : i64)
      spv.Branch ^loop_header(%arg2 : i64)
// CHECK-NEXT:     ^[[LOOP_MERGE]]:
    ^loop_merge:
// CHECK-NEXT:       spv.mlir.merge
      spv.mlir.merge
// CHECK-NEXT:     }
    }
// CHECK-NEXT:     spv.Return
    spv.Return
  }
}
