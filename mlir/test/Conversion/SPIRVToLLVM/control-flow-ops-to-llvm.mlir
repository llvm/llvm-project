// RUN: mlir-opt -convert-spirv-to-llvm -split-input-file -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spirv.Branch
//===----------------------------------------------------------------------===//

spirv.module Logical GLSL450 {
  spirv.func @branch_without_arguments() -> () "None" {
	  // CHECK: llvm.br ^bb1
    spirv.Branch ^label
  // CHECK: ^bb1
  ^label:
    spirv.Return
  }

  spirv.func @branch_with_arguments() -> () "None" {
    %0 = spirv.Constant 0 : i32
    %1 = spirv.Constant true
    // CHECK: llvm.br ^bb1(%{{.*}}, %{{.*}} : i32, i1)
    spirv.Branch ^label(%0, %1: i32, i1)
  // CHECK: ^bb1(%{{.*}}: i32, %{{.*}}: i1)
  ^label(%arg0: i32, %arg1: i1):
    spirv.Return
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.BranchConditional
//===----------------------------------------------------------------------===//

spirv.module Logical GLSL450 {
  spirv.func @cond_branch_without_arguments() -> () "None" {
    // CHECK: %[[COND:.*]] = llvm.mlir.constant(true) : i1
    %cond = spirv.Constant true
    // CHECK: lvm.cond_br %[[COND]], ^bb1, ^bb2
    spirv.BranchConditional %cond, ^true, ^false
    // CHECK: ^bb1:
  ^true:
    spirv.Return
    // CHECK: ^bb2:
  ^false:
    spirv.Return
  }

  spirv.func @cond_branch_with_arguments_nested() -> () "None" {
    // CHECK: %[[COND1:.*]] = llvm.mlir.constant(true) : i1
    %cond = spirv.Constant true
    %0 = spirv.Constant 0 : i32
    // CHECK: %[[COND2:.*]] = llvm.mlir.constant(false) : i1
    %false = spirv.Constant false
    // CHECK: llvm.cond_br %[[COND1]], ^bb1(%{{.*}}, %[[COND2]] : i32, i1), ^bb2
    spirv.BranchConditional %cond, ^outer_true(%0, %false: i32, i1), ^outer_false
  // CHECK: ^bb1(%{{.*}}: i32, %[[COND:.*]]: i1):
  ^outer_true(%arg0: i32, %arg1: i1):
    // CHECK: llvm.cond_br %[[COND]], ^bb3, ^bb4(%{{.*}}, %{{.*}} : i32, i32)
    spirv.BranchConditional %arg1, ^inner_true, ^inner_false(%arg0, %arg0: i32, i32)
  // CHECK: ^bb2:
  ^outer_false:
    spirv.Return
  // CHECK: ^bb3:
  ^inner_true:
    spirv.Return
  // CHECK: ^bb4(%{{.*}}: i32, %{{.*}}: i32):
  ^inner_false(%arg3: i32, %arg4: i32):
    spirv.Return
  }

  spirv.func @cond_branch_with_weights(%cond: i1) -> () "None" {
    // CHECK: llvm.cond_br %{{.*}} weights(dense<[1, 2]> : vector<2xi32>), ^bb1, ^bb2
    spirv.BranchConditional %cond [1, 2], ^true, ^false
  // CHECK: ^bb1:
  ^true:
    spirv.Return
  // CHECK: ^bb2:
  ^false:
    spirv.Return
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.mlir.loop
//===----------------------------------------------------------------------===//

spirv.module Logical GLSL450 {
  // CHECK-LABEL: @infinite_loop
  spirv.func @infinite_loop(%count : i32) -> () "None" {
    // CHECK:   llvm.br ^[[BB1:.*]]
    // CHECK: ^[[BB1]]:
    // CHECK:   %[[COND:.*]] = llvm.mlir.constant(true) : i1
    // CHECK:   llvm.cond_br %[[COND]], ^[[BB2:.*]], ^[[BB4:.*]]
    // CHECK: ^[[BB2]]:
    // CHECK:   llvm.br ^[[BB3:.*]]
    // CHECK: ^[[BB3]]:
    // CHECK:   llvm.br ^[[BB1:.*]]
    // CHECK: ^[[BB4]]:
    // CHECK:   llvm.br ^[[BB5:.*]]
    // CHECK: ^[[BB5]]:
    // CHECK:   llvm.return
    spirv.mlir.loop {
      spirv.Branch ^header
    ^header:
      %cond = spirv.Constant true
      spirv.BranchConditional %cond, ^body, ^merge
    ^body:
      // Do nothing
      spirv.Branch ^continue
    ^continue:
      // Do nothing
      spirv.Branch ^header
    ^merge:
      spirv.mlir.merge
    }
    spirv.Return
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.mlir.selection
//===----------------------------------------------------------------------===//

spirv.module Logical GLSL450 {
  spirv.func @selection_empty() -> () "None" {
    // CHECK: llvm.return
    spirv.mlir.selection {
    }
    spirv.Return
  }

  spirv.func @selection_with_merge_block_only() -> () "None" {
    %cond = spirv.Constant true
    // CHECK: llvm.return
    spirv.mlir.selection {
      spirv.BranchConditional %cond, ^merge, ^merge
    ^merge:
      spirv.mlir.merge
    }
    spirv.Return
  }

  spirv.func @selection_with_true_block_only() -> () "None" {
    // CHECK: %[[COND:.*]] = llvm.mlir.constant(true) : i1
    %cond = spirv.Constant true
    // CHECK: llvm.cond_br %[[COND]], ^bb1, ^bb2
    spirv.mlir.selection {
      spirv.BranchConditional %cond, ^true, ^merge
    // CHECK: ^bb1:
    ^true:
    // CHECK: llvm.br ^bb2
      spirv.Branch ^merge
    // CHECK: ^bb2:
    ^merge:
      // CHECK: llvm.br ^bb3
      spirv.mlir.merge
    }
    // CHECK: ^bb3:
    // CHECK-NEXT: llvm.return
    spirv.Return
  }

  spirv.func @selection_with_both_true_and_false_block() -> () "None" {
    // CHECK: %[[COND:.*]] = llvm.mlir.constant(true) : i1
    %cond = spirv.Constant true
    // CHECK: llvm.cond_br %[[COND]], ^bb1, ^bb2
    spirv.mlir.selection {
      spirv.BranchConditional %cond, ^true, ^false
    // CHECK: ^bb1:
    ^true:
    // CHECK: llvm.br ^bb3
      spirv.Branch ^merge
    // CHECK: ^bb2:
    ^false:
    // CHECK: llvm.br ^bb3
      spirv.Branch ^merge
    // CHECK: ^bb3:
    ^merge:
      // CHECK: llvm.br ^bb4
      spirv.mlir.merge
    }
    // CHECK: ^bb4:
    // CHECK-NEXT: llvm.return
    spirv.Return
  }

  spirv.func @selection_with_early_return(%arg0: i1) -> i32 "None" {
    // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
    %0 = spirv.Constant 0 : i32
    // CHECK: llvm.cond_br %{{.*}}, ^bb1(%[[ZERO]] : i32), ^bb2
    spirv.mlir.selection {
      spirv.BranchConditional %arg0, ^true(%0 : i32), ^merge
    // CHECK: ^bb1(%[[ARG:.*]]: i32):
    ^true(%arg1: i32):
      // CHECK: llvm.return %[[ARG]] : i32
      spirv.ReturnValue %arg1 : i32
    // CHECK: ^bb2:
    ^merge:
      // CHECK: llvm.br ^bb3
      spirv.mlir.merge
    }
    // CHECK: ^bb3:
    %one = spirv.Constant 1 : i32
    spirv.ReturnValue %one : i32
  }
}
