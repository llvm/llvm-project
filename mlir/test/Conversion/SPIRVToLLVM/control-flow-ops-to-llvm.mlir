// RUN: mlir-opt -convert-spirv-to-llvm -split-input-file -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.Branch
//===----------------------------------------------------------------------===//

spv.module Logical GLSL450 {
  spv.func @branch_without_arguments() -> () "None" {
	  // CHECK: llvm.br ^bb1
    spv.Branch ^label
  // CHECK: ^bb1
  ^label:
    spv.Return
  }

  spv.func @branch_with_arguments() -> () "None" {
    %0 = spv.constant 0 : i32
    %1 = spv.constant true
    // CHECK: llvm.br ^bb1(%{{.*}}, %{{.*}} : !llvm.i32, !llvm.i1)
    spv.Branch ^label(%0, %1: i32, i1)
  // CHECK: ^bb1(%{{.*}}: !llvm.i32, %{{.*}}: !llvm.i1)
  ^label(%arg0: i32, %arg1: i1):
    spv.Return
  }
}

// -----

//===----------------------------------------------------------------------===//
// spv.BranchConditional
//===----------------------------------------------------------------------===//

spv.module Logical GLSL450 {
  spv.func @cond_branch_without_arguments() -> () "None" {
    // CHECK: %[[COND:.*]] = llvm.mlir.constant(true) : !llvm.i1
    %cond = spv.constant true
    // CHECK: lvm.cond_br %[[COND]], ^bb1, ^bb2
    spv.BranchConditional %cond, ^true, ^false
    // CHECK: ^bb1:
  ^true:
    spv.Return
    // CHECK: ^bb2:
  ^false:
    spv.Return
  }

  spv.func @cond_branch_with_arguments_nested() -> () "None" {
    // CHECK: %[[COND1:.*]] = llvm.mlir.constant(true) : !llvm.i1
    %cond = spv.constant true
    %0 = spv.constant 0 : i32
    // CHECK: %[[COND2:.*]] = llvm.mlir.constant(false) : !llvm.i1
    %false = spv.constant false
    // CHECK: llvm.cond_br %[[COND1]], ^bb1(%{{.*}}, %[[COND2]] : !llvm.i32, !llvm.i1), ^bb2
    spv.BranchConditional %cond, ^outer_true(%0, %false: i32, i1), ^outer_false
  // CHECK: ^bb1(%{{.*}}: !llvm.i32, %[[COND:.*]]: !llvm.i1):
  ^outer_true(%arg0: i32, %arg1: i1):
    // CHECK: llvm.cond_br %[[COND]], ^bb3, ^bb4(%{{.*}}, %{{.*}} : !llvm.i32, !llvm.i32)
    spv.BranchConditional %arg1, ^inner_true, ^inner_false(%arg0, %arg0: i32, i32)
  // CHECK: ^bb2:
  ^outer_false:
    spv.Return
  // CHECK: ^bb3:
  ^inner_true:
    spv.Return
  // CHECK: ^bb4(%{{.*}}: !llvm.i32, %{{.*}}: !llvm.i32):
  ^inner_false(%arg3: i32, %arg4: i32):
    spv.Return
  }

  spv.func @cond_branch_with_weights(%cond: i1) -> () "None" {
    // CHECK: llvm.cond_br %{{.*}} weights(dense<[1, 2]> : vector<2xi32>), ^bb1, ^bb2
    spv.BranchConditional %cond [1, 2], ^true, ^false
  // CHECK: ^bb1:
  ^true:
    spv.Return
  // CHECK: ^bb2:
  ^false:
    spv.Return
  }
}

// -----

//===----------------------------------------------------------------------===//
// spv.selection
//===----------------------------------------------------------------------===//

spv.module Logical GLSL450 {
  spv.func @selection_empty() -> () "None" {
    // CHECK: llvm.return
    spv.selection {
    }
    spv.Return
  }

  spv.func @selection_with_merge_block_only() -> () "None" {
    %cond = spv.constant true
    // CHECK: llvm.return
    spv.selection {
      spv.BranchConditional %cond, ^merge, ^merge
    ^merge:
      spv._merge
    }
    spv.Return
  }

  spv.func @selection_with_true_block_only() -> () "None" {
    // CHECK: %[[COND:.*]] = llvm.mlir.constant(true) : !llvm.i1
    %cond = spv.constant true
    // CHECK: llvm.cond_br %[[COND]], ^bb1, ^bb2
    spv.selection {
      spv.BranchConditional %cond, ^true, ^merge
    // CHECK: ^bb1:
    ^true:
    // CHECK: llvm.br ^bb2
      spv.Branch ^merge
    // CHECK: ^bb2:
    ^merge:
      // CHECK: llvm.br ^bb3
      spv._merge
    }
    // CHECK: ^bb3:
    // CHECK-NEXT: llvm.return
    spv.Return
  }

  spv.func @selection_with_both_true_and_false_block() -> () "None" {
    // CHECK: %[[COND:.*]] = llvm.mlir.constant(true) : !llvm.i1
    %cond = spv.constant true
    // CHECK: llvm.cond_br %[[COND]], ^bb1, ^bb2
    spv.selection {
      spv.BranchConditional %cond, ^true, ^false
    // CHECK: ^bb1:
    ^true:
    // CHECK: llvm.br ^bb3
      spv.Branch ^merge
    // CHECK: ^bb2:
    ^false:
    // CHECK: llvm.br ^bb3
      spv.Branch ^merge
    // CHECK: ^bb3:
    ^merge:
      // CHECK: llvm.br ^bb4
      spv._merge
    }
    // CHECK: ^bb4:
    // CHECK-NEXT: llvm.return
    spv.Return
  }

  spv.func @selection_with_early_return(%arg0: i1) -> i32 "None" {
    // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    %0 = spv.constant 0 : i32
    // CHECK: llvm.cond_br %{{.*}}, ^bb1(%[[ZERO]] : !llvm.i32), ^bb2
    spv.selection {
      spv.BranchConditional %arg0, ^true(%0 : i32), ^merge
    // CHECK: ^bb1(%[[ARG:.*]]: !llvm.i32):
    ^true(%arg1: i32):
      // CHECK: llvm.return %[[ARG]] : !llvm.i32
      spv.ReturnValue %arg1 : i32
    // CHECK: ^bb2:
    ^merge:
      // CHECK: llvm.br ^bb3
      spv._merge
    }
    // CHECK: ^bb3:
    %one = spv.constant 1 : i32
    spv.ReturnValue %one : i32
  }
}
