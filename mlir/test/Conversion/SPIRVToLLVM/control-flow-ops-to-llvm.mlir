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
    %0 = spv.Constant 0 : i32
    %1 = spv.Constant true
    // CHECK: llvm.br ^bb1(%{{.*}}, %{{.*}} : i32, i1)
    spv.Branch ^label(%0, %1: i32, i1)
  // CHECK: ^bb1(%{{.*}}: i32, %{{.*}}: i1)
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
    // CHECK: %[[COND:.*]] = llvm.mlir.constant(true) : i1
    %cond = spv.Constant true
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
    // CHECK: %[[COND1:.*]] = llvm.mlir.constant(true) : i1
    %cond = spv.Constant true
    %0 = spv.Constant 0 : i32
    // CHECK: %[[COND2:.*]] = llvm.mlir.constant(false) : i1
    %false = spv.Constant false
    // CHECK: llvm.cond_br %[[COND1]], ^bb1(%{{.*}}, %[[COND2]] : i32, i1), ^bb2
    spv.BranchConditional %cond, ^outer_true(%0, %false: i32, i1), ^outer_false
  // CHECK: ^bb1(%{{.*}}: i32, %[[COND:.*]]: i1):
  ^outer_true(%arg0: i32, %arg1: i1):
    // CHECK: llvm.cond_br %[[COND]], ^bb3, ^bb4(%{{.*}}, %{{.*}} : i32, i32)
    spv.BranchConditional %arg1, ^inner_true, ^inner_false(%arg0, %arg0: i32, i32)
  // CHECK: ^bb2:
  ^outer_false:
    spv.Return
  // CHECK: ^bb3:
  ^inner_true:
    spv.Return
  // CHECK: ^bb4(%{{.*}}: i32, %{{.*}}: i32):
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
// spv.mlir.loop
//===----------------------------------------------------------------------===//

spv.module Logical GLSL450 {
  // CHECK-LABEL: @infinite_loop
  spv.func @infinite_loop(%count : i32) -> () "None" {
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
    spv.mlir.loop {
      spv.Branch ^header
    ^header:
      %cond = spv.Constant true
      spv.BranchConditional %cond, ^body, ^merge
    ^body:
      // Do nothing
      spv.Branch ^continue
    ^continue:
      // Do nothing
      spv.Branch ^header
    ^merge:
      spv.mlir.merge
    }
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
    %cond = spv.Constant true
    // CHECK: llvm.return
    spv.selection {
      spv.BranchConditional %cond, ^merge, ^merge
    ^merge:
      spv.mlir.merge
    }
    spv.Return
  }

  spv.func @selection_with_true_block_only() -> () "None" {
    // CHECK: %[[COND:.*]] = llvm.mlir.constant(true) : i1
    %cond = spv.Constant true
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
      spv.mlir.merge
    }
    // CHECK: ^bb3:
    // CHECK-NEXT: llvm.return
    spv.Return
  }

  spv.func @selection_with_both_true_and_false_block() -> () "None" {
    // CHECK: %[[COND:.*]] = llvm.mlir.constant(true) : i1
    %cond = spv.Constant true
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
      spv.mlir.merge
    }
    // CHECK: ^bb4:
    // CHECK-NEXT: llvm.return
    spv.Return
  }

  spv.func @selection_with_early_return(%arg0: i1) -> i32 "None" {
    // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
    %0 = spv.Constant 0 : i32
    // CHECK: llvm.cond_br %{{.*}}, ^bb1(%[[ZERO]] : i32), ^bb2
    spv.selection {
      spv.BranchConditional %arg0, ^true(%0 : i32), ^merge
    // CHECK: ^bb1(%[[ARG:.*]]: i32):
    ^true(%arg1: i32):
      // CHECK: llvm.return %[[ARG]] : i32
      spv.ReturnValue %arg1 : i32
    // CHECK: ^bb2:
    ^merge:
      // CHECK: llvm.br ^bb3
      spv.mlir.merge
    }
    // CHECK: ^bb3:
    %one = spv.Constant 1 : i32
    spv.ReturnValue %one : i32
  }
}
