// RUN: mlir-opt %s -split-input-file -pass-pipeline='spv.module(inline{default-pipeline=''})' | FileCheck %s

spv.module Logical GLSL450 {
  spv.func @callee() "None" {
    spv.Return
  }

  // CHECK-LABEL: @calling_single_block_ret_func
  spv.func @calling_single_block_ret_func() "None" {
    // CHECK-NEXT: spv.Return
    spv.FunctionCall @callee() : () -> ()
    spv.Return
  }
}

// -----

spv.module Logical GLSL450 {
  spv.func @callee() -> i32 "None" {
    %0 = spv.constant 42 : i32
    spv.ReturnValue %0 : i32
  }

  // CHECK-LABEL: @calling_single_block_retval_func
  spv.func @calling_single_block_retval_func() -> i32 "None" {
    // CHECK-NEXT: %[[CST:.*]] = spv.constant 42
    %0 = spv.FunctionCall @callee() : () -> (i32)
    // CHECK-NEXT: spv.ReturnValue %[[CST]]
    spv.ReturnValue %0 : i32
  }
}

// -----

spv.module Logical GLSL450 {
  spv.globalVariable @data bind(0, 0) : !spv.ptr<!spv.struct<(!spv.rtarray<i32> [0])>, StorageBuffer>
  spv.func @callee() "None" {
    %0 = spv.mlir.addressof @data : !spv.ptr<!spv.struct<(!spv.rtarray<i32> [0])>, StorageBuffer>
    %1 = spv.constant 0: i32
    %2 = spv.AccessChain %0[%1, %1] : !spv.ptr<!spv.struct<(!spv.rtarray<i32> [0])>, StorageBuffer>, i32, i32
    spv.Branch ^next

  ^next:
    %3 = spv.constant 42: i32
    spv.Store "StorageBuffer" %2, %3 : i32
    spv.Return
  }

  // CHECK-LABEL: @calling_multi_block_ret_func
  spv.func @calling_multi_block_ret_func() "None" {
    // CHECK-NEXT:   spv.mlir.addressof
    // CHECK-NEXT:   spv.constant 0
    // CHECK-NEXT:   spv.AccessChain
    // CHECK-NEXT:   spv.Branch ^bb1
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT:   spv.constant
    // CHECK-NEXT:   spv.Store
    // CHECK-NEXT:   spv.Branch ^bb2
    spv.FunctionCall @callee() : () -> ()
    // CHECK-NEXT: ^bb2:
    // CHECK-NEXT:   spv.Return
    spv.Return
  }
}

// TODO: calling_multi_block_retval_func

// -----

spv.module Logical GLSL450 {
  spv.func @callee(%cond : i1) -> () "None" {
    spv.selection {
      spv.BranchConditional %cond, ^then, ^merge
    ^then:
      spv.Return
    ^merge:
      spv.mlir.merge
    }
    spv.Return
  }

  // CHECK-LABEL: @calling_selection_ret_func
  spv.func @calling_selection_ret_func() "None" {
    %0 = spv.constant true
    // CHECK: spv.FunctionCall
    spv.FunctionCall @callee(%0) : (i1) -> ()
    spv.Return
  }
}

// -----

spv.module Logical GLSL450 {
  spv.func @callee(%cond : i1) -> () "None" {
    spv.selection {
      spv.BranchConditional %cond, ^then, ^merge
    ^then:
      spv.Branch ^merge
    ^merge:
      spv.mlir.merge
    }
    spv.Return
  }

  // CHECK-LABEL: @calling_selection_no_ret_func
  spv.func @calling_selection_no_ret_func() "None" {
    // CHECK-NEXT: %[[TRUE:.*]] = spv.constant true
    %0 = spv.constant true
    // CHECK-NEXT: spv.selection
    // CHECK-NEXT:   spv.BranchConditional %[[TRUE]], ^bb1, ^bb2
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT:   spv.Branch ^bb2
    // CHECK-NEXT: ^bb2:
    // CHECK-NEXT:   spv.mlir.merge
    spv.FunctionCall @callee(%0) : (i1) -> ()
    spv.Return
  }
}

// -----

spv.module Logical GLSL450 {
  spv.func @callee(%cond : i1) -> () "None" {
    spv.loop {
      spv.Branch ^header
    ^header:
      spv.BranchConditional %cond, ^body, ^merge
    ^body:
      spv.Return
    ^continue:
      spv.Branch ^header
    ^merge:
      spv.mlir.merge
    }
    spv.Return
  }

  // CHECK-LABEL: @calling_loop_ret_func
  spv.func @calling_loop_ret_func() "None" {
    %0 = spv.constant true
    // CHECK: spv.FunctionCall
    spv.FunctionCall @callee(%0) : (i1) -> ()
    spv.Return
  }
}

// -----

spv.module Logical GLSL450 {
  spv.func @callee(%cond : i1) -> () "None" {
    spv.loop {
      spv.Branch ^header
    ^header:
      spv.BranchConditional %cond, ^body, ^merge
    ^body:
      spv.Branch ^continue
    ^continue:
      spv.Branch ^header
    ^merge:
      spv.mlir.merge
    }
    spv.Return
  }

  // CHECK-LABEL: @calling_loop_no_ret_func
  spv.func @calling_loop_no_ret_func() "None" {
    // CHECK-NEXT: %[[TRUE:.*]] = spv.constant true
    %0 = spv.constant true
    // CHECK-NEXT: spv.loop
    // CHECK-NEXT:   spv.Branch ^bb1
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT:   spv.BranchConditional %[[TRUE]], ^bb2, ^bb4
    // CHECK-NEXT: ^bb2:
    // CHECK-NEXT:   spv.Branch ^bb3
    // CHECK-NEXT: ^bb3:
    // CHECK-NEXT:   spv.Branch ^bb1
    // CHECK-NEXT: ^bb4:
    // CHECK-NEXT:   spv.mlir.merge
    spv.FunctionCall @callee(%0) : (i1) -> ()
    spv.Return
  }
}

// -----

spv.module Logical GLSL450 {
  spv.globalVariable @arg_0 bind(0, 0) : !spv.ptr<!spv.struct<(i32 [0])>, StorageBuffer>
  spv.globalVariable @arg_1 bind(0, 1) : !spv.ptr<!spv.struct<(i32 [0])>, StorageBuffer>

  // CHECK: @inline_into_selection_region
  spv.func @inline_into_selection_region() "None" {
    %1 = spv.constant 0 : i32
    // CHECK-DAG: [[ADDRESS_ARG0:%.*]] = spv.mlir.addressof @arg_0
    // CHECK-DAG: [[ADDRESS_ARG1:%.*]] = spv.mlir.addressof @arg_1
    // CHECK-DAG: [[LOADPTR:%.*]] = spv.AccessChain [[ADDRESS_ARG0]]
    // CHECK: [[VAL:%.*]] = spv.Load "StorageBuffer" [[LOADPTR]]
    %2 = spv.mlir.addressof @arg_0 : !spv.ptr<!spv.struct<(i32 [0])>, StorageBuffer>
    %3 = spv.mlir.addressof @arg_1 : !spv.ptr<!spv.struct<(i32 [0])>, StorageBuffer>
    %4 = spv.AccessChain %2[%1] : !spv.ptr<!spv.struct<(i32 [0])>, StorageBuffer>, i32
    %5 = spv.Load "StorageBuffer" %4 : i32
    %6 = spv.SGreaterThan %5, %1 : i32
    // CHECK: spv.selection
    spv.selection {
      spv.BranchConditional %6, ^bb1, ^bb2
    ^bb1: // pred: ^bb0
      // CHECK: [[STOREPTR:%.*]] = spv.AccessChain [[ADDRESS_ARG1]]
      %7 = spv.AccessChain %3[%1] : !spv.ptr<!spv.struct<(i32 [0])>, StorageBuffer>, i32
      // CHECK-NOT: spv.FunctionCall
      // CHECK: spv.AtomicIAdd "Device" "AcquireRelease" [[STOREPTR]], [[VAL]]
      // CHECK: spv.Branch
      spv.FunctionCall @atomic_add(%5, %7) : (i32, !spv.ptr<i32, StorageBuffer>) -> ()
      spv.Branch ^bb2
    ^bb2 : // 2 preds: ^bb0, ^bb1
      spv.mlir.merge
    }
    // CHECK: spv.Return
    spv.Return
  }
  spv.func @atomic_add(%arg0: i32, %arg1: !spv.ptr<i32, StorageBuffer>) "None" {
    %0 = spv.AtomicIAdd "Device" "AcquireRelease" %arg1, %arg0 : !spv.ptr<i32, StorageBuffer>
    spv.Return
  }
  spv.EntryPoint "GLCompute" @inline_into_selection_region
  spv.ExecutionMode @inline_into_selection_region "LocalSize", 32, 1, 1
}

// TODO: Add tests for inlining structured control flow into
// structured control flow.
