// RUN: mlir-opt %s -split-input-file -pass-pipeline='builtin.module(spirv.module(inline{default-pipeline=''}))' | FileCheck %s

spirv.module Logical GLSL450 {
  spirv.func @callee() "None" {
    spirv.Return
  }

  // CHECK-LABEL: @calling_single_block_ret_func
  spirv.func @calling_single_block_ret_func() "None" {
    // CHECK-NEXT: spirv.Return
    spirv.FunctionCall @callee() : () -> ()
    spirv.Return
  }
}

// -----

spirv.module Logical GLSL450 {
  spirv.func @callee() -> i32 "None" {
    %0 = spirv.Constant 42 : i32
    spirv.ReturnValue %0 : i32
  }

  // CHECK-LABEL: @calling_single_block_retval_func
  spirv.func @calling_single_block_retval_func() -> i32 "None" {
    // CHECK-NEXT: %[[CST:.*]] = spirv.Constant 42
    %0 = spirv.FunctionCall @callee() : () -> (i32)
    // CHECK-NEXT: spirv.ReturnValue %[[CST]]
    spirv.ReturnValue %0 : i32
  }
}

// -----

spirv.module Logical GLSL450 {
  spirv.GlobalVariable @data bind(0, 0) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32> [0])>, StorageBuffer>
  spirv.func @callee() "None" {
    %0 = spirv.mlir.addressof @data : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32> [0])>, StorageBuffer>
    %1 = spirv.Constant 0: i32
    %2 = spirv.AccessChain %0[%1, %1] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<i32, StorageBuffer>
    spirv.Branch ^next

  ^next:
    %3 = spirv.Constant 42: i32
    spirv.Store "StorageBuffer" %2, %3 : i32
    spirv.Return
  }

  // CHECK-LABEL: @calling_multi_block_ret_func
  spirv.func @calling_multi_block_ret_func() "None" {
    // CHECK-NEXT:   spirv.mlir.addressof
    // CHECK-NEXT:   spirv.Constant 0
    // CHECK-NEXT:   spirv.AccessChain
    // CHECK-NEXT:   spirv.Branch ^bb1
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT:   spirv.Constant
    // CHECK-NEXT:   spirv.Store
    // CHECK-NEXT:   spirv.Branch ^bb2
    spirv.FunctionCall @callee() : () -> ()
    // CHECK-NEXT: ^bb2:
    // CHECK-NEXT:   spirv.Return
    spirv.Return
  }
}

// TODO: calling_multi_block_retval_func

// -----

spirv.module Logical GLSL450 {
  spirv.func @callee(%cond : i1) -> () "None" {
    spirv.mlir.selection {
      spirv.BranchConditional %cond, ^then, ^merge
    ^then:
      spirv.Return
    ^merge:
      spirv.mlir.merge
    }
    spirv.Return
  }

  // CHECK-LABEL: @calling_selection_ret_func
  spirv.func @calling_selection_ret_func() "None" {
    %0 = spirv.Constant true
    // CHECK: spirv.FunctionCall
    spirv.FunctionCall @callee(%0) : (i1) -> ()
    spirv.Return
  }
}

// -----

spirv.module Logical GLSL450 {
  spirv.func @callee(%cond : i1) -> () "None" {
    spirv.mlir.selection {
      spirv.BranchConditional %cond, ^then, ^merge
    ^then:
      spirv.Branch ^merge
    ^merge:
      spirv.mlir.merge
    }
    spirv.Return
  }

  // CHECK-LABEL: @calling_selection_no_ret_func
  spirv.func @calling_selection_no_ret_func() "None" {
    // CHECK-NEXT: %[[TRUE:.*]] = spirv.Constant true
    %0 = spirv.Constant true
    // CHECK-NEXT: spirv.mlir.selection
    // CHECK-NEXT:   spirv.BranchConditional %[[TRUE]], ^bb1, ^bb2
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT:   spirv.Branch ^bb2
    // CHECK-NEXT: ^bb2:
    // CHECK-NEXT:   spirv.mlir.merge
    spirv.FunctionCall @callee(%0) : (i1) -> ()
    spirv.Return
  }
}

// -----

spirv.module Logical GLSL450 {
  spirv.func @callee(%cond : i1) -> () "None" {
    spirv.mlir.loop {
      spirv.Branch ^header
    ^header:
      spirv.BranchConditional %cond, ^body, ^merge
    ^body:
      spirv.Return
    ^continue:
      spirv.Branch ^header
    ^merge:
      spirv.mlir.merge
    }
    spirv.Return
  }

  // CHECK-LABEL: @calling_loop_ret_func
  spirv.func @calling_loop_ret_func() "None" {
    %0 = spirv.Constant true
    // CHECK: spirv.FunctionCall
    spirv.FunctionCall @callee(%0) : (i1) -> ()
    spirv.Return
  }
}

// -----

spirv.module Logical GLSL450 {
  spirv.func @callee(%cond : i1) -> () "None" {
    spirv.mlir.loop {
      spirv.Branch ^header
    ^header:
      spirv.BranchConditional %cond, ^body, ^merge
    ^body:
      spirv.Branch ^continue
    ^continue:
      spirv.Branch ^header
    ^merge:
      spirv.mlir.merge
    }
    spirv.Return
  }

  // CHECK-LABEL: @calling_loop_no_ret_func
  spirv.func @calling_loop_no_ret_func() "None" {
    // CHECK-NEXT: %[[TRUE:.*]] = spirv.Constant true
    %0 = spirv.Constant true
    // CHECK-NEXT: spirv.mlir.loop
    // CHECK-NEXT:   spirv.Branch ^bb1
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT:   spirv.BranchConditional %[[TRUE]], ^bb2, ^bb4
    // CHECK-NEXT: ^bb2:
    // CHECK-NEXT:   spirv.Branch ^bb3
    // CHECK-NEXT: ^bb3:
    // CHECK-NEXT:   spirv.Branch ^bb1
    // CHECK-NEXT: ^bb4:
    // CHECK-NEXT:   spirv.mlir.merge
    spirv.FunctionCall @callee(%0) : (i1) -> ()
    spirv.Return
  }
}

// -----

spirv.module Logical GLSL450 {
  spirv.GlobalVariable @arg_0 bind(0, 0) : !spirv.ptr<!spirv.struct<(i32 [0])>, StorageBuffer>
  spirv.GlobalVariable @arg_1 bind(0, 1) : !spirv.ptr<!spirv.struct<(i32 [0])>, StorageBuffer>

  // CHECK: @inline_into_selection_region
  spirv.func @inline_into_selection_region() "None" {
    %1 = spirv.Constant 0 : i32
    // CHECK-DAG: [[ADDRESS_ARG0:%.*]] = spirv.mlir.addressof @arg_0
    // CHECK-DAG: [[ADDRESS_ARG1:%.*]] = spirv.mlir.addressof @arg_1
    // CHECK-DAG: [[LOADPTR:%.*]] = spirv.AccessChain [[ADDRESS_ARG0]]
    // CHECK: [[VAL:%.*]] = spirv.Load "StorageBuffer" [[LOADPTR]]
    %2 = spirv.mlir.addressof @arg_0 : !spirv.ptr<!spirv.struct<(i32 [0])>, StorageBuffer>
    %3 = spirv.mlir.addressof @arg_1 : !spirv.ptr<!spirv.struct<(i32 [0])>, StorageBuffer>
    %4 = spirv.AccessChain %2[%1] : !spirv.ptr<!spirv.struct<(i32 [0])>, StorageBuffer>, i32 -> !spirv.ptr<i32, StorageBuffer>
    %5 = spirv.Load "StorageBuffer" %4 : i32
    %6 = spirv.SGreaterThan %5, %1 : i32
    // CHECK: spirv.mlir.selection
    spirv.mlir.selection {
      spirv.BranchConditional %6, ^bb1, ^bb2
    ^bb1: // pred: ^bb0
      // CHECK: [[STOREPTR:%.*]] = spirv.AccessChain [[ADDRESS_ARG1]]
      %7 = spirv.AccessChain %3[%1] : !spirv.ptr<!spirv.struct<(i32 [0])>, StorageBuffer>, i32 -> !spirv.ptr<i32, StorageBuffer>
      // CHECK-NOT: spirv.FunctionCall
      // CHECK: spirv.AtomicIAdd <Device> <AcquireRelease> [[STOREPTR]], [[VAL]]
      // CHECK: spirv.Branch
      spirv.FunctionCall @atomic_add(%5, %7) : (i32, !spirv.ptr<i32, StorageBuffer>) -> ()
      spirv.Branch ^bb2
    ^bb2 : // 2 preds: ^bb0, ^bb1
      spirv.mlir.merge
    }
    // CHECK: spirv.Return
    spirv.Return
  }
  spirv.func @atomic_add(%arg0: i32, %arg1: !spirv.ptr<i32, StorageBuffer>) "None" {
    %0 = spirv.AtomicIAdd <Device> <AcquireRelease> %arg1, %arg0 : !spirv.ptr<i32, StorageBuffer>
    spirv.Return
  }
  spirv.EntryPoint "GLCompute" @inline_into_selection_region
  spirv.ExecutionMode @inline_into_selection_region "LocalSize", 32, 1, 1
}

// -----

spirv.module Logical GLSL450 {
  // CHECK-LABEL: @foo
  spirv.func @foo(%arg0: i32) -> i32 "None" {
    // CHECK-NOT: spirv.FunctionCall
    // CHECK-NEXT: spirv.Constant 1
    %res = spirv.FunctionCall @bar(%arg0) : (i32) -> i32
    spirv.ReturnValue %res : i32
  }

  spirv.func @bar(%arg1: i32) -> i32 "None" attributes {sym_visibility = "private"} {
    %cst1_i32 = spirv.Constant 1 : i32
    %0 = spirv.IEqual %arg1, %cst1_i32 : i32
    spirv.BranchConditional %0, ^bb1(%cst1_i32 : i32), ^bb2
  ^bb1(%1: i32):
    spirv.ReturnValue %1 : i32
  ^bb2:
    spirv.ReturnValue %cst1_i32 : i32
  }
}

// TODO: Add tests for inlining structured control flow into
// structured control flow.
