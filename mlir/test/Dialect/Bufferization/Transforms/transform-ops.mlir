// RUN: mlir-opt --test-transform-dialect-interpreter %s -split-input-file -verify-diagnostics | FileCheck %s

// Test One-Shot Bufferize.

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  sequence %arg0 : !pdl.operation failures(propagate) {
    ^bb0(%arg1: !pdl.operation):
      %0 = transform.structured.match ops{["func.func"]} in %arg1
      transform.bufferization.one_shot_bufferize %0
          {target_is_module = false}
  }
}

// CHECK-LABEL: func @test_function(
//  CHECK-SAME:     %[[A:.*]]: tensor<?xf32>
func.func @test_function(%A : tensor<?xf32>, %v : vector<4xf32>) -> (tensor<?xf32>) {
  %c0 = arith.constant 0 : index

  // CHECK: %[[A_memref:.*]] = bufferization.to_memref %[[A]]
  // CHECK: %[[dim:.*]] = memref.dim %[[A_memref]]
  // CHECK: %[[alloc:.*]] = memref.alloc(%[[dim]])
  // CHECK: memref.copy %[[A_memref]], %[[alloc]]
  // CHECK: vector.transfer_write %{{.*}}, %[[alloc]]
  // CHECK: %[[res_tensor:.*]] = bufferization.to_tensor %[[alloc]]
  %0 = vector.transfer_write %v, %A[%c0] : vector<4xf32>, tensor<?xf32>

  // CHECK: memref.dealloc %[[alloc]]
  // CHECK: return %[[res_tensor]]
  return %0 : tensor<?xf32>
}

// -----

// Test analysis of One-Shot Bufferize only.

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  sequence %arg0 : !pdl.operation failures(propagate) {
    ^bb0(%arg1: !pdl.operation):
      %0 = transform.structured.match ops{["func.func"]} in %arg1
      transform.bufferization.one_shot_bufferize %0
          {target_is_module = false, test_analysis_only = true}
  }
}

// CHECK-LABEL: func @test_function_analysis(
//  CHECK-SAME:     %[[A:.*]]: tensor<?xf32>
func.func @test_function_analysis(%A : tensor<?xf32>, %v : vector<4xf32>) -> (tensor<?xf32>) {
  %c0 = arith.constant 0 : index
  // CHECK: vector.transfer_write
  // CHECK-SAME: {__inplace_operands_attr__ = ["none", "false", "none"]}
  // CHECK-SAME: tensor<?xf32>
  %0 = vector.transfer_write %v, %A[%c0] : vector<4xf32>, tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

// Test One-Shot Bufferize transform failure with an unknown op. This would be
// allowed with `allow_unknown_ops`.

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  sequence %arg0 : !pdl.operation failures(propagate) {
    ^bb0(%arg1: !pdl.operation):
      %0 = transform.structured.match ops{["func.func"]} in %arg1
      // expected-error @+1 {{bufferization failed}}
      transform.bufferization.one_shot_bufferize %0 {target_is_module = false}
  }
}

func.func @test_unknown_op_failure() -> (tensor<?xf32>) {
  // expected-error @+1 {{op was not bufferized}}
  %0 = "test.dummy_op"() : () -> (tensor<?xf32>)
  return %0 : tensor<?xf32>
}

// -----

// Test One-Shot Bufferize transform failure with a module op.

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  sequence %arg0 : !pdl.operation failures(propagate) {
    ^bb0(%arg1: !pdl.operation):
      // %arg1 is the module
      transform.bufferization.one_shot_bufferize %arg1
  }
}

module {
  // CHECK-LABEL: func @test_function(
  //  CHECK-SAME:     %[[A:.*]]: tensor<?xf32>
  func.func @test_function(%A : tensor<?xf32>, %v : vector<4xf32>) -> (tensor<?xf32>) {
    %c0 = arith.constant 0 : index

    // CHECK: %[[A_memref:.*]] = bufferization.to_memref %[[A]]
    // CHECK: %[[dim:.*]] = memref.dim %[[A_memref]]
    // CHECK: %[[alloc:.*]] = memref.alloc(%[[dim]])
    // CHECK: memref.copy %[[A_memref]], %[[alloc]]
    // CHECK: vector.transfer_write %{{.*}}, %[[alloc]]
    // CHECK: %[[res_tensor:.*]] = bufferization.to_tensor %[[alloc]]
    %0 = vector.transfer_write %v, %A[%c0] : vector<4xf32>, tensor<?xf32>

    // CHECK: memref.dealloc %[[alloc]]
    // CHECK: return %[[res_tensor]]
    return %0 : tensor<?xf32>
  }
}
