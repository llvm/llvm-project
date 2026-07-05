// RUN: mlir-opt -split-input-file -pass-pipeline="builtin.module(mlprogram-pipeline-globals)" --allow-unregistered-dialect %s

// CHECK-LABEL: @global_variable
ml_program.global private mutable @global_variable(dense<4> : tensor<4xi32>) : tensor<4xi32>

// CHECK-LABEL: @global_double_load
func.func @global_double_load() {
  // CHECK: %[[LOAD:.+]] = ml_program.global_load @global_variable
  // CHECK-NOT: ml_program.global_load @global_variable
  %0 = ml_program.global_load @global_variable : tensor<4xi32>
  %1 = ml_program.global_load @global_variable : tensor<4xi32>

  // CHECK: %[[DUMMY:.+]] = "unregistered.dummy"(%[[LOAD]], %[[LOAD]])
  %2 = "unregistered.dummy"(%0, %1) : (tensor<4xi32>, tensor<4xi32>) -> (tensor<4xi32>)

  // CHECK: ml_program.global_store @global_variable %[[DUMMY]]
  ml_program.global_store @global_variable = %2 : tensor<4xi32>
  func.return
}

// -----

// CHECK-LABEL: @global_variable
ml_program.global private mutable @global_variable(dense<4> : tensor<4xi32>) : tensor<4xi32>

// CHECK-LABEL: @global_double_store
func.func @global_double_store() {
  // CHECK: %[[LOAD:.+]] = ml_program.global_load @global_variable
  %0 = ml_program.global_load @global_variable : tensor<4xi32>

  // CHECK: %[[DUMMY:.+]] = "unregistered.dummy"(%[[LOAD]])
  %1 = "unregistered.dummy"(%0) : (tensor<4xi32>) -> (tensor<4xi32>)

  // CHECK: ml_program.global_store @global_variable %[[DUMMY]]
  ml_program.global_store @global_variable = %1 : tensor<4xi32>

  // CHECK-NOT: ml_program.global_store
  ml_program.global_store @global_variable = %1 : tensor<4xi32>
  func.return
}

// -----

// CHECK-LABEL: @global_variable
ml_program.global private mutable @global_variable(dense<4> : tensor<4xi32>) : tensor<4xi32>

// CHECK-LABEL: @global_store_load
func.func @global_store_load() {
  // CHECK: %[[LOAD:.+]] = ml_program.global_load @global_variable
  %0 = ml_program.global_load @global_variable : tensor<4xi32>

  // CHECK: %[[DUMMY:.+]] = "unregistered.dummy"(%[[LOAD]])
  // CHECK: %[[DUMMY2:.+]] = "unregistered.dummy"(%[[DUMMY2]])
  %1 = "unregistered.dummy"(%0) : (tensor<4xi32>) -> (tensor<4xi32>)
  ml_program.global_store @global_variable = %1 : tensor<4xi32>
  %2 = ml_program.global_load @global_variable : tensor<4xi32>
  %3 = "unregistered.dummy"(%2) : (tensor<4xi32>) -> (tensor<4xi32>)

  // CHECK: ml_program.global_store @global_variable %[[DUMMY2]]
  ml_program.global_store @global_variable = %3 : tensor<4xi32>
  func.return
}

// -----

// CHECK-LABEL: @global_variable
ml_program.global private mutable @global_variable(dense<4> : tensor<4xi32>) : tensor<4xi32>

// CHECK-LABEL: @global_store_load_region
func.func @global_store_load_region() {
  // CHECK: %[[LOAD:.+]] = ml_program.global_load @global_variable
  %0 = ml_program.global_load @global_variable : tensor<4xi32>

  // CHECK: %[[DUMMY:.+]] = "unregistered.dummy"(%[[LOAD]])
  %1 = "unregistered.dummy"(%0) : (tensor<4xi32>) -> (tensor<4xi32>)

  // CHECK: ml_program.global_store @global_variable %[[DUMMY]]
  ml_program.global_store @global_variable = %1 : tensor<4xi32>

  // CHECK: "unregistered.dummy2"
  "unregistered.dummy2"() ({
    ^bb():
    %cst = arith.constant dense<0> : tensor<4xi32>
    // CHECK: ml_program.global_store @global_variable
    ml_program.global_store @global_variable = %cst : tensor<4xi32>
    "unregistered.terminator"() : () -> ()
  }) : () -> ()

  // CHECK: %[[LOAD:.+]] ml_program.global_load @global_variable
  %2 = ml_program.global_load @global_variable : tensor<4xi32>

  // CHECK: %[[DUMMY2:.+]] = "unregistered.dummy"(%[[LOAD]])
  %3 = "unregistered.dummy"(%2) : (tensor<4xi32>) -> (tensor<4xi32>)

  // CHECK: ml_program.global_store @global_variable %[[DUMMY2]]
  ml_program.global_store @global_variable = %3 : tensor<4xi32>
  func.return
}

// -----

// CHECK-LABEL: @global_variable
ml_program.global private mutable @global_variable(dense<4> : tensor<4xi32>) : tensor<4xi32>

// CHECK-LABEL: @interrupt
func.func @interrupt() {
  %cst = arith.constant dense<0> : tensor<4xi32>
  // CHECK: ml_program.global_store
  ml_program.global_store @global_variable = %cst : tensor<4xi32>
  func.return
}

// CHECK-LABEL: @call_global_store
func.func @call_global_store() {
  // CHECK: %[[LOAD:.+]] = ml_program.global_load @global_variable
  %0 = ml_program.global_load @global_variable : tensor<4xi32>

  // CHECK: %[[DUMMY:.+]] = "unregistered.dummy"(%[[LOAD]])
  %1 = "unregistered.dummy"(%0) : (tensor<4xi32>) -> (tensor<4xi32>)

  // CHECK: ml_program.global_store @global_variable %[[DUMMY]]
  ml_program.global_store @global_variable = %1 : tensor<4xi32>
  call @interrupt() : () -> ()

  // CHECK: %[[LOAD:.+]] ml_program.global_load @global_variable
  %2 = ml_program.global_load @global_variable : tensor<4xi32>

  // CHECK: %[[DUMMY:.+]] = "unregistered.dummy"(%[[LOAD]])
  %3 = "unregistered.dummy"(%2) : (tensor<4xi32>) -> (tensor<4xi32>)

  // CHECK: ml_program.global_store @global_variable %[[DUMMY]]
  ml_program.global_store @global_variable = %3 : tensor<4xi32>
  func.return
}


// -----

// CHECK-LABEL: @global_variable
ml_program.global private mutable @global_variable(dense<4> : tensor<4xi32>) : tensor<4xi32>

// CHECK-LABEL: @interrupt_indirect
func.func @interrupt_indirect() {
  %cst = arith.constant dense<0> : tensor<4xi32>
  // CHECK: ml_program.global_store
  ml_program.global_store @global_variable = %cst : tensor<4xi32>
  func.return
}

// CHECK-LABEL: @interrupt
func.func @interrupt() {
  call @interrupt_indirect() : () -> ()
  func.return
}

// CHECK-LABEL: @call_indirect_store
func.func @call_indirect_store() {
  // CHECK: %[[LOAD:.+]] = ml_program.global_load @global_variable
  %0 = ml_program.global_load @global_variable : tensor<4xi32>

  // CHECK: %[[DUMMY:.+]] = "unregistered.dummy"(%[[LOAD]])
  %1 = "unregistered.dummy"(%0) : (tensor<4xi32>) -> (tensor<4xi32>)

  // CHECK: ml_program.global_store @global_variable %[[DUMMY]]
  ml_program.global_store @global_variable = %1 : tensor<4xi32>
  call @interrupt() : () -> ()

  // CHECK: %[[LOAD:.+]] ml_program.global_load @global_variable
  %2 = ml_program.global_load @global_variable : tensor<4xi32>

  // CHECK: %[[DUMMY:.+]] = "unregistered.dummy"(%[[LOAD]])
  %3 = "unregistered.dummy"(%2) : (tensor<4xi32>) -> (tensor<4xi32>)

  // CHECK: ml_program.global_store @global_variable %[[DUMMY]]
  ml_program.global_store @global_variable = %3 : tensor<4xi32>
  func.return
}


// -----

// CHECK-LABEL: @global_variable
ml_program.global private mutable @global_variable(dense<4> : tensor<4xi32>) : tensor<4xi32>

// CHECK-LABEL: @interrupt_indirect
func.func @interrupt_indirect() -> tensor<4xi32> {
  // CHECK: ml_program.global_load
  %0 = ml_program.global_load @global_variable : tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// CHECK-LABEL: @interrupt
func.func @interrupt() {
  %0 = call @interrupt_indirect() : () -> (tensor<4xi32>)
  "unregistered.dummy"(%0) : (tensor<4xi32>) -> ()
  func.return
}

// CHECK-LABEL: @call_indirect_load
func.func @call_indirect_load() {
  // CHECK: %[[LOAD:.+]] = ml_program.global_load @global_variable
  %0 = ml_program.global_load @global_variable : tensor<4xi32>

  // CHECK: %[[DUMMY:.+]] = "unregistered.dummy"(%[[LOAD]])
  %1 = "unregistered.dummy"(%0) : (tensor<4xi32>) -> (tensor<4xi32>)

  // CHECK: ml_program.global_store @global_variable %[[DUMMY]]
  ml_program.global_store @global_variable = %1 : tensor<4xi32>
  call @interrupt() : () -> ()

  // CHECK: %[[DUMMY:.+]] = "unregistered.dummy"(%[[LOAD]])
  %2 = ml_program.global_load @global_variable : tensor<4xi32>
  %3 = "unregistered.dummy"(%2) : (tensor<4xi32>) -> (tensor<4xi32>)

  // CHECK: ml_program.global_store @global_variable %[[DUMMY]]
  ml_program.global_store @global_variable = %3 : tensor<4xi32>
  func.return
}

// -----

// Calling via a CallOpInterface op whose callee symbol is not defined in this
// module should not crash - the pass should bail out gracefully.
// See https://github.com/llvm/llvm-project/issues/109649

// CHECK-LABEL: @global_variable
ml_program.global private mutable @global_variable(dense<4> : tensor<4xi32>) : tensor<4xi32>

// CHECK-LABEL: @call_with_unresolvable_callee
func.func @call_with_unresolvable_callee(%arg0: memref<f32>) {
  // Both loads must be preserved; the pass conservatively bails out when it
  // encounters a call whose callee symbol cannot be resolved.
  // CHECK: ml_program.global_load @global_variable
  %0 = ml_program.global_load @global_variable : tensor<4xi32>
  // @callee is not defined anywhere in this module.
  test.call_and_store @callee(%arg0), %arg0 {store_before_call = false} : (memref<f32>, memref<f32>) -> ()
  // CHECK: ml_program.global_load @global_variable
  %1 = ml_program.global_load @global_variable : tensor<4xi32>
  func.return
}

// -----

// CHECK-LABEL: @global_variable
ml_program.global private mutable @global_variable(dense<4> : tensor<4xi32>) : tensor<4xi32>

// CHECK-LABEL: @call_recursive
func.func @call_recursive() {
  // CHECK: %[[LOAD:.+]] = ml_program.global_load @global_variable
  %0 = ml_program.global_load @global_variable : tensor<4xi32>

  // CHECK: %[[DUMMY:.+]] = "unregistered.dummy"(%[[LOAD]])
  %1 = "unregistered.dummy"(%0) : (tensor<4xi32>) -> (tensor<4xi32>)

  // CHECK: ml_program.global_store @global_variable %[[DUMMY]]
  ml_program.global_store @global_variable = %1 : tensor<4xi32>
  call @call_recursive() : () -> ()

  // CHECK: %[[LOAD:.+]] ml_program.global_load @global_variable
  %2 = ml_program.global_load @global_variable : tensor<4xi32>

  // CHECK: %[[DUMMY:.+]] = "unregistered.dummy"(%[[LOAD]])
  %3 = "unregistered.dummy"(%2) : (tensor<4xi32>) -> (tensor<4xi32>)

  // CHECK: ml_program.global_store @global_variable %[[DUMMY]]
  ml_program.global_store @global_variable = %3 : tensor<4xi32>
  func.return
}
