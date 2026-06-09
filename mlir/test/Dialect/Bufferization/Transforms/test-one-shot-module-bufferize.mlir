// RUN: mlir-opt %s -test-one-shot-module-bufferize -split-input-file | FileCheck %s

#enc1 = #test.tensor_encoding<"hello">
#enc2 = #test.tensor_encoding<"not hello">

module @BufferizeEncodingThroughFunctionBoundaryAndCustomOps {
  // CHECK: func @inner_func(
  // CHECK-SAME:  %[[arg0:.*]]: memref<?xf32, #test.memref_layout<"hello">>)
  // CHECK-SAME:  -> memref<?xf32, #test.memref_layout<"hello">>
  func.func @inner_func(%t: tensor<?xf32, #enc1>)
      -> tensor<?xf32, #enc1> {
    // CHECK: return %[[arg0]]
    return %t : tensor<?xf32, #enc1>
  }

  // CHECK: func @outer_func(
  // CHECK-SAME:  %[[arg0:.*]]: memref<?xf32, #test.memref_layout<"hello">>)
  // CHECK-SAME:  -> (memref<?xf32, #test.memref_layout<"hello">>,
  // CHECK-SAME:      memref<?xf32, #test.memref_layout<"not hello">>)
  func.func @outer_func(%t0: tensor<?xf32, #enc1>)
      -> (tensor<?xf32, #enc1>, tensor<?xf32, #enc2>) {
    // CHECK: %[[call:.*]] = call @inner_func(%[[arg0]])
    %0 = call @inner_func(%t0)
      : (tensor<?xf32, #enc1>) -> (tensor<?xf32, #enc1>)

    // CHECK: %[[local:.*]] = "test.create_memref_op"() : ()
    // CHECK-SAME:  -> memref<?xf32, #test.memref_layout<"not hello">>
    %local = "test.create_tensor_op"() : () -> tensor<?xf32, #enc2>
    // CHECK: %[[dummy:.*]] = "test.dummy_memref_op"(%[[local]])
    %1 = "test.dummy_tensor_op"(%local) : (tensor<?xf32, #enc2>)
      -> tensor<?xf32, #enc2>

    // CHECK: return %[[call]], %[[dummy]]
    return %0, %1 : tensor<?xf32, #enc1>, tensor<?xf32, #enc2>
  }
}

// -----

// CHECK:   func.func @custom_types(
// CHECK-SAME:    %[[arg:.*]]: !test.test_memref<[4, 4], f64>
// CHECK-SAME:  ) -> (!test.test_memref<[4, 8], f64>,
// CHECK-SAME:        !test.test_memref<[4, 8], f64>)
func.func @custom_types(%arg: !test.test_tensor<[4, 4], f64>)
    -> (!test.test_tensor<[4, 8], f64>, !test.test_tensor<[4, 8], f64>) {
  // CHECK: %[[out1:.*]] = "test.dummy_memref_op"(%[[arg]]) :
  // CHECK-SAME: (!test.test_memref<[4, 4], f64>) -> !test.test_memref<[4, 8], f64>
  %out1 = "test.dummy_tensor_op"(%arg) : (!test.test_tensor<[4, 4], f64>)
    -> !test.test_tensor<[4, 8], f64>

  // CHECK: %[[alloc:.*]] = "test.create_memref_op"
  // CHECK: %[[out2:.*]] = "test.dummy_memref_op"(%[[alloc]])
  // CHECK-SAME: (!test.test_memref<[4, 4], f64>) -> !test.test_memref<[4, 8], f64>
  %alloc = "test.create_tensor_op"() : () -> !test.test_tensor<[4, 4], f64>
  %out2 = "test.dummy_tensor_op"(%alloc) : (!test.test_tensor<[4, 4], f64>)
    -> !test.test_tensor<[4, 8], f64>

  // CHECK: return %[[out1]], %[[out2]]
  return %out1, %out2 :
    !test.test_tensor<[4, 8], f64>, !test.test_tensor<[4, 8], f64>
}

// -----

// CHECK:   func.func @custom_types_foo(
// CHECK-SAME:    %[[arg:.*]]: !test.test_memref<[4, 4], f64>
// CHECK-SAME:  ) -> !test.test_memref<[4, 4], f64>
func.func @custom_types_foo(%arg: !test.test_tensor<[4, 4], f64>)
    -> !test.test_tensor<[4, 4], f64> {
  // CHECK: %[[out:.*]] = "test.dummy_memref_op"(%[[arg]])
  %out = "test.dummy_tensor_op"(%arg) : (!test.test_tensor<[4, 4], f64>)
    -> !test.test_tensor<[4, 4], f64>
  // CHECK: return %[[out]]
  return %out : !test.test_tensor<[4, 4], f64>
}

// CHECK:   func.func @custom_types_bar(
// CHECK-SAME:    %[[arg:.*]]: !test.test_memref<[4, 4], f64>
// CHECK-SAME:  ) -> !test.test_memref<[4, 8], f64>
func.func @custom_types_bar(%arg: !test.test_tensor<[4, 4], f64>)
    -> !test.test_tensor<[4, 8], f64> {
  // CHECK: %[[call:.*]] = call @custom_types_foo(%[[arg]])
  %call = func.call @custom_types_foo(%arg) : (!test.test_tensor<[4, 4], f64>)
    -> !test.test_tensor<[4, 4], f64>

  // CHECK: %[[out:.*]] = "test.dummy_memref_op"(%[[call]])
  %out = "test.dummy_tensor_op"(%call) : (!test.test_tensor<[4, 4], f64>)
    -> !test.test_tensor<[4, 8], f64>

  // CHECK: return %[[out]]
  return %out : !test.test_tensor<[4, 8], f64>
}


// -----

// CHECK:   func.func @custom_types_scf_for_inplace(
// CHECK-SAME:    %[[arg:.+]]: !test.test_memref<[4, 4], f64>,
// CHECK-SAME:    %[[lb:.+]]: index, %[[ub:.+]]: index, %[[step:.+]]: index
// CHECK-SAME:  ) -> !test.test_memref<[4, 4], f64>
func.func @custom_types_scf_for_inplace(
    %arg: !test.test_tensor<[4, 4], f64>,
    %lb: index, %ub: index, %step: index)
    -> !test.test_tensor<[4, 4], f64> {
  // CHECK: %[[loop:.+]] = scf.for %{{.*}} = %[[lb]] to %[[ub]] step %[[step]]
  // CHECK-SAME: iter_args(%[[iter:.+]] = %[[arg]]) -> (!test.test_memref<[4, 4], f64>) {
  // CHECK: %[[call:.+]] = "test.dummy_memref_op"(%[[iter]])
  // CHECK: scf.yield %[[call]] : !test.test_memref<[4, 4], f64>
  %loop = scf.for %i = %lb to %ub step %step
      iter_args(%iter = %arg) -> (!test.test_tensor<[4, 4], f64>) {
    // Inside loop: use iter_args directly (this is inplace modifiable op)
    %call = "test.dummy_tensor_op"(%iter) : (!test.test_tensor<[4, 4], f64>)
      -> !test.test_tensor<[4, 4], f64>
    // Yield: return the same iter_args value (or result of inplace op on it)
    scf.yield %call : !test.test_tensor<[4, 4], f64>
  }

  // CHECK: return %[[loop]] : !test.test_memref<[4, 4], f64>
  return %loop : !test.test_tensor<[4, 4], f64>
}

// -----

func.func private @custom_types_identity_2d(%arg: !test.test_tensor<[4, 4], f64>)
    -> !test.test_tensor<[4, 4], f64> {
  %out = "test.dummy_tensor_op"(%arg) : (!test.test_tensor<[4, 4], f64>)
    -> !test.test_tensor<[4, 4], f64>
  return %out : !test.test_tensor<[4, 4], f64>
}

// Same as @custom_types_scf_for_inplace, but with an inner call to test alias analysis
// through function boundaries.
// CHECK-LABEL: func.func @custom_types_scf_for_inplace_with_call(
// CHECK-SAME: %[[arg:.+]]: !test.test_memref<[4, 4], f64>
// CHECK-SAME: %[[lb:.+]]: index, %[[ub:.+]]: index, %[[step:.+]]: index
// CHECK-SAME: ) -> !test.test_memref<[4, 4], f64>
// CHECK: %[[loop:.+]] = scf.for %{{.*}} = %[[lb]] to %[[ub]] step %[[step]] iter_args(%[[iter:.+]] = %[[arg]]) -> (!test.test_memref<[4, 4], f64>) {
// CHECK: %[[call:.+]] = func.call @custom_types_identity_2d(%[[iter]]) : (!test.test_memref<[4, 4], f64>) -> !test.test_memref<[4, 4], f64>
// CHECK: scf.yield %[[call]] : !test.test_memref<[4, 4], f64>
// CHECK: return %[[loop]] : !test.test_memref<[4, 4], f64>
func.func @custom_types_scf_for_inplace_with_call(
    %arg: !test.test_tensor<[4, 4], f64>,
    %lb: index, %ub: index, %step: index)
    -> !test.test_tensor<[4, 4], f64> {
  %loop = scf.for %i = %lb to %ub step %step
      iter_args(%iter = %arg) -> (!test.test_tensor<[4, 4], f64>) {
    %call = func.call @custom_types_identity_2d(%iter)
      : (!test.test_tensor<[4, 4], f64>) -> !test.test_tensor<[4, 4], f64>
    scf.yield %call : !test.test_tensor<[4, 4], f64>
  }

  return %loop : !test.test_tensor<[4, 4], f64>
}

// -----

// CHECK-LABEL: func.func @custom_types_scf_if_inplace(
// CHECK-SAME: %[[arg:.+]]: !test.test_memref<[4, 4], f64>
// CHECK-SAME: %[[cond:.+]]: i1
// CHECK-SAME: ) -> !test.test_memref<[4, 4], f64>
// CHECK: %[[res:.+]] = scf.if %[[cond]] -> (!test.test_memref<[4, 4], f64>) {
// CHECK: %[[dummy:.+]] = "test.dummy_memref_op"(%[[arg]]) : (!test.test_memref<[4, 4], f64>) -> !test.test_memref<[4, 4], f64>
// CHECK: scf.yield %[[dummy]] : !test.test_memref<[4, 4], f64>
// CHECK: } else {
// CHECK: scf.yield %[[arg]] : !test.test_memref<[4, 4], f64>
// CHECK: }
// CHECK: return %[[res]] : !test.test_memref<[4, 4], f64>
func.func @custom_types_scf_if_inplace(
    %arg: !test.test_tensor<[4, 4], f64>,
    %cond: i1)
    -> !test.test_tensor<[4, 4], f64> {
  %res = scf.if %cond -> (!test.test_tensor<[4, 4], f64>) {
    %dummy = "test.dummy_tensor_op"(%arg) : (!test.test_tensor<[4, 4], f64>)
      -> !test.test_tensor<[4, 4], f64>
    scf.yield %dummy : !test.test_tensor<[4, 4], f64>
  } else {
    scf.yield %arg : !test.test_tensor<[4, 4], f64>
  }
  return %res : !test.test_tensor<[4, 4], f64>
}

// -----

func.func private @custom_types_identity_2d(%arg: !test.test_tensor<[4, 4], f64>)
    -> !test.test_tensor<[4, 4], f64> {
  %out = "test.dummy_tensor_op"(%arg) : (!test.test_tensor<[4, 4], f64>)
    -> !test.test_tensor<[4, 4], f64>
  return %out : !test.test_tensor<[4, 4], f64>
}

// CHECK-LABEL: func.func @custom_types_scf_if_inplace_with_call(
// CHECK-SAME: %[[arg:.+]]: !test.test_memref<[4, 4], f64>
// CHECK-SAME: %[[cond:.+]]: i1
// CHECK-SAME: ) -> !test.test_memref<[4, 4], f64>
// CHECK: %[[res:.+]] = scf.if %[[cond]] -> (!test.test_memref<[4, 4], f64>) {
// CHECK: %[[call:.+]] = func.call @custom_types_identity_2d(%[[arg]]) : (!test.test_memref<[4, 4], f64>) -> !test.test_memref<[4, 4], f64>
// CHECK: scf.yield %[[call]] : !test.test_memref<[4, 4], f64>
// CHECK: } else {
// CHECK: scf.yield %[[arg]] : !test.test_memref<[4, 4], f64>
// CHECK: }
// CHECK: return %[[res]] : !test.test_memref<[4, 4], f64>
func.func @custom_types_scf_if_inplace_with_call(
    %arg: !test.test_tensor<[4, 4], f64>,
    %cond: i1)
    -> !test.test_tensor<[4, 4], f64> {
  %res = scf.if %cond -> (!test.test_tensor<[4, 4], f64>) {
    %call = func.call @custom_types_identity_2d(%arg)
      : (!test.test_tensor<[4, 4], f64>) -> !test.test_tensor<[4, 4], f64>
    scf.yield %call : !test.test_tensor<[4, 4], f64>
  } else {
    scf.yield %arg : !test.test_tensor<[4, 4], f64>
  }
  return %res : !test.test_tensor<[4, 4], f64>
}

// -----

// CHECK-LABEL: func.func @scf_while_inplace(
// CHECK-SAME: !test.test_memref<[4, 4], f64>
// CHECK: scf.while
// CHECK: scf.condition
// CHECK: scf.yield
// CHECK: return
func.func @scf_while_inplace(
    %arg: !test.test_tensor<[4, 4], f64>,
    %cond: i1)
    -> !test.test_tensor<[4, 4], f64> {
  %loop = scf.while (%iter = %arg)
      : (!test.test_tensor<[4, 4], f64>) -> !test.test_tensor<[4, 4], f64> {
    scf.condition(%cond) %iter : !test.test_tensor<[4, 4], f64>
  } do {
  ^bb0(%current: !test.test_tensor<[4, 4], f64>):
    %dummy = "test.dummy_tensor_op"(%current) : (!test.test_tensor<[4, 4], f64>)
      -> !test.test_tensor<[4, 4], f64>
    scf.yield %dummy : !test.test_tensor<[4, 4], f64>
  }
  return %loop : !test.test_tensor<[4, 4], f64>
}

// -----

func.func private @custom_types_identity_2d(%arg: !test.test_tensor<[4, 4], f64>)
    -> !test.test_tensor<[4, 4], f64> {
  %out = "test.dummy_tensor_op"(%arg) : (!test.test_tensor<[4, 4], f64>)
    -> !test.test_tensor<[4, 4], f64>
  return %out : !test.test_tensor<[4, 4], f64>
}

// CHECK-LABEL: func.func @scf_while_inplace(
// CHECK-SAME: !test.test_memref<[4, 4], f64>
// CHECK: scf.while
// CHECK: scf.condition
// CHECK: scf.yield
// CHECK: return
func.func @scf_while_inplace(
    %arg: !test.test_tensor<[4, 4], f64>,
    %cond: i1)
    -> !test.test_tensor<[4, 4], f64> {
  %loop = scf.while (%iter = %arg)
      : (!test.test_tensor<[4, 4], f64>) -> !test.test_tensor<[4, 4], f64> {
    scf.condition(%cond) %iter : !test.test_tensor<[4, 4], f64>
  } do {
  ^bb0(%current: !test.test_tensor<[4, 4], f64>):
    %call = func.call @custom_types_identity_2d(%current)
      : (!test.test_tensor<[4, 4], f64>) -> !test.test_tensor<[4, 4], f64>
    scf.yield %call : !test.test_tensor<[4, 4], f64>
  }
  return %loop : !test.test_tensor<[4, 4], f64>
}

// -----

#enc1 = #test.tensor_encoding<"custom">

module @BufferizeEncodingForCustomOpsInsideScf {
  // CHECK: func.func @custom_encoding_inside_scf(
  // CHECK-SAME:  %[[arg:.*]]: memref<42xf64, #test.memref_layout<"custom">>,
  // CHECK-SAME:  %[[lb:.*]]: index, %[[ub:.*]]: index, %[[step:.*]]: index)
  // CHECK-SAME:  -> memref<42xf64, #test.memref_layout<"custom">>
  func.func @custom_encoding_inside_scf(
      %arg: tensor<42xf64, #enc1>,
      %lb: index, %ub: index, %step: index)
      -> tensor<42xf64, #enc1> {
    // CHECK: %[[loop:.+]] = scf.for %{{.*}} = %[[lb]] to %[[ub]] step %[[step]]
    // CHECK-SAME: iter_args(%[[iter:.+]] = %[[arg]]) -> (memref<42xf64, #test.memref_layout<"custom">>) {
    // CHECK: %[[call:.+]] = "test.dummy_memref_op"(%[[iter]])
    // CHECK: scf.yield %[[call]] : memref<42xf64, #test.memref_layout<"custom">>
    %loop = scf.for %i = %lb to %ub step %step
        iter_args(%iter = %arg) -> (tensor<42xf64, #enc1>) {
      %call = "test.dummy_tensor_op"(%iter) : (tensor<42xf64, #enc1>)
        -> tensor<42xf64, #enc1>
      scf.yield %call : tensor<42xf64, #enc1>
    }

    // CHECK: return %[[loop]]
    return %loop : tensor<42xf64, #enc1>
  }
}
