// RUN: mlir-opt %s -test-one-shot-module-bufferize -split-input-file | FileCheck %s

#enc1 = #test.tensor_encoding<"hello">
#enc2 = #test.tensor_encoding<"not hello">

// CHECK-LABEL: @BufferizeEncodingThroughFunctionBoundaryAndCustomOps
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

// CHECK-LABEL: @BufferizeEncodingForCustomOpsInsideScf
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

// -----

#layout1 = #test.memref_layout<"layout_a">

module @BufferizeLayoutForFunction {
  // CHECK: func.func @layout_for_func
  // CHECK-SAME: -> memref<10xf32>
  func.func @layout_for_func() -> tensor<10xf32> {
    // CHECK: %[[memref:.*]] = "test.create_memref_op"
    // CHECK-SAME:  -> memref<10xf32, #test.memref_layout<"layout_a">>
    %memref = "test.tensor_with_future_layout"() {layout = #layout1}
      : () -> tensor<10xf32>

    // CHECK: %[[out:.*]] = memref.cast %[[memref]]
    // CHECK-SAME:  to memref<10xf32>
    // CHECK: return %[[out]]
    return %memref : tensor<10xf32>
  }
}

// -----

#layout1 = #test.memref_layout<"layout_a">
#layout2 = #test.memref_layout<"layout_b">

// CHECK-LABEL: @BufferizeLayoutMismatchInsideScfIf
module @BufferizeLayoutMismatchInsideScfIf {
  // CHECK: func.func @mismatch_in_if
  // CHECK-SAME: -> memref<10xf32>
  func.func @mismatch_in_if(%cond: i1) -> tensor<10xf32> {
    // CHECK: %[[if:.*]] = scf.if
    // CHECK-SAME: -> (memref<10xf32, #test.memref_layout<"layout_a">>)
    %ret = scf.if %cond -> tensor<10xf32> {
      // CHECK: %[[one:.*]] = "test.create_memref_op"
      // CHECK-SAME: -> memref<10xf32, #test.memref_layout<"layout_a">>
      %one = "test.tensor_with_future_layout"() {layout = #layout1}
        : () -> tensor<10xf32>
      // CHECK: scf.yield %[[one]]
      scf.yield %one : tensor<10xf32>
    } else {
      // CHECK: %[[another:.*]] = "test.create_memref_op"
      // CHECK-SAME: -> memref<10xf32, #test.memref_layout<"layout_b">>
      // CHECK: %[[cast:.*]] = memref.cast %[[another]]
      // CHECK-SAME: to memref<10xf32, #test.memref_layout<"layout_a">>
      %another = "test.tensor_with_future_layout"() {layout = #layout2}
        : () -> tensor<10xf32>
      // CHECK: scf.yield %[[cast]]
      scf.yield %another : tensor<10xf32>
    }

    // CHECK: %[[out:.*]] = memref.cast %[[if]]
    // CHECK-SAME: to memref<10xf32>
    // CHECK: return %[[out]]
    return %ret : tensor<10xf32>
  }
}

// -----

#layout1 = #test.memref_layout<"layout_a">
#layout2 = #test.memref_layout<"layout_b">
#layout3 = #test.memref_layout<"layout_c">

// CHECK-LABEL: @BufferizeLayoutMismatchInsideScfSwitch
module @BufferizeLayoutMismatchInsideScfSwitch {
  // CHECK: func.func @mismatch_in_switch
  // CHECK-SAME:  -> memref<10xf32>
  func.func @mismatch_in_switch(%idx: index) -> tensor<10xf32> {
    // CHECK: %[[switch:.*]] = scf.index_switch
    // CHECK-SAME: -> memref<10xf32, #test.memref_layout<"layout_a">>
    %ret = scf.index_switch %idx -> tensor<10xf32>
    case 0 {
      // CHECK: %[[one:.*]] = "test.create_memref_op"
      // CHECK-SAME: -> memref<10xf32, #test.memref_layout<"layout_a">>
      %one = "test.tensor_with_future_layout"() {layout = #layout1}
        : () -> tensor<10xf32>
      // CHECK: scf.yield %[[one]]
      scf.yield %one : tensor<10xf32>
    }
    case 1 {
      // CHECK: %[[another:.*]] = "test.create_memref_op"
      // CHECK-SAME: -> memref<10xf32, #test.memref_layout<"layout_b">>
      // CHECK: %[[cast:.*]] = memref.cast %[[another]]
      // CHECK-SAME: to memref<10xf32, #test.memref_layout<"layout_a">>
      %another = "test.tensor_with_future_layout"() {layout = #layout2}
        : () -> tensor<10xf32>
      // CHECK: scf.yield %[[cast]]
      scf.yield %another : tensor<10xf32>
    }
    default {
      // CHECK: %[[yet_another:.*]] = "test.create_memref_op"
      // CHECK-SAME: -> memref<10xf32, #test.memref_layout<"layout_c">>
      // CHECK: %[[cast:.*]] = memref.cast %[[yet_another]]
      // CHECK-SAME: to memref<10xf32, #test.memref_layout<"layout_a">>
      %yet_another = "test.tensor_with_future_layout"() {layout = #layout3}
        : () -> tensor<10xf32>
      // CHECK: scf.yield %[[cast]]
      scf.yield %yet_another : tensor<10xf32>
    }

    // CHECK: %[[out:.*]] = memref.cast %[[switch]]
    // CHECK-SAME: to memref<10xf32>
    // CHECK: return %[[out]]
    return %ret : tensor<10xf32>
  }
}

// -----

#layout1 = #test.memref_layout<"layout_a">
#layout2 = #test.memref_layout<"layout_b">

// CHECK-LABEL: @BufferizeLayoutMismatchInsideScfFor
module @BufferizeLayoutMismatchInsideScfFor {
  // CHECK: func.func @mismatch_in_for
  // CHECK-SAME:  -> memref<10xf32>
  func.func @mismatch_in_for(
      %lb: index, %ub: index, %step: index)
      -> tensor<10xf32> {
    // CHECK: %[[init:.*]] = "test.create_memref_op"
    // CHECK-SAME: -> memref<10xf32, #test.memref_layout<"layout_a">>
    %init = "test.tensor_with_future_layout"() {layout = #layout1}
      : () -> tensor<10xf32>

    // CHECK: %[[loop:.+]] = scf.for
    // CHECK-SAME: iter_args(%[[iter:.*]] = %[[init]])
    // CHECK-SAME: -> (memref<10xf32, #test.memref_layout<"layout_a">>)
    %loop = scf.for %i = %lb to %ub step %step
        iter_args(%iter = %init) -> (tensor<10xf32>) {
      // CHECK: %[[conflict:.*]] = "test.dummy_memref_op"(%[[iter]])
      // CHECK-SAME:  -> memref<10xf32, #test.memref_layout<"layout_b">>
      // CHECK: %[[cast:.*]] = memref.cast %[[conflict]]
      // CHECK-SAME: to memref<10xf32, #test.memref_layout<"layout_a">>
      %conflict = "test.force_new_layout"(%iter) {layout = #layout2}
        : (tensor<10xf32>) -> tensor<10xf32>
      // CHECK: scf.yield %[[cast]]
      scf.yield %conflict : tensor<10xf32>
    }

    // CHECK: %[[out:.*]] = memref.cast %[[loop]]
    // CHECK-SAME: to memref<10xf32>
    // CHECK: return %[[out]]
    return %loop : tensor<10xf32>
  }
}

// -----

#layout1 = #test.memref_layout<"layout_a">
#layout2 = #test.memref_layout<"layout_b">

// Test that custom layout can co-exist in principle within an "end-to-end"
// SCF example (`for { extract slice -> custom op -> insert slice }`) without
// bufferization failing completely due to a layout mismatch. The fact that the
// produced IR itself is rather dumb (e.g. memref.subview drops user-specified
// layout) is out of scope for now.

// CHECK-LABEL: @BufferizeLayoutMismatchInsideScfForWithSubviews
module @BufferizeLayoutMismatchInsideScfForWithSubviews {
  // CHECK: func.func @mismatch_in_for
  // CHECK-SAME:  -> memref<10xf32>
  func.func @mismatch_in_for(
      %lb: index, %ub: index, %step: index)
      -> tensor<10xf32> {
    %c0 = arith.constant 0 : index

    // CHECK: %[[init:.*]] = "test.create_memref_op"
    // CHECK-SAME: -> memref<10xf32, #test.memref_layout<"layout_a">>
    %init = "test.tensor_with_future_layout"() {layout = #layout1}
      : () -> tensor<10xf32>

    // CHECK: %[[loop:.+]] = scf.for
    // CHECK-SAME: iter_args(%[[inout:.*]] = %[[init]])
    // CHECK-SAME: -> (memref<10xf32, #test.memref_layout<"layout_a">>)
    %loop = scf.for %i = %lb to %ub step %step
        iter_args(%inout = %init) -> (tensor<10xf32>) {
      // CHECK: %[[in:.*]] = memref.subview %[[inout]]
      // CHECK-SAME: to memref<5xf32, strided<[1], offset: ?>>
      %in = tensor.extract_slice %inout[%c0] [5] [1]
        : tensor<10xf32> to tensor<5xf32>

      // CHECK: %[[conflict:.*]] = "test.dummy_memref_op"(%[[in]])
      // CHECK-SAME: -> memref<5xf32, #test.memref_layout<"layout_b">>
      %conflict = "test.force_new_layout"(%in) {layout = #layout2}
        : (tensor<5xf32>) -> tensor<5xf32>

      // CHECK: %[[out:.*]] = memref.subview %[[inout]]
      // CHECK-SAME: to memref<5xf32, strided<[1], offset: ?>>
      // CHECK: memref.copy %[[conflict]], %[[out]]
      %out = tensor.insert_slice %conflict into %inout[%c0] [5] [1]
        : tensor<5xf32> into tensor<10xf32>

      // CHECK: scf.yield %[[inout]]
      scf.yield %out : tensor<10xf32>
    }

    // CHECK: %[[out:.*]] = memref.cast %[[loop]]
    // CHECK-SAME: to memref<10xf32>
    // CHECK: return %[[out]]
    return %loop : tensor<10xf32>
  }
}

// -----

#layout1 = #test.memref_layout<"layout_a">
#layout2 = #test.memref_layout<"layout_b">

// CHECK-LABEL: @BufferizeLayoutMismatchInScfExecuteRegion
module @BufferizeLayoutMismatchInScfExecuteRegion {
  // CHECK: func.func @mismatch_in_scf_execute_region
  // CHECK-SAME: -> memref<10xf32>
  func.func @mismatch_in_scf_execute_region(%cond: i1) -> tensor<10xf32> {
    // CHECK: %[[region:.*]] = scf.execute_region
    // CHECK-SAME: -> memref<10xf32, #test.memref_layout<"layout_a">>
    %out = scf.execute_region -> tensor<10xf32> {
      cf.cond_br %cond, ^bb1, ^bb2

      ^bb1:
        // CHECK: %[[one:.*]] = "test.create_memref_op"
        // CHECK-SAME: -> memref<10xf32, #test.memref_layout<"layout_a">>
        %one = "test.tensor_with_future_layout"() {layout = #layout1}
          : () -> tensor<10xf32>
        // CHECK: cf.br ^bb3(%[[one]]
        cf.br ^bb3(%one : tensor<10xf32>)

      ^bb2:
        // CHECK: %[[another:.*]] = "test.create_memref_op"
        // CHECK-SAME: -> memref<10xf32, #test.memref_layout<"layout_b">>
        // CHECK: %[[cast:.*]] = memref.cast %[[another]]
        // CHECK-SAME: to memref<10xf32, #test.memref_layout<"layout_a">>
        %another = "test.tensor_with_future_layout"() {layout = #layout2}
          : () -> tensor<10xf32>
        // CHECK: cf.br ^bb3(%[[cast]]
        cf.br ^bb3(%another : tensor<10xf32>)

      ^bb3(%res: tensor<10xf32>):
        // CHECK: scf.yield {{%.*}} : memref<10xf32, #test.memref_layout<"layout_a">>
        scf.yield %res : tensor<10xf32>
    }

    // CHECK: %[[out:.*]] = memref.cast %[[region]]
    // CHECK-SAME: to memref<10xf32>
    // CHECK: return %[[out]]
    return %out : tensor<10xf32>
  }
}

// -----

#layout1 = #test.memref_layout<"layout_a">
#layout2 = #test.memref_layout<"layout_b">

// CHECK-LABEL: @BufferizeLayoutMismatchInArithSelect
module @BufferizeLayoutMismatchInArithSelect {
  // CHECK: func.func @mismatch_in_select(%[[cond:.*]]: i1)
  // CHECK-SAME:  -> memref<10xf32>
  func.func @mismatch_in_select(%cond: i1) -> tensor<10xf32> {
    // CHECK: %[[memref1:.*]] = "test.create_memref_op"
    // CHECK-SAME: -> memref<10xf32, #test.memref_layout<"layout_a">>
    %tensor1 = "test.tensor_with_future_layout"() {layout = #layout1}
      : () -> tensor<10xf32>
    // CHECK: %[[memref2:.*]] = "test.create_memref_op"
    // CHECK-SAME: -> memref<10xf32, #test.memref_layout<"layout_b">>
    %tensor2 = "test.tensor_with_future_layout"() {layout = #layout2}
      : () -> tensor<10xf32>

    // CHECK: %[[cast:.*]] = memref.cast %[[memref2]]
    // CHECK-SAME: to memref<10xf32, #test.memref_layout<"layout_a">>

    // CHECK: %[[select:.*]] = arith.select %[[cond]], %[[memref1]], %[[cast]]
    %select = arith.select %cond, %tensor1, %tensor2 : i1, tensor<10xf32>

    // CHECK: %[[out:.*]] = memref.cast %[[select]]
    // CHECK-SAME: to memref<10xf32>
    // CHECK: return %[[out]]
    return %select : tensor<10xf32>
  }
}
