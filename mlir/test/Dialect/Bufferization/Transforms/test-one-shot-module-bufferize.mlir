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

#enc1 = #test.tensor_encoding<"hello">
#enc2 = #test.tensor_encoding<"not hello">

// The memref's layout must come from the encoding, not from the default
// static-identity layout.
module @BufferizeEncodingForAlloc {
  // CHECK: func @some_func(
  // CHECK-SAME:  %[[arg0:.*]]: memref<42xf32, #test.memref_layout<"hello">>)
  // CHECK-SAME:  -> (memref<42xf32, #test.memref_layout<"hello">>,
  // CHECK-SAME:      memref<42xf32, #test.memref_layout<"not hello">>)
  func.func @some_func(%t0: tensor<42xf32, #enc1>)
      -> (tensor<42xf32, #enc1>, tensor<42xf32, #enc2>) {
    // CHECK: %[[T0:.+]] = memref.alloc() {{.*}} : memref<42xf32, #test.memref_layout<"hello">>
    %0 = bufferization.alloc_tensor() : tensor<42xf32, #enc1>

    // CHECK: %[[T1:.+]] = memref.alloc() {{.*}} : memref<42xf32, #test.memref_layout<"not hello">>
    %1 = bufferization.alloc_tensor() : tensor<42xf32, #enc2>

    // CHECK: return %[[T0]], %[[T1]]
    return %0, %1 : tensor<42xf32, #enc1>, tensor<42xf32, #enc2>
  }
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
