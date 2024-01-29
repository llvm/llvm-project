// RUN: mlir-opt %s -allow-unregistered-dialect \
// RUN:     -transform-interpreter -canonicalize \
// RUN:     -split-input-file -verify-diagnostics | FileCheck %s

// CHECK: #[[$map:.*]] = affine_map<()[s0] -> (s0 - 1)>
// CHECK-LABEL: func @make_alloca_loop_independent(
//  CHECK-SAME:     %[[lb:.*]]: index, %[[ub:.*]]: index, %[[step:.*]]: index)
func.func @make_alloca_loop_independent(%lb: index, %ub: index, %step: index) {
  %cst = arith.constant 5.5 : f32
  %c0 = arith.constant 0 : index
  // CHECK: scf.for %[[iv:.*]] = %[[lb]] to %[[ub]]
  scf.for %i = %lb to %ub step %step {
    // CHECK: %[[sz:.*]] = affine.apply #[[$map]]()[%[[ub]]]
    // CHECK: %[[alloca:.*]] = memref.alloca(%[[sz]])
    // CHECK: %[[subview:.*]] = memref.subview %[[alloca]][0] [%[[iv]]] [1] : memref<?xf32> to memref<?xf32, strided<[1]>>
    // CHECK: %[[cast:.*]] = builtin.unrealized_conversion_cast %[[subview]] : memref<?xf32, strided<[1]>> to memref<?xf32>
    %alloc = memref.alloca(%i) : memref<?xf32>

    // memref.subview has special handling.
    // CHECK: %[[subview2:.*]] = memref.subview %[[subview]][1] [5] [1] : memref<?xf32, strided<[1]>> to memref<5xf32, strided<[1], offset: 1>>
    %view = memref.subview %alloc[1][5][1] : memref<?xf32> to memref<5xf32, strided<[1], offset: 1>>

    // This op takes a memref but does not produce one. The new alloc is used
    // directly.
    // CHECK: "test.some_use"(%[[subview2]])
    "test.some_use"(%view) : (memref<5xf32, strided<[1], offset: 1>>) -> ()

    // This op produces a memref, so the new alloc cannot be used directly.
    // It is wrapped in a unrealized_conversion_cast.
    // CHECK: "test.another_use"(%[[cast]]) : (memref<?xf32>) -> memref<?xf32>
    "test.another_use"(%alloc) : (memref<?xf32>) -> (memref<?xf32>)

    // CHECK: memref.store %{{.*}}, %[[subview]]
    memref.store %cst, %alloc[%c0] : memref<?xf32>
  }
  return
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["memref.alloca"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.memref.make_loop_independent %0 {num_loops = 1} : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK: #[[$map:.*]] = affine_map<(d0) -> (-d0 + 128)>
// CHECK-LABEL: func @make_alloca_loop_independent_static(
func.func @make_alloca_loop_independent_static(%step: index) {
  %cst = arith.constant 5.5 : f32
  %c0 = arith.constant 0 : index
  %ub = arith.constant 128 : index
  // CHECK: scf.for %[[iv:.*]] =
  scf.for %i = %c0 to %ub step %step {
    // CHECK: %[[sz:.*]] = affine.apply #[[$map]](%[[iv]])
    %sz = affine.apply affine_map<(d0)[s0] -> (-d0 + s0)>(%i)[%ub]

    // CHECK: %[[alloca:.*]] = memref.alloca() : memref<128xf32>
    // CHECK: %[[subview:.*]] = memref.subview %[[alloca]][0] [%[[sz]]] [1] : memref<128xf32> to memref<?xf32, strided<[1]>>
    %alloc = memref.alloca(%sz) : memref<?xf32>

    // CHECK: memref.store %{{.*}}, %[[subview]]
    memref.store %cst, %alloc[%c0] : memref<?xf32>

    // CHECK: vector.print %[[sz]]
    %dim = memref.dim %alloc, %c0 : memref<?xf32>
    vector.print %dim : index
  }
  return
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["memref.alloca"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.memref.make_loop_independent %0 {num_loops = 1} : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
