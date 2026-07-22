// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-cg-to-gpu))" | FileCheck %s
// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-cg-to-gpu{max-thread-private-stack=1}))" | FileCheck %s --check-prefix=FALLBACK

// CHECK-LABEL: func.func @threadprivate
// CHECK:       acc.privatize [#acc<par_dims[thread_x]>] : () -> !acc.private_type<memref<i32>>
// CHECK:       gpu.launch
// CHECK:         memref.alloca() : memref<i32>
// CHECK-NOT:     acc.gpu_shared_memory

func.func @threadprivate(%host: memref<i32>) {
  %c99 = arith.constant 99 : i32
  memref.store %c99, %host[] : memref<i32>
  %init = memref.load %host[] : memref<i32>
  %priv = acc.privatize [#acc<par_dims[thread_x]>] : () -> !acc.private_type<memref<i32>>

  acc.compute_region ins(%priv_in = %priv, %init_in = %init) :
      (!acc.private_type<memref<i32>>, i32) {
    %local = acc.private_local %priv_in : (!acc.private_type<memref<i32>>) -> memref<i32>
    memref.store %init_in, %local[] : memref<i32>
    %v = memref.load %local[] : memref<i32>
    %next = arith.addi %v, %v : i32
    memref.store %next, %local[] : memref<i32>
    acc.yield
  } {origin = "acc.parallel"}
  return
}

// CHECK-LABEL: func.func @threadprivate_array_reduction
// CHECK:       gpu.launch
// CHECK:         %[[PRIVATE:.*]] = memref.alloca() : memref<2xi32>
// CHECK:         %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK:         scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:           memref.store %[[ZERO]], %[[PRIVATE]][%[[IV]]]
// FALLBACK-LABEL: func.func @threadprivate_array_reduction
// FALLBACK:       gpu.launch
// FALLBACK-NOT:     memref.alloca() : memref<2xi32>
// FALLBACK:         %[[PRIVATE:.*]] = memref.subview
// FALLBACK:         %[[ZERO:.*]] = arith.constant 0 : i32
// FALLBACK:         scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// FALLBACK:           memref.store %[[ZERO]], %[[PRIVATE]][%[[IV]]]

func.func @threadprivate_array_reduction() {
  %priv = acc.privatize [#acc<par_dims[thread_x]>] : () -> !acc.private_type<memref<2xi32>>
  acc.compute_region ins(%priv_in = %priv) :
      (!acc.private_type<memref<2xi32>>) {
    %local = acc.private_local %priv_in
        {reduction_operator = #acc.reduction_operator<add>}
        : (!acc.private_type<memref<2xi32>>) -> memref<2xi32>
    acc.yield
  } {origin = "acc.parallel"}
  return
}

// CHECK-LABEL: func.func @workerprivate_array_reduction
// CHECK:       gpu.launch
// CHECK:         %[[TX:.*]] = gpu.thread_id x
// CHECK:         %[[LEADER:.*]] = arith.cmpi eq, %[[TX]], %{{.*}} : index
// CHECK:         scf.if %[[LEADER]]
// CHECK:           memref.store
// CHECK:         gpu.barrier scope <subgroup>
// CHECK-NOT:     gpu.barrier

func.func @workerprivate_array_reduction() {
  %priv = acc.privatize [#acc<par_dims[thread_y]>] : () -> !acc.private_type<memref<2xi32>>
  acc.compute_region ins(%priv_in = %priv) :
      (!acc.private_type<memref<2xi32>>) {
    %local = acc.private_local %priv_in
        {reduction_operator = #acc.reduction_operator<add>}
        : (!acc.private_type<memref<2xi32>>) -> memref<2xi32>
    acc.yield
  } {origin = "acc.parallel"}
  return
}
