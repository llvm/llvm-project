// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-cg-to-gpu))" | FileCheck %s

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
