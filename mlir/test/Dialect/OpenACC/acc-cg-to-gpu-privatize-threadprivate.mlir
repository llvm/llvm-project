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

// CHECK-LABEL: func.func @dynamic_threadprivate
// CHECK:       gpu.launch
// CHECK:         %[[STORAGE:.*]] = memref.view %{{.*}}[%{{.*}}][%{{.*}}, %{{.*}}, %{{.*}}]
// CHECK:         %[[SLICE:.*]] = memref.subview %[[STORAGE]]
// CHECK:         %{{.*}}, %[[OFFSET:.*]], %{{.*}}, %{{.*}} = memref.extract_strided_metadata %[[SLICE]]
// CHECK:         %[[ELEMENT_BYTES:.*]] = arith.constant 4 : index
// CHECK:         %[[BYTE_OFFSET:.*]] = arith.muli %[[OFFSET]], %[[ELEMENT_BYTES]]
// CHECK:         %[[PRIVATE:.*]] = memref.view %{{.*}}[%[[BYTE_OFFSET]]]
// CHECK:         memref.store %{{.*}}, %[[PRIVATE]][%{{.*}}] : memref<?xi32>

func.func @dynamic_threadprivate(%n: index) {
  %c4 = arith.constant 4 : index
  %c128 = arith.constant 128 : index
  %bx = acc.par_width %c4 {par_dim = #acc.par_dim<block_x>}
  %tx = acc.par_width %c128 {par_dim = #acc.par_dim<thread_x>}
  %private = acc.privatize(%n) [#acc<par_dims[block_x, thread_x]>]
      : (index) -> !acc.private_type<memref<?xi32>>

  acc.compute_region launch(%kbx = %bx, %ktx = %tx)
      ins(%arg = %private) : (!acc.private_type<memref<?xi32>>) {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1 : i32
    %local = acc.private_local %arg
        {acc.par_dims = #acc<par_dims[block_x, thread_x]>}
        : (!acc.private_type<memref<?xi32>>) -> memref<?xi32>
    memref.store %one, %local[%c0] : memref<?xi32>
    acc.yield
  } {origin = "acc.parallel"}
  return
}
