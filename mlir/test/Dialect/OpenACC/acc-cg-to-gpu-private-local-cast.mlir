// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-cg-to-gpu))" --split-input-file | FileCheck %s

// Materializing `acc.private_local` casts the storage to the requested surface
// type. A static per-thread allocation requested as a dynamically shaped memref
// produces a `memref.cast`.

// CHECK-LABEL: func.func @private_local_surface_cast
// CHECK:         %[[ALLOCA:.*]] = memref.alloca() : memref<4xf32>
// CHECK:         %[[CAST:.*]] = memref.cast %[[ALLOCA]] : memref<4xf32> to memref<?xf32>
// CHECK:         memref.store %{{.*}}, %[[CAST]][%{{.*}}] : memref<?xf32>

func.func @private_local_surface_cast() {
  %c4 = arith.constant 4 : index
  %c128 = arith.constant 128 : index
  %bx = acc.par_width %c4 par_dim(#acc.par_dim<block_x>)
  %tx = acc.par_width %c128 par_dim(#acc.par_dim<thread_x>)
  %priv = acc.privatize par_dims(#acc<par_dims[thread_x]>) : () -> !acc.private_type<memref<4xf32>>

  acc.compute_region launch(%kbx = %bx, %ktx = %tx) ins(%arg = %priv) : (!acc.private_type<memref<4xf32>>) {
    %c0 = arith.constant 0 : index
    %f0 = arith.constant 0.0 : f32
    %local = acc.private_local %arg : (!acc.private_type<memref<4xf32>>) -> memref<?xf32>
    memref.store %f0, %local[%c0] : memref<?xf32>
    acc.yield
  } <{origin = "acc.parallel"}>
  return
}

// -----

// Shared memory storage lives in the workgroup address space, so reaching the
// requested surface type produces a `memref.memory_space_cast`.

// CHECK-LABEL: func.func @private_local_shared_memory_cast
// CHECK:         %[[SHARED:.*]] = acc.gpu_shared_memory
// CHECK-SAME:      memref<2xi32, #gpu.address_space<workgroup>>
// CHECK:         %[[CAST:.*]] = memref.memory_space_cast %[[SHARED]] : memref<2xi32, #gpu.address_space<workgroup>> to memref<2xi32>
// CHECK:         memref.store %{{.*}}, %[[CAST]][%{{.*}}] : memref<2xi32>

func.func @private_local_shared_memory_cast() {
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index
  %c32 = arith.constant 32 : index
  %nw = arith.addi %c1, %c1 : index
  %block_x = acc.par_width %c5 par_dim(#acc.par_dim<block_x>)
  %thread_y = acc.par_width %nw par_dim(#acc.par_dim<thread_y>)
  %thread_x = acc.par_width %c32 par_dim(#acc.par_dim<thread_x>)
  acc.kernel_environment {
    %priv = acc.privatize : () -> !acc.private_type<memref<2xi32>>
    acc.compute_region launch(%arg0 = %block_x, %arg1 = %thread_y, %arg2 = %thread_x) ins(%arg10 = %priv) : (!acc.private_type<memref<2xi32>>) {
      %c0 = arith.constant 0 : index
      %c1_inner = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      scf.parallel (%iv) = (%c0) to (%arg1) step (%c1_inner) {
        %local = acc.private_local %arg10 : (!acc.private_type<memref<2xi32>>) -> memref<2xi32>
        memref.store %c0_i32, %local[%c0] : memref<2xi32>
        scf.reduce
      } {acc.par_dims = #acc<par_dims[thread_y]>}
      acc.yield
    } <{origin = "acc.parallel"}>
  }
  return
}

// -----

// When the storage already has the requested surface type, no cast is emitted.

// CHECK-LABEL: func.func @private_local_no_cast
// CHECK:         %[[ALLOCA:.*]] = memref.alloca() : memref<4xf32>
// CHECK-NOT:     memref.cast
// CHECK:         memref.store %{{.*}}, %[[ALLOCA]][%{{.*}}] : memref<4xf32>

func.func @private_local_no_cast() {
  %c4 = arith.constant 4 : index
  %c128 = arith.constant 128 : index
  %bx = acc.par_width %c4 par_dim(#acc.par_dim<block_x>)
  %tx = acc.par_width %c128 par_dim(#acc.par_dim<thread_x>)
  %priv = acc.privatize par_dims(#acc<par_dims[thread_x]>) : () -> !acc.private_type<memref<4xf32>>

  acc.compute_region launch(%kbx = %bx, %ktx = %tx) ins(%arg = %priv) : (!acc.private_type<memref<4xf32>>) {
    %c0 = arith.constant 0 : index
    %f0 = arith.constant 0.0 : f32
    %local = acc.private_local %arg : (!acc.private_type<memref<4xf32>>) -> memref<4xf32>
    memref.store %f0, %local[%c0] : memref<4xf32>
    acc.yield
  } <{origin = "acc.parallel"}>
  return
}
