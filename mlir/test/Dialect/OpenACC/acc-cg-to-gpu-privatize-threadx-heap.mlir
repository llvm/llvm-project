// RUN: mlir-opt %s --pass-pipeline="builtin.module(gpu.module(gpu.func(acc-cg-to-gpu{max-thread-private-stack=16384})))" --split-input-file | FileCheck %s

// ThreadX-private storage in a vector device routine that cannot use
// memref.alloca (over the stack budget, or a dynamic shape) must not go
// through the workgroup pointer-broadcast slot. That slot is indexed only by
// ThreadY, so vector lanes would race. Each lane gets its own memref.alloc.

module attributes {gpu.container_module} {
  acc.routine @routine_vector func(@large_static) vector

  gpu.module @cuda_device_mod {
    // CHECK-LABEL: gpu.func @large_static
    // CHECK:       %[[MEM:.*]] = memref.alloc() : memref<4096xi32>
    // CHECK-NOT:   memref.store %[[MEM]], {{.*}}#gpu.address_space<workgroup>
    // CHECK-NOT:   gpu.barrier
    // CHECK:       memref.store %{{.*}}, %[[MEM]][%{{.*}}] : memref<4096xi32>
    // CHECK:       memref.dealloc %[[MEM]] : memref<4096xi32>
    gpu.func @large_static()
        attributes {acc.specialized_routine = #acc.specialized_routine<@routine_vector, <vector>, "large_static">} {
      %c32 = arith.constant 32 : index
      %tx = acc.par_width %c32 par_dim(#acc.par_dim<thread_x>)
      %priv = acc.privatize {acc.active_par_dims = #acc<active_par_dims[thread_x]>}
          : () -> !acc.private_type<memref<4096xi32>>
      acc.compute_region launch(%arg0 = %tx) ins(%arg10 = %priv)
          : (!acc.private_type<memref<4096xi32>>) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : i32
        %local = acc.private_local %arg10
            {acc.active_par_dims = #acc<active_par_dims[thread_x]>}
            : (!acc.private_type<memref<4096xi32>>) -> memref<4096xi32>
        memref.store %c1, %local[%c0] : memref<4096xi32>
        acc.yield
      } <{origin = "acc.routine"}>
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  acc.routine @routine_vector_dyn func(@dynamic) vector

  gpu.module @cuda_device_mod {
    // CHECK-LABEL: gpu.func @dynamic
    // CHECK:       %[[MEM:.*]] = memref.alloc(%{{.*}}) : memref<?xi32>
    // CHECK-NOT:   memref.store %[[MEM]], {{.*}}#gpu.address_space<workgroup>
    // CHECK-NOT:   gpu.barrier
    // CHECK:       memref.store %{{.*}}, %[[MEM]][%{{.*}}] : memref<?xi32>
    // CHECK:       memref.dealloc %[[MEM]] : memref<?xi32>
    gpu.func @dynamic(%n: index)
        attributes {acc.specialized_routine = #acc.specialized_routine<@routine_vector_dyn, <vector>, "dynamic">} {
      %c32 = arith.constant 32 : index
      %tx = acc.par_width %c32 par_dim(#acc.par_dim<thread_x>)
      %priv = acc.privatize(%n) {acc.active_par_dims = #acc<active_par_dims[thread_x]>}
          : (index) -> !acc.private_type<memref<?xi32>>
      acc.compute_region launch(%arg0 = %tx) ins(%arg10 = %priv)
          : (!acc.private_type<memref<?xi32>>) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : i32
        %local = acc.private_local %arg10
            {acc.active_par_dims = #acc<active_par_dims[thread_x]>}
            : (!acc.private_type<memref<?xi32>>) -> memref<?xi32>
        memref.store %c1, %local[%c0] : memref<?xi32>
        acc.yield
      } <{origin = "acc.routine"}>
      gpu.return
    }
  }
}
