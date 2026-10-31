// RUN: mlir-opt %s --pass-pipeline="builtin.module(gpu.module(gpu.func(acc-cg-to-gpu)))" --split-input-file | FileCheck %s

// A gang routine runs once per block, so a small static private shared by the
// block's threads is placed in a workgroup attribution of its own instead of a
// thread-0 heap allocation whose pointer is broadcast through one.

module attributes {gpu.container_module} {
  acc.routine @routine_gang func(@gang_scalar) gang

  gpu.module @cuda_device_mod {
    func.func private @use(memref<f64>)

    // CHECK-LABEL: gpu.func @gang_scalar
    // CHECK-SAME:    workgroup(%[[SLOT:[^ ]+]] : memref<f64, #gpu.address_space<workgroup>> {llvm.align = 16 : i32})
    // CHECK-NOT:     memref.alloc
    // CHECK:         %[[MEM:.*]] = memref.memory_space_cast %[[SLOT]] : memref<f64, #gpu.address_space<workgroup>> to memref<f64>
    // CHECK:         memref.store %{{.*}}, %[[MEM]][] : memref<f64>
    // CHECK:         func.call @use(%[[MEM]])
    // CHECK:         gpu.barrier
    // CHECK-NOT:     memref.dealloc
    // CHECK:         gpu.return
    gpu.func @gang_scalar()
        attributes {acc.specialized_routine = #acc.specialized_routine<@routine_gang, <gang_dim1>, "gang_scalar">} {
      %c1 = arith.constant 1 : index
      %bx = acc.par_width %c1 par_dim(#acc.par_dim<block_x>)
      %tx = acc.par_width %c1 par_dim(#acc.par_dim<thread_x>)
      %priv = acc.privatize {acc.active_par_dims = #acc<active_par_dims[block_x]>}
          : () -> !acc.private_type<memref<f64>>
      acc.compute_region launch(%arg0 = %bx, %arg1 = %tx) ins(%arg2 = %priv)
          : (!acc.private_type<memref<f64>>) {
        %cst = arith.constant 2.0 : f64
        %local = acc.private_local %arg2
            {acc.active_par_dims = #acc<active_par_dims[block_x]>}
            : (!acc.private_type<memref<f64>>) -> memref<f64>
        acc.predicate_region {
          memref.store %cst, %local[] : memref<f64>
        }
        acc.predicate_region {
          func.call @use(%local) : (memref<f64>) -> ()
        }
        acc.yield
      } <{origin = "acc.routine"}>
      gpu.return
    }
  }
}

// -----

// The slots of one routine share a small byte budget; a private that does not
// fit keeps the heap allocation broadcast through a pointer slot.

module attributes {gpu.container_module} {
  acc.routine @routine_gang func(@gang_budget) gang

  gpu.module @cuda_device_mod {
    func.func private @use2(memref<24xf64>, memref<16xf64>)

    // CHECK-LABEL: gpu.func @gang_budget
    // CHECK-SAME:    workgroup(%{{[^ ]+}} : memref<24xf64, #gpu.address_space<workgroup>> {llvm.align = 16 : i32}, %{{[^ ]+}} : memref<memref<16xf64>, #gpu.address_space<workgroup>> {llvm.align = 16 : i32})
    // CHECK-NOT:     memref.alloc() : memref<24xf64>
    // CHECK:         memref.alloc() : memref<16xf64>
    // CHECK:         memref.dealloc %{{.*}} : memref<16xf64>
    gpu.func @gang_budget()
        attributes {acc.specialized_routine = #acc.specialized_routine<@routine_gang, <gang_dim1>, "gang_budget">} {
      %c1 = arith.constant 1 : index
      %bx = acc.par_width %c1 par_dim(#acc.par_dim<block_x>)
      %tx = acc.par_width %c1 par_dim(#acc.par_dim<thread_x>)
      %p0 = acc.privatize {acc.active_par_dims = #acc<active_par_dims[block_x]>}
          : () -> !acc.private_type<memref<24xf64>>
      %p1 = acc.privatize {acc.active_par_dims = #acc<active_par_dims[block_x]>}
          : () -> !acc.private_type<memref<16xf64>>
      acc.compute_region launch(%arg0 = %bx, %arg1 = %tx) ins(%arg2 = %p0, %arg3 = %p1)
          : (!acc.private_type<memref<24xf64>>, !acc.private_type<memref<16xf64>>) {
        %l0 = acc.private_local %arg2
            {acc.active_par_dims = #acc<active_par_dims[block_x]>}
            : (!acc.private_type<memref<24xf64>>) -> memref<24xf64>
        %l1 = acc.private_local %arg3
            {acc.active_par_dims = #acc<active_par_dims[block_x]>}
            : (!acc.private_type<memref<16xf64>>) -> memref<16xf64>
        acc.predicate_region {
          func.call @use2(%l0, %l1) : (memref<24xf64>, memref<16xf64>) -> ()
        }
        acc.yield
      } <{origin = "acc.routine"}>
      gpu.return
    }
  }
}

// -----

// A runtime-sized private keeps the heap allocation.

module attributes {gpu.container_module} {
  acc.routine @routine_gang func(@gang_dynamic) gang

  gpu.module @cuda_device_mod {
    func.func private @use_dyn(memref<?xf64>)

    // CHECK-LABEL: gpu.func @gang_dynamic
    // CHECK-SAME:    workgroup(%{{[^ ]+}} : memref<memref<?xf64>, #gpu.address_space<workgroup>> {llvm.align = 16 : i32})
    // CHECK:         memref.alloc(%{{.*}}) : memref<?xf64>
    // CHECK:         memref.dealloc %{{.*}} : memref<?xf64>
    gpu.func @gang_dynamic(%n: index)
        attributes {acc.specialized_routine = #acc.specialized_routine<@routine_gang, <gang_dim1>, "gang_dynamic">} {
      %c1 = arith.constant 1 : index
      %bx = acc.par_width %c1 par_dim(#acc.par_dim<block_x>)
      %tx = acc.par_width %c1 par_dim(#acc.par_dim<thread_x>)
      %priv = acc.privatize(%n) {acc.active_par_dims = #acc<active_par_dims[block_x]>}
          : (index) -> !acc.private_type<memref<?xf64>>
      acc.compute_region launch(%arg0 = %bx, %arg1 = %tx) ins(%arg2 = %priv)
          : (!acc.private_type<memref<?xf64>>) {
        %local = acc.private_local %arg2
            {acc.active_par_dims = #acc<active_par_dims[block_x]>}
            : (!acc.private_type<memref<?xf64>>) -> memref<?xf64>
        acc.predicate_region {
          func.call @use_dyn(%local) : (memref<?xf64>) -> ()
        }
        acc.yield
      } <{origin = "acc.routine"}>
      gpu.return
    }
  }
}

// -----

// A worker routine keeps the heap allocation broadcast through a pointer slot.

module attributes {gpu.container_module} {
  acc.routine @routine_worker func(@worker_scalar) worker

  gpu.module @cuda_device_mod {
    func.func private @use(memref<f64>)

    // CHECK-LABEL: gpu.func @worker_scalar
    // CHECK-SAME:    workgroup(%{{[^ ]+}} : memref<memref<f64>, #gpu.address_space<workgroup>> {llvm.align = 16 : i32})
    // CHECK:         memref.alloc() : memref<f64>
    // CHECK:         memref.dealloc %{{.*}} : memref<f64>
    gpu.func @worker_scalar()
        attributes {acc.specialized_routine = #acc.specialized_routine<@routine_worker, <worker>, "worker_scalar">} {
      %c1 = arith.constant 1 : index
      %ty = acc.par_width %c1 par_dim(#acc.par_dim<thread_y>)
      %tx = acc.par_width %c1 par_dim(#acc.par_dim<thread_x>)
      %priv = acc.privatize : () -> !acc.private_type<memref<f64>>
      acc.compute_region launch(%arg0 = %ty, %arg1 = %tx) ins(%arg2 = %priv)
          : (!acc.private_type<memref<f64>>) {
        %local = acc.private_local %arg2 : (!acc.private_type<memref<f64>>) -> memref<f64>
        acc.predicate_region {
          func.call @use(%local) : (memref<f64>) -> ()
        }
        acc.yield
      } <{origin = "acc.routine"}>
      gpu.return
    }
  }
}

// -----

// A private larger than the slot budget keeps the heap allocation.

module attributes {gpu.container_module} {
  acc.routine @routine_gang func(@gang_large) gang

  gpu.module @cuda_device_mod {
    func.func private @use_large(memref<64xf64>)

    // CHECK-LABEL: gpu.func @gang_large
    // CHECK-SAME:    workgroup(%{{[^ ]+}} : memref<memref<64xf64>, #gpu.address_space<workgroup>> {llvm.align = 16 : i32})
    // CHECK:         memref.alloc() : memref<64xf64>
    // CHECK:         memref.dealloc %{{.*}} : memref<64xf64>
    gpu.func @gang_large()
        attributes {acc.specialized_routine = #acc.specialized_routine<@routine_gang, <gang_dim1>, "gang_large">} {
      %c1 = arith.constant 1 : index
      %bx = acc.par_width %c1 par_dim(#acc.par_dim<block_x>)
      %tx = acc.par_width %c1 par_dim(#acc.par_dim<thread_x>)
      %priv = acc.privatize {acc.active_par_dims = #acc<active_par_dims[block_x]>}
          : () -> !acc.private_type<memref<64xf64>>
      acc.compute_region launch(%arg0 = %bx, %arg1 = %tx) ins(%arg2 = %priv)
          : (!acc.private_type<memref<64xf64>>) {
        %local = acc.private_local %arg2
            {acc.active_par_dims = #acc<active_par_dims[block_x]>}
            : (!acc.private_type<memref<64xf64>>) -> memref<64xf64>
        acc.predicate_region {
          func.call @use_large(%local) : (memref<64xf64>) -> ()
        }
        acc.yield
      } <{origin = "acc.routine"}>
      gpu.return
    }
  }
}
