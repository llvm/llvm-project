// RUN: mlir-opt %s --pass-pipeline="builtin.module(gpu.module(gpu.func(acc-cg-to-gpu{device-type=nvidia})))" --split-input-file | FileCheck %s

// A small static private shared by the threads of a routine is placed in a
// workgroup attribution of its own instead of a thread-0 heap allocation whose
// pointer is broadcast through one. A gang routine runs once per block.

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

// The slots of one routine share a small byte budget, padded to 16-byte
// alignment; a private that does not fit keeps the heap allocation broadcast
// through a pointer slot.

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

// A worker routine is called by all threads of a block, which share one slot.

module attributes {gpu.container_module} {
  acc.routine @routine_worker func(@worker_scalar) worker

  gpu.module @cuda_device_mod {
    func.func private @use(memref<f64>)

    // CHECK-LABEL: gpu.func @worker_scalar
    // CHECK-SAME:    workgroup(%[[SLOT:[^ ]+]] : memref<f64, #gpu.address_space<workgroup>> {llvm.align = 16 : i32})
    // CHECK-NOT:     memref.alloc
    // CHECK:         %[[MEM:.*]] = memref.memory_space_cast %[[SLOT]]
    // CHECK:         func.call @use(%[[MEM]])
    // CHECK:         gpu.barrier{{$}}
    // CHECK-NOT:     memref.dealloc
    // CHECK:         gpu.return
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

// -----

// A gang routine that calls another gang routine still takes a slot, which
// stays live across the call: the two routines' slots are distinct.

module attributes {gpu.container_module} {
  func.func private @gang_inner() attributes {acc.routine_info = #acc.routine_info<[@routine_inner]>}
  acc.routine @routine_inner func(@gang_inner) gang
  acc.routine @routine_gang func(@gang_outer) gang

  gpu.module @cuda_device_mod {
    func.func private @gang_inner()
    func.func private @use(memref<f64>)

    // CHECK-LABEL: gpu.func @gang_outer
    // CHECK-SAME:    workgroup(%[[SLOT:[^ ]+]] : memref<f64, #gpu.address_space<workgroup>> {llvm.align = 16 : i32})
    // CHECK-NOT:     memref.alloc
    // CHECK:         %[[MEM:.*]] = memref.memory_space_cast %[[SLOT]]
    // CHECK:         memref.store %{{.*}}, %[[MEM]][] : memref<f64>
    // CHECK:         call @gang_inner()
    // CHECK:         func.call @use(%[[MEM]])
    // CHECK:         gpu.barrier
    // CHECK-NOT:     memref.dealloc
    // CHECK:         gpu.return
    gpu.func @gang_outer()
        attributes {acc.specialized_routine = #acc.specialized_routine<@routine_gang, <gang_dim1>, "gang_outer">} {
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
        func.call @gang_inner() : () -> ()
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

// Slots are padded to 16-byte alignment: 1 + 232 + 7 bytes fit in the budget
// unpadded, but the slots start at offsets 0, 16 and 256, so the last does not.

module attributes {gpu.container_module} {
  acc.routine @routine_gang func(@gang_padding) gang

  gpu.module @cuda_device_mod {
    func.func private @use3(memref<i8>, memref<29xf64>, memref<7xi8>)

    // CHECK-LABEL: gpu.func @gang_padding
    // CHECK-SAME:    workgroup(%{{[^ ]+}} : memref<i8, #gpu.address_space<workgroup>> {llvm.align = 16 : i32}, %{{[^ ]+}} : memref<29xf64, #gpu.address_space<workgroup>> {llvm.align = 16 : i32}, %{{[^ ]+}} : memref<memref<7xi8>, #gpu.address_space<workgroup>> {llvm.align = 16 : i32})
    // CHECK:         memref.alloc() : memref<7xi8>
    gpu.func @gang_padding()
        attributes {acc.specialized_routine = #acc.specialized_routine<@routine_gang, <gang_dim1>, "gang_padding">} {
      %c1 = arith.constant 1 : index
      %bx = acc.par_width %c1 par_dim(#acc.par_dim<block_x>)
      %tx = acc.par_width %c1 par_dim(#acc.par_dim<thread_x>)
      %p0 = acc.privatize {acc.active_par_dims = #acc<active_par_dims[block_x]>}
          : () -> !acc.private_type<memref<i8>>
      %p1 = acc.privatize {acc.active_par_dims = #acc<active_par_dims[block_x]>}
          : () -> !acc.private_type<memref<29xf64>>
      %p2 = acc.privatize {acc.active_par_dims = #acc<active_par_dims[block_x]>}
          : () -> !acc.private_type<memref<7xi8>>
      acc.compute_region launch(%arg0 = %bx, %arg1 = %tx) ins(%arg2 = %p0, %arg3 = %p1, %arg4 = %p2)
          : (!acc.private_type<memref<i8>>, !acc.private_type<memref<29xf64>>, !acc.private_type<memref<7xi8>>) {
        %l0 = acc.private_local %arg2
            {acc.active_par_dims = #acc<active_par_dims[block_x]>}
            : (!acc.private_type<memref<i8>>) -> memref<i8>
        %l1 = acc.private_local %arg3
            {acc.active_par_dims = #acc<active_par_dims[block_x]>}
            : (!acc.private_type<memref<29xf64>>) -> memref<29xf64>
        %l2 = acc.private_local %arg4
            {acc.active_par_dims = #acc<active_par_dims[block_x]>}
            : (!acc.private_type<memref<7xi8>>) -> memref<7xi8>
        acc.predicate_region {
          func.call @use3(%l0, %l1, %l2) : (memref<i8>, memref<29xf64>, memref<7xi8>) -> ()
        }
        acc.yield
      } <{origin = "acc.routine"}>
      gpu.return
    }
  }
}

// -----

// The slots of every compute region of a routine share one budget.

module attributes {gpu.container_module} {
  acc.routine @routine_gang func(@gang_two_regions) gang

  gpu.module @cuda_device_mod {
    func.func private @use24(memref<24xf64>)

    // CHECK-LABEL: gpu.func @gang_two_regions
    // CHECK-SAME:    workgroup(%{{[^ ]+}} : memref<24xf64, #gpu.address_space<workgroup>> {llvm.align = 16 : i32}, %{{[^ ]+}} : memref<memref<24xf64>, #gpu.address_space<workgroup>> {llvm.align = 16 : i32})
    // CHECK:         memref.alloc() : memref<24xf64>
    // CHECK-NOT:     memref.alloc() : memref<24xf64>
    // CHECK:         gpu.return
    gpu.func @gang_two_regions()
        attributes {acc.specialized_routine = #acc.specialized_routine<@routine_gang, <gang_dim1>, "gang_two_regions">} {
      %c1 = arith.constant 1 : index
      %bx = acc.par_width %c1 par_dim(#acc.par_dim<block_x>)
      %tx = acc.par_width %c1 par_dim(#acc.par_dim<thread_x>)
      %p0 = acc.privatize {acc.active_par_dims = #acc<active_par_dims[block_x]>}
          : () -> !acc.private_type<memref<24xf64>>
      acc.compute_region launch(%arg0 = %bx, %arg1 = %tx) ins(%arg2 = %p0)
          : (!acc.private_type<memref<24xf64>>) {
        %l0 = acc.private_local %arg2
            {acc.active_par_dims = #acc<active_par_dims[block_x]>}
            : (!acc.private_type<memref<24xf64>>) -> memref<24xf64>
        acc.predicate_region {
          func.call @use24(%l0) : (memref<24xf64>) -> ()
        }
        acc.yield
      } <{origin = "acc.routine"}>
      %p1 = acc.privatize {acc.active_par_dims = #acc<active_par_dims[block_x]>}
          : () -> !acc.private_type<memref<24xf64>>
      acc.compute_region launch(%arg0 = %bx, %arg1 = %tx) ins(%arg2 = %p1)
          : (!acc.private_type<memref<24xf64>>) {
        %l1 = acc.private_local %arg2
            {acc.active_par_dims = #acc<active_par_dims[block_x]>}
            : (!acc.private_type<memref<24xf64>>) -> memref<24xf64>
        acc.predicate_region {
          func.call @use24(%l1) : (memref<24xf64>) -> ()
        }
        acc.yield
      } <{origin = "acc.routine"}>
      gpu.return
    }
  }
}

// -----

// ThreadY rows call a vector routine independently, so its slot holds one
// 16-byte-aligned copy per row, selected by the thread's ThreadY id.

module attributes {gpu.container_module} {
  acc.routine @routine_vector func(@vector_scalar) vector

  gpu.module @cuda_device_mod {
    func.func private @use(memref<f64>)

    // CHECK-LABEL: gpu.func @vector_scalar
    // CHECK-SAME:    workgroup(%[[SLAB:[^ ]+]] : memref<512xi8, #gpu.address_space<workgroup>> {llvm.align = 16 : i32})
    // CHECK-NOT:     memref.alloc
    // CHECK:         arith.constant 32 : index
    // CHECK-NEXT:    %[[TY:.*]] = gpu.thread_id y
    // CHECK-NEXT:    %[[STRIDE:.*]] = arith.constant 16 : index
    // CHECK-NEXT:    %[[OFF:.*]] = arith.muli %[[TY]], %[[STRIDE]] : index
    // CHECK:         %[[VIEW:.*]] = memref.view %[[SLAB]][%[[OFF]]][] : memref<512xi8, #gpu.address_space<workgroup>> to memref<f64, #gpu.address_space<workgroup>>
    // CHECK:         %[[MEM:.*]] = memref.memory_space_cast %[[VIEW]]
    // CHECK:         func.call @use(%[[MEM]])
    // CHECK:         gpu.barrier scope <subgroup>
    // CHECK-NOT:     memref.dealloc
    // CHECK:         gpu.return
    gpu.func @vector_scalar()
        attributes {acc.specialized_routine = #acc.specialized_routine<@routine_vector, <vector>, "vector_scalar">} {
      %c32 = arith.constant 32 : index
      %tx = acc.par_width %c32 par_dim(#acc.par_dim<thread_x>)
      %priv = acc.privatize {acc.active_par_dims = #acc<active_par_dims[]>}
          : () -> !acc.private_type<memref<f64>>
      acc.compute_region launch(%arg0 = %tx) ins(%arg1 = %priv)
          : (!acc.private_type<memref<f64>>) {
        %local = acc.private_local %arg1 {acc.active_par_dims = #acc<active_par_dims[]>}
            : (!acc.private_type<memref<f64>>) -> memref<f64>
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

// A vector routine's per-row slots share a budget of 64 bytes per row: four
// padded f64 copies fit, the fifth keeps the heap allocation.

module attributes {gpu.container_module} {
  acc.routine @routine_vector func(@vector_budget) vector

  gpu.module @cuda_device_mod {
    // CHECK-LABEL: gpu.func @vector_budget
    // CHECK-COUNT-4: memref.view
    // CHECK-NOT:     memref.view
    // CHECK:         memref.alloc() : memref<f64>
    // CHECK:         memref.dealloc
    gpu.func @vector_budget()
        attributes {acc.specialized_routine = #acc.specialized_routine<@routine_vector, <vector>, "vector_budget">} {
      %c32 = arith.constant 32 : index
      %tx = acc.par_width %c32 par_dim(#acc.par_dim<thread_x>)
      %p0 = acc.privatize {acc.active_par_dims = #acc<active_par_dims[]>}
          : () -> !acc.private_type<memref<f64>>
      %p1 = acc.privatize {acc.active_par_dims = #acc<active_par_dims[]>}
          : () -> !acc.private_type<memref<f64>>
      %p2 = acc.privatize {acc.active_par_dims = #acc<active_par_dims[]>}
          : () -> !acc.private_type<memref<f64>>
      %p3 = acc.privatize {acc.active_par_dims = #acc<active_par_dims[]>}
          : () -> !acc.private_type<memref<f64>>
      %p4 = acc.privatize {acc.active_par_dims = #acc<active_par_dims[]>}
          : () -> !acc.private_type<memref<f64>>
      acc.compute_region launch(%arg0 = %tx) ins(%a0 = %p0, %a1 = %p1, %a2 = %p2, %a3 = %p3, %a4 = %p4)
          : (!acc.private_type<memref<f64>>, !acc.private_type<memref<f64>>, !acc.private_type<memref<f64>>, !acc.private_type<memref<f64>>, !acc.private_type<memref<f64>>) {
        %cst = arith.constant 0.0 : f64
        %l0 = acc.private_local %a0 {acc.active_par_dims = #acc<active_par_dims[]>}
            : (!acc.private_type<memref<f64>>) -> memref<f64>
        %l1 = acc.private_local %a1 {acc.active_par_dims = #acc<active_par_dims[]>}
            : (!acc.private_type<memref<f64>>) -> memref<f64>
        %l2 = acc.private_local %a2 {acc.active_par_dims = #acc<active_par_dims[]>}
            : (!acc.private_type<memref<f64>>) -> memref<f64>
        %l3 = acc.private_local %a3 {acc.active_par_dims = #acc<active_par_dims[]>}
            : (!acc.private_type<memref<f64>>) -> memref<f64>
        %l4 = acc.private_local %a4 {acc.active_par_dims = #acc<active_par_dims[]>}
            : (!acc.private_type<memref<f64>>) -> memref<f64>
        acc.predicate_region {
          memref.store %cst, %l0[] : memref<f64>
          memref.store %cst, %l1[] : memref<f64>
          memref.store %cst, %l2[] : memref<f64>
          memref.store %cst, %l3[] : memref<f64>
          memref.store %cst, %l4[] : memref<f64>
        }
        acc.yield
      } <{origin = "acc.routine"}>
      gpu.return
    }
  }
}
