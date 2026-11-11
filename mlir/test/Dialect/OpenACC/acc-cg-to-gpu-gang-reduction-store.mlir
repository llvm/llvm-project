// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-cg-to-gpu))" --split-input-file | FileCheck %s

// A gang-only reduction stored to memory outside the kernel runs the store
// once per block, so it must become a cross-block atomic. The identity is
// stored by a launch ahead of the kernel, ordered before every atomic.

// CHECK-LABEL: func.func @gang_reduction_store
// CHECK-SAME:    %{{.*}}: memref<3xi32>, %[[RES:.*]]: memref<i32>
// CHECK:       gpu.launch
// CHECK:         memref.store %{{.*}}, %[[RES]][] : memref<i32>
// CHECK:         gpu.terminator
// CHECK:       gpu.launch
// CHECK-NOT:     memref.store %{{.*}}, %[[RES]]
// CHECK:         acc.atomic.update %[[RES]] : memref<i32> {
// CHECK-NEXT:    ^bb0(%[[ARG:.*]]: i32):
// CHECK-NEXT:      arith.addi %{{.*}}, %[[ARG]]
// CHECK-NOT:     memref.store %{{.*}}, %[[RES]]
// CHECK:         gpu.terminator

module attributes {gpu.container_module} {
  gpu.module @cuda_device_mod {
    gpu.func @gang_reduction_store_kernel() kernel {
      gpu.return
    }
  }

  func.func @gang_reduction_store(%arg_in: memref<3xi32>, %arg_res: memref<i32>) {
    %bx = acc.par_width par_dim(#acc.par_dim<block_x>)
    acc.compute_region launch(%kbx = %bx) ins(%a_in = %arg_in, %a_res = %arg_res) : (memref<3xi32>, memref<i32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      %c0_i32 = arith.constant 0 : i32
      %acc = memref.alloca() : memref<i32>
      scf.parallel (%bx_iv) = (%c0) to (%kbx) step (%c1) {
        %red = scf.parallel (%i) = (%bx_iv) to (%c3) step (%kbx) init (%c0_i32) -> i32 {
          %v = memref.load %a_in[%i] : memref<3xi32>
          scf.reduce(%v : i32) {
          ^bb0(%lhs: i32, %rhs: i32):
            %sum = arith.addi %lhs, %rhs : i32
            scf.reduce.return %sum : i32
          }
        } {acc.par_dims = #acc<par_dims[sequential]>}
        acc.reduction_accumulate %red to %acc <add> par_dims(#acc<par_dims[block_x]>) : i32 -> memref<i32>
        scf.reduce
      } {acc.par_dims = #acc<par_dims[block_x]>}
      %r = memref.load %acc[] : memref<i32>
      acc.predicate_region {
        memref.store %r, %a_res[] : memref<i32>
      }
      acc.yield
    } <{kernel_func_name = @gang_reduction_store_kernel, kernel_module_name = @cuda_device_mod, origin = "acc.kernels"}>
    return
  }
}


// -----

// The destination index depends on the grid size, which the launch defines, so
// the identity is stored inside the kernel rather than from a launch ahead.

// CHECK-LABEL: func.func @gang_reduction_store_grid_index
// CHECK-NOT:   gpu.grid_dim
// CHECK:       gpu.launch {{.*}}function(@gang_reduction_store_grid_index_kernel)
// CHECK:         gpu.block_id x
// CHECK:         scf.if
// CHECK:           memref.store
// CHECK:         gpu.barrier
// CHECK:         acc.atomic.update

module attributes {gpu.container_module} {
  gpu.module @cuda_device_mod {
    gpu.func @gang_reduction_store_grid_index_kernel() kernel {
      gpu.return
    }
  }

  func.func @gang_reduction_store_grid_index(%arg_in: memref<3xi32>, %arg_res: memref<8xi32>) {
    %bx = acc.par_width par_dim(#acc.par_dim<block_x>)
    acc.compute_region launch(%kbx = %bx) ins(%a_in = %arg_in, %a_res = %arg_res) : (memref<3xi32>, memref<8xi32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      %c0_i32 = arith.constant 0 : i32
      %acc = memref.alloca() : memref<i32>
      scf.parallel (%bx_iv) = (%c0) to (%kbx) step (%c1) {
        %red = scf.parallel (%i) = (%bx_iv) to (%c3) step (%kbx) init (%c0_i32) -> i32 {
          %v = memref.load %a_in[%i] : memref<3xi32>
          scf.reduce(%v : i32) {
          ^bb0(%lhs: i32, %rhs: i32):
            %sum = arith.addi %lhs, %rhs : i32
            scf.reduce.return %sum : i32
          }
        } {acc.par_dims = #acc<par_dims[sequential]>}
        acc.reduction_accumulate %red to %acc <add> par_dims(#acc<par_dims[block_x]>) : i32 -> memref<i32>
        scf.reduce
      } {acc.par_dims = #acc<par_dims[block_x]>}
      %r = memref.load %acc[] : memref<i32>
      %idx = arith.subi %kbx, %c1 : index
      acc.predicate_region {
        memref.store %r, %a_res[%idx] : memref<8xi32>
      }
      acc.yield
    } <{kernel_func_name = @gang_reduction_store_grid_index_kernel, kernel_module_name = @cuda_device_mod, origin = "acc.kernels"}>
    return
  }
}
