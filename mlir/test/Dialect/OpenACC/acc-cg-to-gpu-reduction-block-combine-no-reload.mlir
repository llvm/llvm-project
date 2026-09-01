// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-cg-to-gpu))" | FileCheck %s

// A [block_x, thread_x] reduction whose accumulator is a grid-shared global
// slot (not a per-thread alloca). The block-level reduction_combine must
// atomically add the block-reduced gpu.all_reduce value directly into the
// result, NOT reload the global slot first.

// CHECK-LABEL: func.func @test_block_combine_no_reload
// CHECK:       gpu.launch
// CHECK:       %[[RED:.*]] = gpu.all_reduce add
// CHECK-NEXT:  } {acc.par_dims = #acc<par_dims[block_x, thread_x]>}
// CHECK-NOT:   memref.load
// CHECK:       acc.atomic.update %[[RESULT:.*]] : memref<i32> {
// CHECK-NEXT:  ^bb0(%[[ARG:.*]]: i32):
// CHECK-NEXT:    %{{.*}} = arith.addi %[[RED]], %[[ARG]]

module attributes {gpu.container_module} {
  gpu.module @cuda_device_mod {
    gpu.func @test_block_combine_no_reload_kernel() kernel {
      gpu.return
    }
  }

  func.func @test_block_combine_no_reload(%arg_slot: memref<i32>, %arg_res: memref<i32>) {
    %c1_pw = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %bx = acc.par_width %c1_pw {par_dim = #acc.par_dim<block_x>}
    %tx = acc.par_width %c128 {par_dim = #acc.par_dim<thread_x>}
    acc.compute_region launch(%kbx = %bx, %ktx = %tx) ins(%a_slot = %arg_slot, %a_res = %arg_res) : (memref<i32>, memref<i32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      scf.parallel (%bx_iv) = (%c0) to (%kbx) step (%c1) {
        scf.parallel (%tx_iv) = (%c0) to (%ktx) step (%c1) {
          %row_step = arith.muli %c1, %ktx : index
          %row_start = arith.muli %tx_iv, %c1 : index
          %row_off = arith.addi %c0, %row_start : index
          %row_stride = arith.muli %row_step, %kbx : index
          %row_block = arith.muli %bx_iv, %row_step : index
          %row_idx = arith.addi %row_off, %row_block : index
          %inner_red = scf.parallel (%i) = (%row_idx) to (%c8) step (%row_stride) init (%c0_i32) -> i32 {
            scf.reduce(%c1_i32 : i32) {
            ^bb0(%lhs: i32, %rhs: i32):
              %sum = arith.addi %lhs, %rhs : i32
              scf.reduce.return %sum : i32
            }
          } {acc.par_dims = #acc<par_dims[sequential]>}
          acc.reduction_accumulate %inner_red to %a_slot <add> : i32 -> memref<i32> {par_dims = #acc<par_dims[block_x, thread_x]>}
          scf.reduce
        } {acc.par_dims = #acc<par_dims[thread_x]>}
        scf.reduce
      } {acc.par_dims = #acc<par_dims[block_x]>}
      acc.predicate_region {
        acc.reduction_combine %a_slot into %a_res <add> : memref<i32> {acc.par_dims = #acc<par_dims[block_x, thread_x]>}
      }
      acc.yield
    } {kernel_func_name = @test_block_combine_no_reload_kernel, kernel_module_name = @cuda_device_mod, origin = "acc.parallel"}
    return
  }
}
