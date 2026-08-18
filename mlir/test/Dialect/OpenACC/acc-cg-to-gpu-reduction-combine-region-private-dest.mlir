// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-cg-to-gpu))" | FileCheck %s

// A gang-vector reduction with an inner seq-loop reduction stages the
// outer-scoped accumulator through an extra inner-loop private accumulator,
// producing an acc.reduction_combine_region that writes one thread-private
// accumulator into another while still carrying a block par_dim. That combine
// must lower to a plain load/combine/store, not acc.atomic.update.

// CHECK-LABEL: func.func @combine_region_private_dest
// CHECK:         gpu.launch
// CHECK:         %[[OUTER:.*]] = memref.alloca() : memref<i32>
// CHECK:         memref.load %[[OUTER]][] : memref<i32>
// CHECK-NEXT:    memref.load %{{.*}}[] : memref<i32>
// CHECK-NEXT:    arith.addi
// CHECK-NEXT:    memref.store %{{.*}}, %[[OUTER]][] : memref<i32>
// CHECK:         acc.atomic.update %[[DEV:.*]] : memref<i32>
// CHECK-NOT:     acc.atomic.update

func.func @combine_region_private_dest(%arg: memref<i32>) {
  %c1_pw = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %bx = acc.par_width %c1_pw {par_dim = #acc.par_dim<block_x>}
  %tx = acc.par_width %c128 {par_dim = #acc.par_dim<thread_x>}
  %pv_outer = acc.privatize {acc.par_dims = #acc<par_dims[block_x, thread_x]>} : () -> !acc.private_type<memref<i32>>
  %pv_inner = acc.privatize {acc.par_dims = #acc<par_dims[block_x, thread_x]>} : () -> !acc.private_type<memref<i32>>
  acc.compute_region launch(%kbx = %bx, %ktx = %tx) ins(%a_res = %arg, %po = %pv_outer, %pi = %pv_inner) : (memref<i32>, !acc.private_type<memref<i32>>, !acc.private_type<memref<i32>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %outer = acc.private_local %po {acc.par_dims = #acc<par_dims[block_x, thread_x]>, acc.var_name = #acc.var_name<"r">} : (!acc.private_type<memref<i32>>) -> memref<i32>
    acc.predicate_region {
      memref.store %c0_i32, %outer[] : memref<i32>
    }
    scf.parallel (%bx_iv) = (%c0) to (%kbx) step (%c1) {
      scf.parallel (%tx_iv) = (%c0) to (%ktx) step (%c1) {
        %row_step = arith.muli %c1, %ktx : index
        %row_off = arith.muli %tx_iv, %c1 : index
        %row_stride = arith.muli %row_step, %kbx : index
        %row_block = arith.muli %bx_iv, %row_step : index
        %row_idx = arith.addi %row_off, %row_block : index
        scf.parallel (%i) = (%row_idx) to (%c8) step (%row_stride) {
          %inner = acc.private_local %pi {acc.par_dims = #acc<par_dims[block_x, thread_x]>, acc.var_name = #acc.var_name<"r">} : (!acc.private_type<memref<i32>>) -> memref<i32>
          acc.predicate_region {
            memref.store %c0_i32, %inner[] : memref<i32>
          }
          acc.predicate_region {
            memref.store %c1_i32, %inner[] : memref<i32>
          }
          %v = memref.load %inner[] : memref<i32>
          acc.reduction_accumulate %v to %inner <add> : i32 -> memref<i32> {par_dims = #acc<par_dims[block_x, thread_x]>}
          acc.predicate_region {
            acc.reduction_combine_region %inner into %outer : memref<i32> {
              %la = memref.load %outer[] : memref<i32>
              %lb = memref.load %inner[] : memref<i32>
              %s = arith.addi %la, %lb : i32
              memref.store %s, %outer[] : memref<i32>
              acc.yield
            } {acc.par_dims = #acc<par_dims[block_x, thread_x]>}
          }
          scf.reduce
        } {acc.par_dims = #acc<par_dims[sequential]>}
        %ov = memref.load %outer[] : memref<i32>
        acc.reduction_accumulate %ov to %outer <add> : i32 -> memref<i32> {par_dims = #acc<par_dims[block_x, thread_x]>}
        scf.reduce
      } {acc.par_dims = #acc<par_dims[thread_x]>}
      scf.reduce
    } {acc.par_dims = #acc<par_dims[block_x]>}
    acc.predicate_region {
      acc.reduction_combine_region %outer into %a_res : memref<i32> {
        %da = memref.load %a_res[] : memref<i32>
        %db = memref.load %outer[] : memref<i32>
        %ds = arith.addi %da, %db : i32
        memref.store %ds, %a_res[] : memref<i32>
        acc.yield
      } {acc.par_dims = #acc<par_dims[block_x, thread_x]>}
    }
    acc.yield
  } {origin = "acc.parallel"}
  return
}
