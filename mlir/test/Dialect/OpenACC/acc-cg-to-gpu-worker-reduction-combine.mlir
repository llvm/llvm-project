// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-cg-to-gpu))" | FileCheck %s

// A worker-private reduction has one shared slot per ThreadY row. The combine
// must keep ThreadY active and predicate only ThreadX.

// CHECK-LABEL: func.func @worker_reduction_combine
// CHECK: gpu.launch {{.*}} threads([[TID_X:%[^,]+]], [[TID_Y:%[^,]+]],
// CHECK-NOT: arith.cmpi eq, [[TID_Y]]
// CHECK: %[[IS_X_ZERO:.*]] = arith.cmpi eq, [[TID_X]],
// CHECK-NOT: arith.andi
// CHECK: scf.if %[[IS_X_ZERO]]
// CHECK: acc.atomic.update

func.func @worker_reduction_combine(%result: memref<i32>) {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c32 = arith.constant 32 : index
  %block_y = acc.par_width %c1 {par_dim = #acc.par_dim<block_y>}
  %thread_y = acc.par_width %c4 {par_dim = #acc.par_dim<thread_y>}
  %thread_x = acc.par_width %c32 {par_dim = #acc.par_dim<thread_x>}
  acc.kernel_environment {
    %private = acc.privatize [#acc<par_dims[thread_y]>]
        : () -> !acc.private_type<memref<i32>>
    acc.compute_region launch(%by = %block_y, %ty = %thread_y, %tx = %thread_x)
        ins(%private_arg = %private, %result_arg = %result)
        : (!acc.private_type<memref<i32>>, memref<i32>) {
      %c0 = arith.constant 0 : index
      %c1_inner = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      scf.parallel (%block_iv) = (%c0) to (%by) step (%c1_inner) {
        %local = acc.private_local %private_arg
            : (!acc.private_type<memref<i32>>) -> memref<i32>
        scf.parallel (%worker_iv) = (%c0) to (%ty) step (%c1_inner) {
          memref.store %c0_i32, %local[] : memref<i32>
          scf.reduce
        } {acc.par_dims = #acc<par_dims[thread_y]>}
        acc.predicate_region {
          acc.reduction_combine %local into %result_arg <add> : memref<i32>
              {acc.par_dims = #acc<par_dims[block_y, thread_y]>}
        }
        scf.reduce
      } {acc.par_dims = #acc<par_dims[block_y]>}
      acc.yield
    } {origin = "acc.parallel"}
  }
  return
}

// CHECK-LABEL: func.func @worker_reduction_combine_region
// CHECK: gpu.launch {{.*}} threads([[REGION_TID_X:%[^,]+]], [[REGION_TID_Y:%[^,]+]],
// CHECK-NOT: arith.cmpi eq, [[REGION_TID_Y]]
// CHECK: %[[REGION_IS_X_ZERO:.*]] = arith.cmpi eq, [[REGION_TID_X]],
// CHECK-NOT: arith.andi
// CHECK: scf.if %[[REGION_IS_X_ZERO]]
// CHECK: arith.addi

func.func @worker_reduction_combine_region(%result: memref<i32>) {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c32 = arith.constant 32 : index
  %block_y = acc.par_width %c1 {par_dim = #acc.par_dim<block_y>}
  %thread_y = acc.par_width %c4 {par_dim = #acc.par_dim<thread_y>}
  %thread_x = acc.par_width %c32 {par_dim = #acc.par_dim<thread_x>}
  acc.kernel_environment {
    %private = acc.privatize [#acc<par_dims[thread_y]>]
        : () -> !acc.private_type<memref<i32>>
    acc.compute_region launch(%by = %block_y, %ty = %thread_y, %tx = %thread_x)
        ins(%private_arg = %private, %result_arg = %result)
        : (!acc.private_type<memref<i32>>, memref<i32>) {
      %c0 = arith.constant 0 : index
      %c1_inner = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      scf.parallel (%block_iv) = (%c0) to (%by) step (%c1_inner) {
        %local = acc.private_local %private_arg
            : (!acc.private_type<memref<i32>>) -> memref<i32>
        scf.parallel (%worker_iv) = (%c0) to (%ty) step (%c1_inner) {
          memref.store %c0_i32, %local[] : memref<i32>
          scf.reduce
        } {acc.par_dims = #acc<par_dims[thread_y]>}
        acc.predicate_region {
          acc.reduction_combine_region %local into %result_arg : memref<i32> {
            %lhs = memref.load %result_arg[] : memref<i32>
            %rhs = memref.load %local[] : memref<i32>
            %sum = arith.addi %lhs, %rhs : i32
            memref.store %sum, %result_arg[] : memref<i32>
            acc.yield
          } {acc.par_dims = #acc<par_dims[block_y, thread_y]>}
        }
        scf.reduce
      } {acc.par_dims = #acc<par_dims[block_y]>}
      acc.yield
    } {origin = "acc.parallel"}
  }
  return
}
