// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// RUN: mlir-opt %s --mlir-print-op-generic | mlir-opt | FileCheck %s

// CHECK-LABEL: test_nd_range
func.func @test_nd_range(%dim: i32) {
  %0 = gen.local_id %dim
  %1 = gen.work_group_id %dim
  %2 = gen.work_group_size %dim
  %3 = gen.num_work_groups %dim
  return
}

// CHECK-LABEL: test_barrier
func.func @test_barrier() {
  gen.barrier
  return
}

// CHECK-LABEL: test_sub_group_shuffle
func.func @test_sub_group_shuffle(%arg0: i32, %arg1: i64, %arg2: f32, %arg3: f64, %arg4: i32) {
  %0 = gen.sub_group_shuffle xor %arg0, %arg4 : i32
  %1 = gen.sub_group_shuffle up %arg1, %arg4 : i64
  %2 = gen.sub_group_shuffle down %arg2, %arg4 : f32
  %3 = gen.sub_group_shuffle idx %arg3, %arg4 : f64
  return
}
