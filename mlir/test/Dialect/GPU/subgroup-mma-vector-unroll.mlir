// RUN: mlir-opt %s --transform-interpreter --split-input-file | FileCheck %s

func.func @matmul(%lhs: memref<32x32xf32>, %rhs: memref<32x32xf32>, %out: memref<32x32xf32>) {
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0.000000e+00> : vector<16x16xf32>
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %3 = gpu.thread_id  x
  %4 = gpu.thread_id  y
  %5 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%4]
  %6 = affine.apply affine_map<()[s0] -> ((s0 floordiv 32) * 16)>()[%3]
  // CHECK:         scf.for {{.*}} -> (vector<16x16xf32>) {
  // CHECK-COUNT-2:   vector.transfer_read {{.*}} vector<16x8xf32>
  // CHECK-COUNT-2:   vector.transfer_read {{.*}} vector<8x16xf32>
  // CHECK-COUNT-2:   vector.contract {{.*}} vector<16x8xf32>, vector<8x16xf32> into vector<16x16xf32>
  // CHECK:           scf.yield {{.*}} : vector<16x16xf32>
  // CHECK:         }
  %7 = scf.for %arg0 = %c0 to %c32 step %c16 iter_args(%arg1 = %cst) -> (vector<16x16xf32>) {
    %10 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%c0)[%5]
    %11 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%c0)[%arg0]
    %12 = vector.transfer_read %lhs[%10, %11], %cst_0 {in_bounds = [true, true]} : memref<32x32xf32>, vector<16x16xf32>
    %16 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%c0)[%6]
    %17 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%c0)[%arg0]
    %18 = vector.transfer_read %rhs[%17, %16], %cst_0 {in_bounds = [true, true]} : memref<32x32xf32>, vector<16x16xf32>
    %22 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %12, %18, %arg1 : vector<16x16xf32>, vector<16x16xf32> into vector<16x16xf32>
    scf.yield %22 : vector<16x16xf32>
  }
  %8 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%c0)[%5]
  %9 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%c0)[%6]
  vector.transfer_write %7, %out[%8, %9] {in_bounds = [true, true]} : vector<16x16xf32>, memref<32x32xf32>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op {
      transform.apply_patterns.gpu.unroll_vectors_subgroup_mma [16, 16, 8]
    } : !transform.op<"func.func">
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @gathered_matmul
func.func @gathered_matmul(%lhs: memref<32x32xf32>, %rhs: memref<32x32xf32>, %out: memref<32x32xf32>) {
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0.000000e+00> : vector<16x16xf32>
  %cst_mask = arith.constant dense<true> : vector<4x4xi1>
  %cst_pt = arith.constant dense<0.000000e+00> : vector<4x4xf32>
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %cst_1 = arith.constant dense<[0, 1, 2, 3]> : vector<4xindex>
  %cst_2 = arith.constant dense<1> : vector<4x4xindex>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
  %3 = gpu.thread_id  x
  %4 = gpu.thread_id  y
  %5 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%4]
  %6 = affine.apply affine_map<()[s0] -> ((s0 floordiv 32) * 16)>()[%3]
  // CHECK:         scf.for {{.*}} -> (vector<16x16xf32>) {
  // CHECK:           arith.addi {{.*}} : vector<4xindex>
  // CHECK:           vector.gather {{.*}} : memref<32x32xf32>, vector<4x4xindex>, vector<4x4xi1>, vector<4x4xf32> into vector<4x4xf32>
  // CHECK-COUNT-8:   vector.transfer_read {{.*}} vector<8x4xf32>
  // CHECK-COUNT-4:   vector.transfer_read {{.*}} vector<4x16xf32>
  // CHECK-COUNT-8:   vector.contract {{.*}} vector<8x4xf32>, vector<4x16xf32> into vector<8x16xf32>
  // CHECK:           scf.yield {{.*}} : vector<16x16xf32>
  // CHECK:         }
  %7 = scf.for %arg0 = %c0 to %c32 step %c16 iter_args(%arg1 = %cst) -> (vector<16x16xf32>) {
    %10 = vector.broadcast %arg0 : index to vector<4xindex>
    %11 = arith.addi %10, %cst_1 : vector<4xindex>
    %12 = vector.broadcast %11 : vector<4xindex> to vector<4x4xindex>
    %13 = arith.addi %12, %cst_2 : vector<4x4xindex>
    %14 = vector.gather %lhs[%c0, %c0] [%13], %cst_mask, %cst_pt : memref<32x32xf32>, vector<4x4xindex>, vector<4x4xi1>, vector<4x4xf32> into vector<4x4xf32>
    vector.transfer_write %14, %alloc[%c0, %c0] {in_bounds = [true, true]} : vector<4x4xf32>, memref<32x32xf32>
    gpu.barrier
    %15 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%c0)[%5]
    %16 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%c0)[%arg0]
    %17 = vector.transfer_read %alloc[%15, %16], %cst_0 {in_bounds = [true, true]} : memref<32x32xf32>, vector<16x16xf32>
    %18 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%c0)[%6]
    %19 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%c0)[%arg0]
    %20 = vector.transfer_read %rhs[%19, %18], %cst_0 {in_bounds = [true, true]} : memref<32x32xf32>, vector<16x16xf32>
    %21 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %17, %20, %arg1 : vector<16x16xf32>, vector<16x16xf32> into vector<16x16xf32>
    scf.yield %21 : vector<16x16xf32>
  }
  %8 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%c0)[%5]
  %9 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%c0)[%6]
  vector.transfer_write %7, %out[%8, %9] {in_bounds = [true, true]} : vector<16x16xf32>, memref<32x32xf32>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op {
      transform.apply_patterns.gpu.unroll_vectors_subgroup_mma [8, 16, 4]
    } : !transform.op<"func.func">
    transform.yield
  }
}
