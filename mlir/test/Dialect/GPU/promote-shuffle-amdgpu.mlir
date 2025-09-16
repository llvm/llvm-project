// RUN: mlir-opt --transform-interpreter --split-input-file %s | FileCheck %s

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %module_op : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.gpu.gpu_shuffle_to_amdgpu chipset = "gfx950"
    } : !transform.any_op
    transform.yield
  }
}

  // CHECK-LABEL: func @gpu_shuffle_swizzle
  //  CHECK-SAME: (%[[ARG:.*]]: i32)
func.func @gpu_shuffle_swizzle(%arg0: i32) -> (i32, i1) {
  // CHECK:  %[[TRUE:.*]] = arith.constant true
  // CHECK:  %[[RES:.*]] = amdgpu.swizzle_bitmode %[[ARG]] 31 0 23 : i32
  // CHECK:  return %[[RES]], %[[TRUE]] : i32, i1
  %width = arith.constant 64 : i32
  %offset = arith.constant 23 : i32
  %shfl, %pred = gpu.shuffle xor %arg0, %offset, %width : i32
  func.return %shfl, %pred : i32, i1
}

  // CHECK-LABEL: func @gpu_shuffle_permlane_swap
  //  CHECK-SAME: (%[[ARG:.*]]: i32)
func.func @gpu_shuffle_permlane_swap(%arg0: i32) -> (i32, i1) {
  // CHECK:  %[[TRUE:.*]] = arith.constant true
  // CHECK:  %[[RES:.*]] = amdgpu.permlane_swap %[[ARG]] 32 : i32
  // CHECK:  return %[[RES]], %[[TRUE]] : i32, i1
  %width = arith.constant 64 : i32
  %offset = arith.constant 32 : i32
  %shfl, %pred = gpu.shuffle xor %arg0, %offset, %width : i32
  func.return %shfl, %pred : i32, i1
}
