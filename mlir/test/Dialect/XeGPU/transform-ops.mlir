// RUN: mlir-opt %s -transform-interpreter -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: @set_desc_layout
func.func @set_desc_layout(%arg0: memref<4096x4096xf16>) {
  // CHECK: %[[V0:.+]] = xegpu.create_nd_tdesc %arg0
  // CHECK-SAME: #xegpu.block_tdesc_attr<boundary_check = false>
  // CHECK-SAME: #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], inst_data = [8, 16]>>
  %0 = xegpu.create_nd_tdesc %arg0 : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x32xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["xegpu.create_nd_tdesc"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // CHECK: transform.xegpu.set_desc_layout %{{.*}}
    %1 = transform.xegpu.set_desc_layout %0 sg_layout = [8, 4] sg_data = [32, 32] inst_data = [8, 16] : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: @set_desc_layout_minimal
func.func @set_desc_layout_minimal(%arg0: memref<4096x4096xf16>) {
  // CHECK: %[[V0:.+]] = xegpu.create_nd_tdesc %arg0
  // CHECK-SAME: #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32]>>
  %0 = xegpu.create_nd_tdesc %arg0 : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x32xf16>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["xegpu.create_nd_tdesc"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // CHECK: transform.xegpu.set_desc_layout %{{.*}}
    %1 = transform.xegpu.set_desc_layout %0 sg_layout = [8, 4] sg_data = [32, 32] : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: @set_desc_layout_param
func.func @set_desc_layout_param(%arg0: memref<4096x4096xf16>) {
  // CHECK: %[[V0:.+]] = xegpu.create_nd_tdesc %arg0
  // CHECK-SAME: #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], inst_data = [8, 16]>>
  %0 = xegpu.create_nd_tdesc %arg0 : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x32xf16>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["xegpu.create_nd_tdesc"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // CHECK: transform.xegpu.set_desc_layout %{{.*}}
    %layout0 = transform.param.constant 8 : i64 -> !transform.param<i64>
    %1 = transform.xegpu.set_desc_layout %0 sg_layout = [%layout0, 4] sg_data = [32, 32] inst_data = [8, 16] : (!transform.any_op, !transform.param<i64>) -> !transform.any_op
    transform.yield
  }
}
