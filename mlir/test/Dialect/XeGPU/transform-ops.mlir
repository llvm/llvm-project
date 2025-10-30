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

// -----

// CHECK-LABEL: @set_op_layout_attr_result_default_index
func.func @set_op_layout_attr_result_default_index(%arg0: memref<4096x4096xf16>, %arg1: memref<4096x4096xf16>, %arg2: memref<4096x4096xf16>) {
  %0 = xegpu.create_nd_tdesc %arg0 : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x32xf16>
  %1 = xegpu.load_nd %0[0, 0] : !xegpu.tensor_desc<256x32xf16> -> vector<256x32xf16>
  %2 = xegpu.create_nd_tdesc %arg1 : memref<4096x4096xf16> -> !xegpu.tensor_desc<32x256xf16>
  %3 = xegpu.load_nd %2[0, 0]  : !xegpu.tensor_desc<32x256xf16> -> vector<32x256xf16>
  %4 = xegpu.create_nd_tdesc %arg2 : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x256xf16>
  %5 = xegpu.load_nd %4[0, 0]  : !xegpu.tensor_desc<256x256xf16> -> vector<256x256xf16>
  // CHECK: = xegpu.dpas
  // CHECK-SAME: {layout_result_0 = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 64], inst_data = [8, 16]>}
  %6 = xegpu.dpas %1, %3, %5 : vector<256x32xf16>, vector<32x256xf16>, vector<256x256xf16> -> vector<256x256xf16>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["xegpu.dpas"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // CHECK: transform.xegpu.set_op_layout_attr %{{.*}}
    transform.xegpu.set_op_layout_attr %0 result sg_layout = [8, 4] sg_data = [32, 64] inst_data = [8, 16] : !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: @set_op_layout_attr_result_sg_param
func.func @set_op_layout_attr_result_sg_param(%arg0: memref<4096x4096xf16>) {
  %0 = xegpu.create_nd_tdesc %arg0 : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x32xf16>
  %1 = xegpu.load_nd %0[0, 0]  : !xegpu.tensor_desc<256x32xf16> -> vector<256x32xf16>
  // CHECK: = arith.extf %1
  // CHECK-SAME: {layout_result_0 = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 64], inst_data = [8, 16]>}
  %2 = arith.extf %1 : vector<256x32xf16> to vector<256x32xf32>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["arith.extf"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // CHECK: transform.xegpu.set_op_layout_attr %{{.*}}
    %layout0 = transform.param.constant 8 : i64 -> !transform.param<i64>
    transform.xegpu.set_op_layout_attr %0 result sg_layout = [%layout0, 4] sg_data = [32, 64] inst_data = [8, 16] : !transform.any_op, !transform.param<i64>
    transform.yield
  }
}

// -----

// CHECK-LABEL: @set_op_layout_attr_result_sg_param2
func.func @set_op_layout_attr_result_sg_param2(%arg0: memref<4096x4096xf16>) {
  %0 = xegpu.create_nd_tdesc %arg0 : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x32xf16>
  %1 = xegpu.load_nd %0[0, 0]  : !xegpu.tensor_desc<256x32xf16> -> vector<256x32xf16>
  // CHECK: = arith.extf %1
  // CHECK-SAME: {layout_result_0 = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 64], inst_data = [8, 16]>}
  %2 = arith.extf %1 : vector<256x32xf16> to vector<256x32xf32>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["arith.extf"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // CHECK: transform.xegpu.set_op_layout_attr %{{.*}}
    %layout0 = transform.param.constant 8 : i64 -> !transform.param<i64>
    %layout1 = transform.param.constant 4 : i64 -> !transform.param<i64>
    transform.xegpu.set_op_layout_attr %0 result sg_layout = [%layout0, %layout1] sg_data = [32, 64] inst_data = [8, 16] : !transform.any_op, !transform.param<i64>, !transform.param<i64>
    transform.yield
  }
}

// -----

// CHECK-LABEL: @set_op_layout_attr_result0
func.func @set_op_layout_attr_result0(%arg0: memref<4096x4096xf16>) {
  %0 = xegpu.create_nd_tdesc %arg0 : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x32xf16>
  %1 = xegpu.load_nd %0[0, 0]  : !xegpu.tensor_desc<256x32xf16> -> vector<256x32xf16>
  // CHECK: = arith.extf %1
  // CHECK-SAME: {layout_result_0 = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 64], inst_data = [8, 16]>}
  %2 = arith.extf %1 : vector<256x32xf16> to vector<256x32xf32>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["arith.extf"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // CHECK: transform.xegpu.set_op_layout_attr %{{.*}}
    transform.xegpu.set_op_layout_attr %0 result index = 0 sg_layout = [8, 4] sg_data = [32, 64] inst_data = [8, 16] : !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: @set_op_layout_attr_operand_minimal
func.func @set_op_layout_attr_operand_minimal(%arg0: memref<4096x4096xf16>) {
  %0 = xegpu.create_nd_tdesc %arg0 : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x32xf16>
  %1 = xegpu.load_nd %0[0, 0]  : !xegpu.tensor_desc<256x32xf16> -> vector<256x32xf16>
  // CHECK: = arith.extf %1
  // CHECK-SAME: {layout_operand_0 = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 64]>}
  %2 = arith.extf %1 : vector<256x32xf16> to vector<256x32xf32>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["arith.extf"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // CHECK: transform.xegpu.set_op_layout_attr %{{.*}}
    transform.xegpu.set_op_layout_attr %0 sg_layout = [8, 4] sg_data = [32, 64] : !transform.any_op
    transform.yield
  }
}
// -----

// CHECK-LABEL: @set_op_layout_attr_operand1
func.func @set_op_layout_attr_operand1(%arg0: memref<4096x4096xf16>, %arg1: memref<4096x4096xf16>) {
  %0 = xegpu.create_nd_tdesc %arg0 : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x32xf16>
  %1 = xegpu.load_nd %0[0, 0]  : !xegpu.tensor_desc<256x32xf16> -> vector<256x32xf16>
  %2 = xegpu.create_nd_tdesc %arg1 : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x32xf16>
  %3 = xegpu.load_nd %2[0, 0]  : !xegpu.tensor_desc<256x32xf16> -> vector<256x32xf16>
  // CHECK: = arith.addf %1, %3
  // CHECK-SAME: {layout_operand_1 = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 64], inst_data = [8, 16]>}
  %6 = arith.addf %1, %3 : vector<256x32xf16>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["arith.addf"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // CHECK: transform.xegpu.set_op_layout_attr %{{.*}}
    transform.xegpu.set_op_layout_attr %0 index = 1 sg_layout = [8, 4] sg_data = [32, 64] inst_data = [8, 16] : !transform.any_op
    transform.yield
  }
}
