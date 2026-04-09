// RUN: mlir-opt %s -transform-interpreter -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: @get_load_op
func.func @get_load_op(%arg0: memref<4096x4096xf16>) {
  %c0 = arith.constant 0 : index
  %0 = xegpu.create_nd_tdesc %arg0 : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x32xf16>
  // CHECK: xegpu.load_nd
  // expected-remark @below {{found load_nd op}}
  %1 = xegpu.load_nd %0[%c0, %c0]  : !xegpu.tensor_desc<256x32xf16> -> vector<256x32xf16>
  %2 = arith.extf %1 : vector<256x32xf16> to vector<256x32xf32>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["arith.extf"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_operand %0[0] : (!transform.any_op) -> !transform.any_value
    %2 = transform.xegpu.get_load_op %1 : (!transform.any_value) -> !transform.any_op
    transform.debug.emit_remark_at %2, "found load_nd op" : !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: @get_load_op_c
func.func @get_load_op_c(%arg0: memref<4096x4096xf16>, %arg1: memref<4096x4096xf16>, %arg2: memref<4096x4096xf16>) {
  %c32 = arith.constant 32 : index
  %c4096 = arith.constant 4096 : index
  %c0 = arith.constant 0 : index
  %0 = xegpu.create_nd_tdesc %arg2 : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x256xf16>
  // expected-remark @below {{found load_nd op}}
  %1 = xegpu.load_nd %0[%c0, %c0]  : !xegpu.tensor_desc<256x256xf16> -> vector<256x256xf16>
  %3 = xegpu.create_nd_tdesc %arg0 : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x32xf16>
  %4 = xegpu.create_nd_tdesc %arg1 : memref<4096x4096xf16> -> !xegpu.tensor_desc<32x256xf16>
  %2 = scf.for %arg3 = %c0 to %c4096 step %c32 iter_args(%arg4 = %1) -> (vector<256x256xf16>) {
    %5 = xegpu.load_nd %3[%c0, %arg3] : !xegpu.tensor_desc<256x32xf16> -> vector<256x32xf16>
    %6 = xegpu.load_nd %4[%arg3, %c0] : !xegpu.tensor_desc<32x256xf16> -> vector<32x256xf16>
    %7 = xegpu.dpas %5, %6, %arg4 : vector<256x32xf16>, vector<32x256xf16>, vector<256x256xf16> -> vector<256x256xf16>
    scf.yield %7 : vector<256x256xf16>
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["xegpu.dpas"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_operand %0[2] : (!transform.any_op) -> !transform.any_value
    %2 = transform.xegpu.get_load_op %1 : (!transform.any_value) -> !transform.any_op
    transform.debug.emit_remark_at %2, "found load_nd op" : !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: @get_load_op_1d
func.func @get_load_op_1d(%arg0: memref<4096xf32>) {
  %cst = arith.constant dense<true> : vector<256xi1>
  %0 = vector.step : vector<256xindex>
  %intptr = memref.extract_aligned_pointer_as_index %arg0 : memref<4096xf32> -> index
  %1 = arith.index_cast %intptr : index to i64
  // CHECK: xegpu.load %1[%0]
  // expected-remark @below {{found load op}}
  %2 = xegpu.load %1[%0], %cst : i64, vector<256xindex>, vector<256xi1> -> vector<256xf32>
  %3 = arith.extf %2 : vector<256xf32> to vector<256xf64>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["arith.extf"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_operand %0[0] : (!transform.any_op) -> !transform.any_value
    %2 = transform.xegpu.get_load_op %1 : (!transform.any_value) -> !transform.any_op
    transform.debug.emit_remark_at %2, "found load op" : !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: @set_anchor_layout
func.func @set_anchor_layout(%arg0: memref<4096x4096xf16>) {
  %0 = xegpu.create_nd_tdesc %arg0 : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x32xf16>
  // CHECK: = xegpu.load_nd %0[0, 0]
  // CHECK-SAME: <{layout = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 64], inst_data = [8, 16]>}>
  %1 = xegpu.load_nd %0[0, 0]  : !xegpu.tensor_desc<256x32xf16> -> vector<256x32xf16>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["xegpu.load_nd"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // CHECK: transform.xegpu.set_anchor_layout %{{.*}}
    transform.xegpu.set_anchor_layout %0 index = 0 sg_layout = [8, 4] sg_data = [32, 64] inst_data = [8, 16] : !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: @set_anchor_layout_multiple
func.func @set_anchor_layout_multiple(%arg0: memref<4096x4096xf16>) {
  %0 = xegpu.create_nd_tdesc %arg0 : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x32xf16>
  // CHECK: xegpu.prefetch_nd %0[0, 0]
  // CHECK-SAME: <{layout = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 64], inst_data = [8, 16]>}>
  // CHECK: xegpu.prefetch_nd %0[16, 0]
  // CHECK-SAME: <{layout = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 64], inst_data = [8, 16]>}>
  xegpu.prefetch_nd %0[0, 0] : !xegpu.tensor_desc<256x32xf16>
  xegpu.prefetch_nd %0[16, 0] : !xegpu.tensor_desc<256x32xf16>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["xegpu.prefetch_nd"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // CHECK: transform.xegpu.set_anchor_layout %{{.*}}
    transform.xegpu.set_anchor_layout %0 index = 0 sg_layout = [8, 4] sg_data = [32, 64] inst_data = [8, 16] : !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: @set_anchor_layout_param
func.func @set_anchor_layout_param(%arg0: memref<4096x4096xf16>) {
  %0 = xegpu.create_nd_tdesc %arg0 : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x32xf16>
  // CHECK: = xegpu.load_nd %0[0, 0]
  // CHECK-SAME: <{layout = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 64], inst_data = [8, 16]>}>
  %1 = xegpu.load_nd %0[0, 0]  : !xegpu.tensor_desc<256x32xf16> -> vector<256x32xf16>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["xegpu.load_nd"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // CHECK: transform.xegpu.set_anchor_layout %{{.*}}
    %layout0 = transform.param.constant 8 : i64 -> !transform.param<i64>
    transform.xegpu.set_anchor_layout %0 index = 0 sg_layout = [%layout0, 4] sg_data = [32, 64] inst_data = [8, 16] : !transform.any_op, !transform.param<i64>
    transform.yield
  }
}

// -----

// CHECK-LABEL: @set_anchor_layout_param2
func.func @set_anchor_layout_param2(%arg0: memref<4096x4096xf16>) {
  %0 = xegpu.create_nd_tdesc %arg0 : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x32xf16>
  // CHECK: = xegpu.load_nd %0[0, 0]
  // CHECK-SAME: <{layout = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 64], inst_data = [8, 16]>}>
  %1 = xegpu.load_nd %0[0, 0]  : !xegpu.tensor_desc<256x32xf16> -> vector<256x32xf16>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["xegpu.load_nd"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // CHECK: transform.xegpu.set_anchor_layout %{{.*}}
    %layout0 = transform.param.constant 8 : i64 -> !transform.param<i64>
    %layout1 = transform.param.constant 4 : i64 -> !transform.param<i64>
    transform.xegpu.set_anchor_layout %0 index = 0 sg_layout = [%layout0, %layout1] sg_data = [32, 64] inst_data = [8, 16] : !transform.any_op, !transform.param<i64>, !transform.param<i64>
    transform.yield
  }
}

// -----

// CHECK-LABEL: @set_anchor_layout_slice
func.func @set_anchor_layout_slice(%arg0: memref<4096xf32>) {
  // CHECK: = xegpu.load %1[%0]
  // CHECK-SAME: <{layout = #xegpu.slice<#xegpu.layout<sg_layout = [8, 8], sg_data = [32, 32], inst_data = [8, 16]>, dims = [0]>}>
  %cst = arith.constant dense<true> : vector<256xi1>
  %0 = vector.step : vector<256xindex>
  %intptr = memref.extract_aligned_pointer_as_index %arg0 : memref<4096xf32> -> index
  %1 = arith.index_cast %intptr : index to i64
  %2 = xegpu.load %1[%0], %cst : i64, vector<256xindex>, vector<256xi1> -> vector<256xf32>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["xegpu.load"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // CHECK: transform.xegpu.set_anchor_layout %{{.*}}
    transform.xegpu.set_anchor_layout %0 sg_layout = [8, 8] sg_data = [32, 32] inst_data = [8, 16] slice_dims = [0] : !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: @set_anchor_layout_order
func.func @set_anchor_layout_order(%arg0: memref<4096x4096xf16>) {
  %0 = xegpu.create_nd_tdesc %arg0 : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x32xf16>
  // CHECK: = xegpu.load_nd %0[0, 0]
  // CHECK-SAME: <{layout = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 64], inst_data = [8, 16], order = [1, 0]>}>
  %1 = xegpu.load_nd %0[0, 0]  : !xegpu.tensor_desc<256x32xf16> -> vector<256x32xf16>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["xegpu.load_nd"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // CHECK: transform.xegpu.set_anchor_layout %{{.*}}
    transform.xegpu.set_anchor_layout %0 index = 0 sg_layout = [8, 4] sg_data = [32, 64] inst_data = [8, 16] order = [1, 0] : !transform.any_op
    transform.yield
  }
}


// -----

// CHECK-LABEL: @set_anchor_layout_dpas_a
func.func @set_anchor_layout_dpas_a(%arg0: memref<4096x4096xf16>, %arg1: memref<4096x4096xf16>, %arg2: memref<4096x4096xf16>) {
  %0 = xegpu.create_nd_tdesc %arg0 : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x32xf16>
  %1 = xegpu.load_nd %0[0, 0] : !xegpu.tensor_desc<256x32xf16> -> vector<256x32xf16>
  %2 = xegpu.create_nd_tdesc %arg1 : memref<4096x4096xf16> -> !xegpu.tensor_desc<32x256xf16>
  %3 = xegpu.load_nd %2[0, 0]  : !xegpu.tensor_desc<32x256xf16> -> vector<32x256xf16>
  %4 = xegpu.create_nd_tdesc %arg2 : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x256xf16>
  %5 = xegpu.load_nd %4[0, 0]  : !xegpu.tensor_desc<256x256xf16> -> vector<256x256xf16>
  // CHECK: = xegpu.dpas
  // CHECK-SAME: {layout_a = #xegpu.layout<sg_layout = [8, 8], sg_data = [32, 32], inst_data = [8, 16]>}
  %6 = xegpu.dpas %1, %3, %5 : vector<256x32xf16>, vector<32x256xf16>, vector<256x256xf16> -> vector<256x256xf16>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["xegpu.dpas"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // CHECK: transform.xegpu.set_anchor_layout %{{.*}}
    transform.xegpu.set_anchor_layout %0 index = 0 sg_layout = [8, 8] sg_data = [32, 32] inst_data = [8, 16] : !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: @set_anchor_layout_dpas_b
func.func @set_anchor_layout_dpas_b(%arg0: memref<4096x4096xf16>, %arg1: memref<4096x4096xf16>, %arg2: memref<4096x4096xf16>) {
  %0 = xegpu.create_nd_tdesc %arg0 : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x32xf16>
  %1 = xegpu.load_nd %0[0, 0] : !xegpu.tensor_desc<256x32xf16> -> vector<256x32xf16>
  %2 = xegpu.create_nd_tdesc %arg1 : memref<4096x4096xf16> -> !xegpu.tensor_desc<32x256xf16>
  %3 = xegpu.load_nd %2[0, 0]  : !xegpu.tensor_desc<32x256xf16> -> vector<32x256xf16>
  %4 = xegpu.create_nd_tdesc %arg2 : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x256xf16>
  %5 = xegpu.load_nd %4[0, 0]  : !xegpu.tensor_desc<256x256xf16> -> vector<256x256xf16>
  // CHECK: = xegpu.dpas
  // CHECK-SAME: {layout_b = #xegpu.layout<sg_layout = [8, 8], sg_data = [32, 32], inst_data = [16, 16]>}
  %6 = xegpu.dpas %1, %3, %5 : vector<256x32xf16>, vector<32x256xf16>, vector<256x256xf16> -> vector<256x256xf16>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["xegpu.dpas"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // CHECK: transform.xegpu.set_anchor_layout %{{.*}}
    transform.xegpu.set_anchor_layout %0 index = 1 sg_layout = [8, 8] sg_data = [32, 32] inst_data = [16, 16] : !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: @set_anchor_layout_dpas_c
func.func @set_anchor_layout_dpas_c(%arg0: memref<4096x4096xf16>, %arg1: memref<4096x4096xf16>, %arg2: memref<4096x4096xf16>) {
  %0 = xegpu.create_nd_tdesc %arg0 : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x32xf16>
  %1 = xegpu.load_nd %0[0, 0] : !xegpu.tensor_desc<256x32xf16> -> vector<256x32xf16>
  %2 = xegpu.create_nd_tdesc %arg1 : memref<4096x4096xf16> -> !xegpu.tensor_desc<32x256xf16>
  %3 = xegpu.load_nd %2[0, 0]  : !xegpu.tensor_desc<32x256xf16> -> vector<32x256xf16>
  %4 = xegpu.create_nd_tdesc %arg2 : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x256xf16>
  %5 = xegpu.load_nd %4[0, 0]  : !xegpu.tensor_desc<256x256xf16> -> vector<256x256xf16>
  // CHECK: = xegpu.dpas
  // CHECK-SAME: {layout_cd = #xegpu.layout<sg_layout = [8, 8], sg_data = [32, 32], inst_data = [8, 16]>}
  %6 = xegpu.dpas %1, %3, %5 : vector<256x32xf16>, vector<32x256xf16>, vector<256x256xf16> -> vector<256x256xf16>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["xegpu.dpas"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // CHECK: transform.xegpu.set_anchor_layout %{{.*}}
    transform.xegpu.set_anchor_layout %0 index = 2 sg_layout = [8, 8] sg_data = [32, 32] inst_data = [8, 16] : !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: @set_gpu_launch_threads
func.func @set_gpu_launch_threads(%arg0: memref<4096x4096xf16>) {
  // CHECK: %[[C1:.+]] = arith.constant 1 : index
  %c1 = arith.constant 1 : index
  // CHECK: %[[C16:.+]] = arith.constant 16 : index
  %c16 = arith.constant 16 : index
  // CHECK: %[[C8:.+]] = arith.constant 8 : index
  // CHECK: %[[C4:.+]] = arith.constant 4 : index
  // CHECK: %[[C1_0:.+]] = arith.constant 1 : index
  // CHECK: gpu.launch blocks(%{{.*}}, %{{.*}}, %{{.*}}) in (%{{.*}} = %[[C16]], %{{.*}} = %[[C16]], %{{.*}} = %[[C1]])
  // CHECK-SAME: threads(%{{.*}}, %{{.*}}, %{{.*}}) in (%{{.*}} = %[[C8]], %{{.*}} = %[[C4]], %{{.*}} = %[[C1_0]])
  gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %c16, %arg10 = %c16, %arg11 = %c1) threads(%arg6, %arg7, %arg8) in (%arg12 = %c1, %arg13 = %c1, %arg14 = %c1) {
    gpu.terminator
  }
  return
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["gpu.launch"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // CHECK: transform.xegpu.set_gpu_launch_threads %{{.*}}
    transform.xegpu.set_gpu_launch_threads %0 threads = [8, 4, 1] : !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: @set_gpu_launch_threads_param
func.func @set_gpu_launch_threads_param(%arg0: memref<4096x4096xf16>) {
  // CHECK: %[[C1:.+]] = arith.constant 1 : index
  %c1 = arith.constant 1 : index
  // CHECK: %[[C16:.+]] = arith.constant 16 : index
  %c16 = arith.constant 16 : index
  // CHECK: %[[C8:.+]] = arith.constant 8 : index
  // CHECK: %[[C4:.+]] = arith.constant 4 : index
  // CHECK: %[[C1_0:.+]] = arith.constant 1 : index
  // CHECK: gpu.launch blocks(%{{.*}}, %{{.*}}, %{{.*}}) in (%{{.*}} = %[[C16]], %{{.*}} = %[[C16]], %{{.*}} = %[[C1]])
  // CHECK-SAME: threads(%{{.*}}, %{{.*}}, %{{.*}}) in (%{{.*}} = %[[C8]], %{{.*}} = %[[C4]], %{{.*}} = %[[C1_0]])
  gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %c16, %arg10 = %c16, %arg11 = %c1) threads(%arg6, %arg7, %arg8) in (%arg12 = %c1, %arg13 = %c1, %arg14 = %c1) {
    gpu.terminator
  }
  return
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["gpu.launch"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // CHECK: transform.xegpu.set_gpu_launch_threads %{{.*}}
    %th1 = transform.param.constant 4 : i64 -> !transform.param<i64>
    transform.xegpu.set_gpu_launch_threads %0 threads = [8, %th1, 1] : !transform.any_op, !transform.param<i64>
    transform.yield
  }
}

// -----

// CHECK-LABEL: @insert_prefetch_dpas_a
func.func @insert_prefetch_dpas_a(%arg0: memref<4096x4096xf16>, %arg1: memref<4096x4096xf16>, %arg2: memref<4096x4096xf16>) {
  // CHECK: %[[C32:.+]] = arith.constant 32 : index
  %c32 = arith.constant 32 : index
  %c4096 = arith.constant 4096 : index
  // CHECK: %[[C0:.+]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  %0 = xegpu.create_nd_tdesc %arg2 : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x256xf16>
  %1 = xegpu.load_nd %0[%c0, %c0]  : !xegpu.tensor_desc<256x256xf16> -> vector<256x256xf16>
  // CHECK: xegpu.create_nd_tdesc %arg0
  // CHECK: xegpu.create_nd_tdesc %arg1
  // CHECK: %[[V0:.+]] = xegpu.create_nd_tdesc %arg0
  // CHECK-SAME: !xegpu.tensor_desc<256x32xf16
  // CHECK: xegpu.prefetch_nd %[[V0]][%[[C0]], %[[C0]]]
  %3 = xegpu.create_nd_tdesc %arg0 : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x32xf16>
  %4 = xegpu.create_nd_tdesc %arg1 : memref<4096x4096xf16> -> !xegpu.tensor_desc<32x256xf16>
  // CHECK: scf.for %[[ARG3:.+]] = %[[C0]]
  %2 = scf.for %arg3 = %c0 to %c4096 step %c32 iter_args(%arg4 = %1) -> (vector<256x256xf16>) {
    // CHECK: %[[ADD:.+]] = arith.addi %[[ARG3]], %[[C32]]
    // CHECK: xegpu.prefetch_nd %[[V0]][%[[C0]], %[[ADD]]]
    %5 = xegpu.load_nd %3[%c0, %arg3] : !xegpu.tensor_desc<256x32xf16> -> vector<256x32xf16>
    %6 = xegpu.load_nd %4[%arg3, %c0] : !xegpu.tensor_desc<32x256xf16> -> vector<32x256xf16>
    %7 = xegpu.dpas %5, %6, %arg4 : vector<256x32xf16>, vector<32x256xf16>, vector<256x256xf16> -> vector<256x256xf16>
    scf.yield %7 : vector<256x256xf16>
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %0 = transform.structured.match ops{["xegpu.dpas"]} in %func : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_operand %0[0] : (!transform.any_op) -> !transform.any_value
    %2 = transform.xegpu.get_load_op %1 : (!transform.any_value) -> !transform.any_op
    // CHECK: transform.xegpu.insert_prefetch %{{.*}}
    %3 = transform.xegpu.insert_prefetch %2 nb_prefetch = 1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op

    transform.yield
  }
}

// -----

// CHECK-LABEL: @insert_prefetch_dpas_a_nb_param2
func.func @insert_prefetch_dpas_a_nb_param2(%arg0: memref<4096x4096xf16>, %arg1: memref<4096x4096xf16>, %arg2: memref<4096x4096xf16>) {
  // CHECK: %[[C64:.+]] = arith.constant 64 : index
  // CHECK: %[[C32:.+]] = arith.constant 32 : index
  %c32 = arith.constant 32 : index
  %c4096 = arith.constant 4096 : index
  // CHECK: %[[C0:.+]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  %0 = xegpu.create_nd_tdesc %arg2 : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x256xf16>
  %1 = xegpu.load_nd %0[0, 0]  : !xegpu.tensor_desc<256x256xf16> -> vector<256x256xf16>
  // CHECK: xegpu.create_nd_tdesc %arg0
  // CHECK: xegpu.create_nd_tdesc %arg1
  // CHECK: %[[V0:.+]] = xegpu.create_nd_tdesc %arg0
  // CHECK-SAME: !xegpu.tensor_desc<256x32xf16
  // CHECK: xegpu.prefetch_nd %[[V0]][0, %[[C0]]]
  // CHECK: xegpu.prefetch_nd %[[V0]][0, %[[C32]]]
  %3 = xegpu.create_nd_tdesc %arg0 : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x32xf16>
  %4 = xegpu.create_nd_tdesc %arg1 : memref<4096x4096xf16> -> !xegpu.tensor_desc<32x256xf16>
  // CHECK: scf.for %[[ARG3:.+]] = %[[C0]]
  %2 = scf.for %arg3 = %c0 to %c4096 step %c32 iter_args(%arg4 = %1) -> (vector<256x256xf16>) {
    // CHECK: %[[ADD:.+]] = arith.addi %[[ARG3]], %[[C64]]
    // CHECK: xegpu.prefetch_nd %[[V0]][0, %[[ADD]]]
    %5 = xegpu.load_nd %3[0, %arg3] : !xegpu.tensor_desc<256x32xf16> -> vector<256x32xf16>
    %6 = xegpu.load_nd %4[%arg3, 0] : !xegpu.tensor_desc<32x256xf16> -> vector<32x256xf16>
    %7 = xegpu.dpas %5, %6, %arg4 : vector<256x32xf16>, vector<32x256xf16>, vector<256x256xf16> -> vector<256x256xf16>
    scf.yield %7 : vector<256x256xf16>
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %0 = transform.structured.match ops{["xegpu.dpas"]} in %func : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_operand %0[0] : (!transform.any_op) -> !transform.any_value
    %2 = transform.xegpu.get_load_op %1 : (!transform.any_value) -> !transform.any_op
    %nb = transform.param.constant 2 : i64 -> !transform.param<i64>
    // CHECK: transform.xegpu.insert_prefetch %{{.*}}
    %3 = transform.xegpu.insert_prefetch %2 nb_prefetch = %nb :  (!transform.any_op, !transform.param<i64>) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: @convert_layout_a
func.func @convert_layout_a(%arg0: memref<4096x4096xf16>, %arg1: memref<4096x4096xf16>, %arg2: memref<4096x4096xf16>) {
  %c0 = arith.constant 0 : index
  // CHECK: %[[V0:.+]] = xegpu.create_nd_tdesc %arg0
  %0 = xegpu.create_nd_tdesc %arg0 : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x32xf16, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], inst_data = [32, 16], order = [1, 0]>>
  // CHECK: %[[V1:.+]] = xegpu.load_nd %[[V0]]
  %1 = xegpu.load_nd %0[%c0, %c0]  : !xegpu.tensor_desc<256x32xf16, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], inst_data = [32, 16], order = [1, 0]>> -> vector<256x32xf16>
  // CHECK: %[[V2:.+]] = xegpu.convert_layout %[[V1]]
  // CHECK: input_layout = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], inst_data = [32, 16], order = [1, 0]>
  // CHECK: target_layout = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], inst_data = [8, 16], order = [1, 0]>
  %2 = xegpu.create_nd_tdesc %arg1 : memref<4096x4096xf16> -> !xegpu.tensor_desc<32x256xf16>
  %3 = xegpu.load_nd %2[%c0, %c0]  : !xegpu.tensor_desc<32x256xf16> -> vector<32x256xf16>
  %4 = xegpu.create_nd_tdesc %arg2 : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x256xf16>
  %5 = xegpu.load_nd %4[%c0, %c0]  : !xegpu.tensor_desc<256x256xf16> -> vector<256x256xf16>
  // CHECK: = xegpu.dpas %[[V2]]
  %6 = xegpu.dpas %1, %3, %5 : vector<256x32xf16>, vector<32x256xf16>, vector<256x256xf16> -> vector<256x256xf16>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["xegpu.dpas"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_operand %0[0] : (!transform.any_op) -> !transform.any_value
    // CHECK: transform.xegpu.convert_layout %{{.*}}
    transform.xegpu.convert_layout %1
      input_sg_layout = [8, 4] input_sg_data = [32, 32] input_inst_data = [32, 16] input_order = [1, 0]
      target_sg_layout = [8, 4] target_sg_data = [32, 32] target_inst_data = [8, 16] target_order = [1, 0]
      : (!transform.any_value) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: @convert_layout_a_sg_param
func.func @convert_layout_a_sg_param(%arg0: memref<4096x4096xf16>, %arg1: memref<4096x4096xf16>, %arg2: memref<4096x4096xf16>) {
  %c0 = arith.constant 0 : index
  // CHECK: %[[V0:.+]] = xegpu.create_nd_tdesc %arg0
  %0 = xegpu.create_nd_tdesc %arg0 : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x32xf16, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], inst_data = [32, 16]>>
  // CHECK: %[[V1:.+]] = xegpu.load_nd %[[V0]]
  %1 = xegpu.load_nd %0[%c0, %c0]  : !xegpu.tensor_desc<256x32xf16, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], inst_data = [32, 16]>> -> vector<256x32xf16>
  // CHECK: %[[V2:.+]] = xegpu.convert_layout %[[V1]]
  // CHECK: input_layout = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], inst_data = [32, 16]>
  // CHECK: target_layout = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], inst_data = [8, 16]>
  %2 = xegpu.create_nd_tdesc %arg1 : memref<4096x4096xf16> -> !xegpu.tensor_desc<32x256xf16>
  %3 = xegpu.load_nd %2[%c0, %c0]  : !xegpu.tensor_desc<32x256xf16> -> vector<32x256xf16>
  %4 = xegpu.create_nd_tdesc %arg2 : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x256xf16>
  %5 = xegpu.load_nd %4[%c0, %c0]  : !xegpu.tensor_desc<256x256xf16> -> vector<256x256xf16>
  // CHECK: = xegpu.dpas %[[V2]]
  %6 = xegpu.dpas %1, %3, %5 : vector<256x32xf16>, vector<32x256xf16>, vector<256x256xf16> -> vector<256x256xf16>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["xegpu.dpas"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_operand %0[0] : (!transform.any_op) -> !transform.any_value
    %layout0 = transform.param.constant 8 : i64 -> !transform.param<i64>
    // CHECK: transform.xegpu.convert_layout %{{.*}}
    transform.xegpu.convert_layout %1
      input_sg_layout = [%layout0, 4] input_sg_data = [32, 32] input_inst_data = [32, 16]
      target_sg_layout = [%layout0, 4] target_sg_data = [32, 32] target_inst_data = [8, 16]
      : (!transform.any_value, !transform.param<i64>, !transform.param<i64>) -> !transform.any_op
    transform.yield
  }
}
