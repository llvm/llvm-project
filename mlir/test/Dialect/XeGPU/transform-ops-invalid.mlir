// RUN: mlir-opt %s -transform-interpreter -split-input-file -verify-diagnostics

func.func @set_desc_layout(%arg0: memref<4096x4096xf16>) {
  %c32 = arith.constant 32 : index // expected-note {{target op}}
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["arith.constant"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error@below {{Expected a xegpu.create_nd_desc op, but got: arith.constant}}
    %1 = transform.xegpu.set_desc_layout %0 sg_layout = [8, 4] sg_data = [32, 32] : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: @set_op_layout_attr_bad_result_index
func.func @set_op_layout_attr_bad_result_index(%arg0: memref<4096x4096xf16>) {
  %0 = xegpu.create_nd_tdesc %arg0 : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x32xf16>
  %1 = xegpu.load_nd %0[0, 0]  : !xegpu.tensor_desc<256x32xf16> -> vector<256x32xf16>
  %2 = arith.extf %1 : vector<256x32xf16> to vector<256x32xf32>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["arith.extf"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error@below {{Index exceeds the number of op results}}
    transform.xegpu.set_op_layout_attr %0 result index = 1 sg_layout = [8, 4] sg_data = [32, 64] : !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: @set_op_layout_attr_bad_operand_index
func.func @set_op_layout_attr_bad_operand_index(%arg0: memref<4096x4096xf16>) {
  %0 = xegpu.create_nd_tdesc %arg0 : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x32xf16>
  %1 = xegpu.load_nd %0[0, 0]  : !xegpu.tensor_desc<256x32xf16> -> vector<256x32xf16>
  %2 = arith.extf %1 : vector<256x32xf16> to vector<256x32xf32>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["arith.extf"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error@below {{Index exceeds the number of op operands}}
    transform.xegpu.set_op_layout_attr %0 index = 1 sg_layout = [8, 4] sg_data = [32, 64] : !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: @set_op_layout_attr_multiple
func.func @set_op_layout_attr_multiple(%arg0: memref<4096x4096xf16>) {
  %0 = xegpu.create_nd_tdesc %arg0 : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x32xf16>
  %1 = xegpu.load_nd %0[0, 0]  : !xegpu.tensor_desc<256x32xf16> -> vector<256x32xf16>
  %2 = arith.extf %1 : vector<256x32xf16> to vector<256x32xf32>
  %3 = arith.extf %2 : vector<256x32xf32> to vector<256x32xf64>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["arith.extf"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error@below {{Requires exactly one targetOp handle (got 2)}}
    transform.xegpu.set_op_layout_attr %0 sg_layout = [8, 4] sg_data = [32, 64] : !transform.any_op
    transform.yield
  }
}

// -----

func.func @set_gpu_launch_threads_bad_handle(%arg0: memref<4096x4096xf16>) {
  %c32 = arith.constant 32 : index // expected-note {{target op}}
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["arith.constant"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error@below {{Expected a gpu.launch op, but got: arith.constant}}
    transform.xegpu.set_gpu_launch_threads %0 threads = [8, 4, 1] : !transform.any_op
    transform.yield
  }
}

// -----

func.func @set_gpu_launch_threads_many_handles(%arg0: memref<4096x4096xf16>) {
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["arith.constant"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error@below {{Requires exactly one targetOp handle (got 2)}}
    transform.xegpu.set_gpu_launch_threads %0 threads = [8, 4, 1] : !transform.any_op
    transform.yield
  }
}

// -----

func.func @set_gpu_launch_threads_bad_threads(%arg0: memref<4096x4096xf16>) {
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %c16, %arg10 = %c16, %arg11 = %c1) threads(%arg6, %arg7, %arg8) in (%arg12 = %c1, %arg13 = %c1, %arg14 = %c1) {
    gpu.terminator
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["gpu.launch"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error@below {{Expected threads argument to consist of three values (got 2)}}
    transform.xegpu.set_gpu_launch_threads %0 threads = [8, 4] : !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: @insert_prefetch_dpas_c
func.func @insert_prefetch_dpas_c(%arg0: memref<4096x4096xf16>, %arg1: memref<4096x4096xf16>, %arg2: memref<4096x4096xf16>) {
  %c32 = arith.constant 32 : index
  %c4096 = arith.constant 4096 : index
  %c0 = arith.constant 0 : index
  %0 = xegpu.create_nd_tdesc %arg2 : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x256xf16>
  // expected-note@below {{load op}}
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
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["xegpu.dpas"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_operand %0[2] : (!transform.any_op) -> !transform.any_value
    // expected-error@below {{Load op is not contained in a scf.for loop.}}
    %2 = transform.xegpu.insert_prefetch %1 nb_prefetch = 1 : (!transform.any_value) -> !transform.any_op
    transform.yield
  }
}
