// RUN: mlir-opt %s -transform-interpreter -split-input-file -verify-diagnostics

// CHECK-LABEL: @set_anchor_layout_not_anchor_op
func.func @set_anchor_layout_not_anchor_op(%arg0: memref<4096x4096xf16>) {
  %0 = xegpu.create_nd_tdesc %arg0 : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x32xf16>
  %1 = xegpu.load_nd %0[0, 0]  : !xegpu.tensor_desc<256x32xf16> -> vector<256x32xf16>
  %2 = arith.extf %1 : vector<256x32xf16> to vector<256x32xf32> // expected-note {{target op}}
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["arith.extf"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error@below {{Cannot set anchor layout to op: arith.extf}}
    transform.xegpu.set_anchor_layout %0 sg_layout = [8, 4] sg_data = [32, 64] : !transform.any_op
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

// CHECK-LABEL: @insert_prefetch_loop_carried_offset
func.func @insert_prefetch_loop_carried_offset(%a: memref<4096x4096xf16>) {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  %desc = xegpu.create_nd_tdesc %a
    : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x32xf16>
  %final = scf.for %i = %c0 to %c128 step %c32
      iter_args(%offset = %c0) -> (index) {
    %tile = xegpu.load_nd %desc[0, %offset]
      : !xegpu.tensor_desc<256x32xf16> -> vector<256x32xf16>
    %next = arith.addi %offset, %c32 : index
    scf.yield %next : index
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %load = transform.structured.match ops{["xegpu.load_nd"]} in %root
      : (!transform.any_op) -> !transform.any_op
    // expected-error@below {{Load offset depends on a value unavailable before the loop.}}
    %prefetch_desc = transform.xegpu.insert_prefetch %load nb_prefetch = 1
      : (!transform.any_op) -> !transform.any_op
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
    %2 = transform.xegpu.get_load_op %1 : (!transform.any_value) -> !transform.any_op
    // expected-error@below {{Load op is not contained in a scf.for loop.}}
    %3 = transform.xegpu.insert_prefetch %2 nb_prefetch = 1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
