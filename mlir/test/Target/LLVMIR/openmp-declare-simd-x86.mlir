// RUN: mlir-translate --mlir-to-llvmir %s | FileCheck %s
//
// This test exercises translation of `omp.declare_simd` from MLIR LLVM dialect
// to LLVM IR function attributes via llvm.
//
// For each `omp.declare_simd`, lowering computes:
//   1) ParamAttrs: one entry per function argument, classifying it as
//      Vector / Uniform / Linear (+ step or var-stride) / Aligned.
//   2) Branch kind: Undefined / Inbranch / Notinbranch.
//   3) VLEN: either from `simdlen(...)` or derived from the CDT size.
//
// llvm then emits x86 declare-simd variants by attaching
// mangled function attributes of the form:
//
//   _ZGV <ISA> <Mask> <VLEN> <ParamAttrs> _ <FunctionName>
//
// where:
//   - ISA   : b (SSE), c (AVX), d (AVX2), e (AVX-512)
//   - Mask  : M (inbranch), N (notinbranch), or both if unspecified
//   - VLEN  : explicit simdlen or computed from CDT size
//   - ParamAttrs encoding:
//       v = vector, u = uniform, l = linear
//       sN = var-stride using argument index N
//       aN = alignment N
//

module attributes {
  llvm.target_triple = "x86_64-unknown-linux-gnu",
  llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
} {

  // - All parameters default to Vector
  // - No branch clause => both masked (M) and unmasked (N) variants emitted
  // - No simdlen => VLEN derived from CDT
  //   * CDT = return type i32 => 32 bits
  //   * VLEN = vector-register-size / 32
  //
  llvm.func @ds_minimal(%x: !llvm.ptr, %y: !llvm.ptr) -> i32 {
    omp.declare_simd
    %vx = llvm.load %x : !llvm.ptr -> i32
    %vy = llvm.load %y : !llvm.ptr -> i32
    %sum = llvm.add %vx, %vy : i32
    llvm.return %sum : i32
  }

  // uniform + linear with variable stride + simdlen
  //
  // The linear step is produced by:
  //   %stepv = llvm.load %step
  //
  // This is recognized as a var-stride case:
  //   - Linear.HasVarStride = true
  //   - Linear.StrideOrArg = argument index of %step
  //
  // ParamAttrs:
  //   [0] Vector
  //   [1] Uniform
  //   [2] Linear(var-stride = arg3)
  //   [3] Vector
  //
  // No branch clause => both masked (M) and unmasked (N) variants emitted.
  //
  llvm.func @ds_uniform_linear_const_step_inbranch(
      %x: !llvm.ptr, %y: !llvm.ptr, %i: !llvm.ptr) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    omp.declare_simd simdlen(8) uniform(%y : !llvm.ptr) linear(%i = %c1 : !llvm.ptr) inbranch {linear_var_types = [i32]}
    %vx = llvm.load %x : !llvm.ptr -> i32
    %vy = llvm.load %y : !llvm.ptr -> i32
    %sum = llvm.add %vx, %vy : i32
    %vi = llvm.load %i : !llvm.ptr -> i32
    %out = llvm.add %sum, %vi : i32
    llvm.return %out : i32
  }

  // uniform + linear with variable stride + simdlen
  //
  // The linear step is produced by:
  //   %stepv = llvm.load %step
  //
  // This is recognized as a var-stride case:
  //   - Linear.HasVarStride = true
  //   - Linear.StrideOrArg = argument index of %step
  //
  // ParamAttrs:
  //   [0] Vector
  //   [1] Uniform
  //   [2] Linear(var-stride = arg3)
  //   [3] Vector
  //
  // No branch clause => both masked (M) and unmasked (N) variants emitted.
  //
  llvm.func @ds_uniform_linear_var_stride(
      %x: !llvm.ptr, %y: !llvm.ptr, %i: !llvm.ptr, %step: !llvm.ptr) -> i32 {
    %stepv = llvm.load %step : !llvm.ptr -> i32
    omp.declare_simd simdlen(8) uniform(%y : !llvm.ptr) linear(%i = %stepv : !llvm.ptr) {linear_var_types = [i32]}
    %vx = llvm.load %x : !llvm.ptr -> i32
    %vy = llvm.load %y : !llvm.ptr -> i32
    %sum = llvm.add %vx, %vy : i32
    %vi = llvm.load %i : !llvm.ptr -> i32
    %prod = llvm.mul %vi, %stepv : i32
    %out = llvm.add %sum, %prod : i32
    llvm.return %out : i32
  }

  // -------------------------------------------------------------------------
  // aligned + uniform + notinbranch (no simdlen)
  //
  // ParamAttrs:
  //   [0] Vector, Alignment = 32
  //   [1] Uniform, Alignment = 128
  //   [2] Vector
  //
  // Branch:
  //   Notinbranch => only unmasked (N) variants emitted
  //
  // VLEN:
  //   No simdlen => derived from CDT (i32)
  //
  llvm.func @ds_aligned_uniform_notinbranch(
      %p0: !llvm.ptr, %p1: !llvm.ptr, %i: !llvm.ptr) -> i32 {
    omp.declare_simd aligned(%p0 : !llvm.ptr -> 32 : i64,
                             %p1 : !llvm.ptr -> 128 : i64)
                    uniform(%p1 : !llvm.ptr)
                    notinbranch
    %v0 = llvm.load %p0 : !llvm.ptr -> i32
    %v1 = llvm.load %p1 : !llvm.ptr -> i32
    %sum = llvm.add %v0, %v1 : i32
    %vi = llvm.load %i : !llvm.ptr -> i32
    %out = llvm.add %sum, %vi : i32
    llvm.return %out : i32
  }

  // Multiple declare_simd ops in the same function body
  //
  // Each omp.declare_simd independently contributes a set of
  // vector-function attributes to the same LLVM function.
  //
  llvm.func @ds_multiple_ops_same_function(%a: !llvm.ptr, %b: !llvm.ptr, %i: !llvm.ptr) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    omp.declare_simd uniform(%b : !llvm.ptr) linear(%i = %c1 : !llvm.ptr) simdlen(4) {linear_var_types = [i32]}
    omp.declare_simd uniform(%a : !llvm.ptr) simdlen(8)

    %va = llvm.load %a : !llvm.ptr -> i32
    %vb = llvm.load %b : !llvm.ptr -> i32
    %sum = llvm.add %va, %vb : i32
    %vi = llvm.load %i : !llvm.ptr -> i32
    %out = llvm.add %sum, %vi : i32
    llvm.return %out : i32
  }
}

// no branch clause => both N and M, VLEN from CDT(i32)=32b
//
// CHECK: attributes {{.+}} = {
// CHECK-SAME: "_ZGVbM4vv_ds_minimal"
// CHECK-SAME: "_ZGVbN4vv_ds_minimal"
// CHECK-SAME: "_ZGVcN8vv_ds_minimal"
// CHECK-SAME: "_ZGVdM8vv_ds_minimal"
// CHECK-SAME: "_ZGVeM16vv_ds_minimal"
// CHECK-SAME: "_ZGVeN16vv_ds_minimal"
// CHECK-SAME: }
//
// uniform + linear with constant step + simdlen + inbranch
//
// CHECK: attributes {{.+}} = {
// CHECK-SAME: "_ZGVbM8vul_ds_uniform_linear_const_step_inbranch"
// CHECK-SAME: "_ZGVcM8vul_ds_uniform_linear_const_step_inbranch"
// CHECK-SAME: "_ZGVdM8vul_ds_uniform_linear_const_step_inbranch"
// CHECK-SAME: "_ZGVeM8vul_ds_uniform_linear_const_step_inbranch"
// CHECK-SAME: }
//
// uniform + linear with var-stride via `llvm.load %step` + simdlen
//
// CHECK: attributes {{.+}} = {
// CHECK-SAME: "_ZGVbM8vuls3v_ds_uniform_linear_var_stride"
// CHECK-SAME: "_ZGVbN8vuls3v_ds_uniform_linear_var_stride"
// CHECK-SAME: "_ZGVcM8vuls3v_ds_uniform_linear_var_stride"
// CHECK-SAME: "_ZGVcN8vuls3v_ds_uniform_linear_var_stride"
// CHECK-SAME: "_ZGVdM8vuls3v_ds_uniform_linear_var_stride"
// CHECK-SAME: "_ZGVdN8vuls3v_ds_uniform_linear_var_stride"
// CHECK-SAME: "_ZGVeM8vuls3v_ds_uniform_linear_var_stride"
// CHECK-SAME: "_ZGVeN8vuls3v_ds_uniform_linear_var_stride"
// CHECK-SAME: }
//
// aligned + uniform + notinbranch
//
// CHECK: attributes {{.+}} = {
// CHECK-SAME: "_ZGVbN4va32ua128v_ds_aligned_uniform_notinbranch"
// CHECK-SAME: "_ZGVcN8va32ua128v_ds_aligned_uniform_notinbranch"
// CHECK-SAME: "_ZGVdN8va32ua128v_ds_aligned_uniform_notinbranch"
// CHECK-SAME: "_ZGVeN16va32ua128v_ds_aligned_uniform_notinbranch"
// CHECK-SAME: }
//
// multiple declare_simd ops in the same function body
//
// CHECK: attributes {{.+}} = {
// CHECK-SAME: "_ZGVbM4vul_ds_multiple_ops_same_function"
// CHECK-SAME: "_ZGVbM8uvv_ds_multiple_ops_same_function"
// CHECK-SAME: "_ZGVbN4vul_ds_multiple_ops_same_function"
// CHECK-SAME: "_ZGVbN8uvv_ds_multiple_ops_same_function"
// CHECK-SAME: "_ZGVcM4vul_ds_multiple_ops_same_function"
// CHECK-SAME: "_ZGVcM8uvv_ds_multiple_ops_same_function"
// CHECK-SAME: "_ZGVcN4vul_ds_multiple_ops_same_function"
// CHECK-SAME: "_ZGVcN8uvv_ds_multiple_ops_same_function"
// CHECK-SAME: "_ZGVdM4vul_ds_multiple_ops_same_function"
// CHECK-SAME: "_ZGVdM8uvv_ds_multiple_ops_same_function"
// CHECK-SAME: "_ZGVdN4vul_ds_multiple_ops_same_function"
// CHECK-SAME: "_ZGVdN8uvv_ds_multiple_ops_same_function"
// CHECK-SAME: "_ZGVeM4vul_ds_multiple_ops_same_function"
// CHECK-SAME: "_ZGVeM8uvv_ds_multiple_ops_same_function"
// CHECK-SAME: "_ZGVeN4vul_ds_multiple_ops_same_function"
// CHECK-SAME: "_ZGVeN8uvv_ds_multiple_ops_same_function"
// CHECK-SAME: }
