// RUN: mlir-translate --mlir-to-llvmir %s 2>&1 | FileCheck %s
//
// Tests for Widest Data Size (WDS) on AArch64 SVE.
//
// WDS is used to check accepted values <N> of simdlen(<N>) when targeting
// fixed-length SVE vector function names. For X = WDS * <N> * 8,
// 128-bit <= X <= 2048-bit and X must be a multiple of 128-bit.

module attributes {
  llvm.target_triple = "aarch64-unknown-linux-gnu",
  llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
} {

  // WDS = sizeof(char) = 1, simdlen(8) and simdlen(272) are invalid.
  llvm.func @WDS_is_sizeof_char(%in: i8) -> i8 attributes {
    target_features = #llvm.target_features<["+sve"]>
  } {
    omp.declare_simd simdlen(8)
    omp.declare_simd simdlen(16)
    omp.declare_simd simdlen(256)
    omp.declare_simd simdlen(272)
    llvm.return %in : i8
  }

  // WDS = sizeof(short) = 2, simdlen(4) and simdlen(136) are invalid.
  llvm.func @WDS_is_sizeof_short(%in: i16) -> i8 attributes {
    target_features = #llvm.target_features<["+sve"]>
  } {
    omp.declare_simd simdlen(4)
    omp.declare_simd simdlen(8)
    omp.declare_simd simdlen(128)
    omp.declare_simd simdlen(136)
    %0 = llvm.trunc %in : i16 to i8
    llvm.return %0 : i8
  }

  // WDS = sizeof(float) = 4 because of the linear clause on float pointer.
  // simdlen(2) and simdlen(68) are invalid.
  llvm.func @WDS_is_sizeof_float_pointee(%in: f32, %sin: !llvm.ptr) attributes {
    target_features = #llvm.target_features<["+sve"]>
  } {
    %c4 = llvm.mlir.constant(4 : i64) : i64
    omp.declare_simd notinbranch simdlen(2)
                     linear(%sin : !llvm.ptr = %c4 : i64) {arg_types = [f32, f32]}
    omp.declare_simd notinbranch simdlen(4)
                     linear(%sin : !llvm.ptr = %c4 : i64) {arg_types = [f32, f32]}
    omp.declare_simd notinbranch simdlen(64)
                     linear(%sin : !llvm.ptr = %c4 : i64) {arg_types = [f32, f32]}
    omp.declare_simd notinbranch simdlen(68)
                     linear(%sin : !llvm.ptr = %c4 : i64) {arg_types = [f32, f32]}
    llvm.return
  }

  // WDS = sizeof(double) = 8 because of the linear clause on double pointer.
  // simdlen(34) is invalid.
  llvm.func @WDS_is_sizeof_double_pointee(%in: f32, %sin: !llvm.ptr) attributes {
    target_features = #llvm.target_features<["+sve"]>
  } {
    %c8 = llvm.mlir.constant(8 : i64) : i64
    omp.declare_simd notinbranch simdlen(2)
                     linear(%sin : !llvm.ptr = %c8 : i64) {arg_types = [f32, f64]}
    omp.declare_simd notinbranch simdlen(4)
                     linear(%sin : !llvm.ptr = %c8 : i64) {arg_types = [f32, f64]}
    omp.declare_simd notinbranch simdlen(32)
                     linear(%sin : !llvm.ptr = %c8 : i64) {arg_types = [f32, f64]}
    omp.declare_simd notinbranch simdlen(34)
                     linear(%sin : !llvm.ptr = %c8 : i64) {arg_types = [f32, f64]}
    llvm.return
  }

  // WDS = sizeof(double) = 8, simdlen(34) is invalid.
  llvm.func @WDS_is_sizeof_double(%in: f64) -> f64 attributes {
    target_features = #llvm.target_features<["+sve"]>
  } {
    omp.declare_simd simdlen(2)
    omp.declare_simd simdlen(4)
    omp.declare_simd simdlen(32)
    omp.declare_simd simdlen(34)
    llvm.return %in : f64
  }
}

// WDS=1: simdlen(8) -> X=64 < 128: invalid; simdlen(272) -> X=2176 > 2048: invalid
// CHECK-DAG: _ZGVsM16v_WDS_is_sizeof_char
// CHECK-DAG: _ZGVsM256v_WDS_is_sizeof_char
// CHECK-NOT: _ZGV{{.*}}_WDS_is_sizeof_char

// WDS=2: simdlen(4) -> X=64 < 128: invalid; simdlen(136) -> X=2176 > 2048: invalid
// CHECK-DAG: _ZGVsM8v_WDS_is_sizeof_short
// CHECK-DAG: _ZGVsM128v_WDS_is_sizeof_short
// CHECK-NOT: _ZGV{{.*}}_WDS_is_sizeof_short

// WDS=4: simdlen(2) -> X=64 < 128: invalid; simdlen(68) -> X=2176 > 2048: invalid
// CHECK-DAG: _ZGVsM4vl4_WDS_is_sizeof_float_pointee
// CHECK-DAG: _ZGVsM64vl4_WDS_is_sizeof_float_pointee
// CHECK-NOT: _ZGV{{.*}}_WDS_is_sizeof_float_pointee

// WDS=8: simdlen(34) -> X=2176 > 2048: invalid
// CHECK-DAG: _ZGVsM2vl8_WDS_is_sizeof_double_pointee
// CHECK-DAG: _ZGVsM4vl8_WDS_is_sizeof_double_pointee
// CHECK-DAG: _ZGVsM32vl8_WDS_is_sizeof_double_pointee
// CHECK-NOT: _ZGV{{.*}}_WDS_is_sizeof_double_pointee

// WDS=8: simdlen(34) -> X=2176 > 2048: invalid
// CHECK-DAG: _ZGVsM2v_WDS_is_sizeof_double
// CHECK-DAG: _ZGVsM4v_WDS_is_sizeof_double
// CHECK-DAG: _ZGVsM32v_WDS_is_sizeof_double
// CHECK-NOT: _ZGV{{.*}}_WDS_is_sizeof_double
