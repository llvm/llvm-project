// RUN: mlir-translate --mlir-to-llvmir %s 2>&1 | FileCheck %s
//
// Tests for Narrowest Data Size (NDS) on AArch64 AdvSIMD.
//
// NDS determines <vlen> for AdvSIMD when no simdlen is specified:
//   NDS=1 -> VLEN=16,8; NDS=2 -> VLEN=8,4;
//   NDS=4 -> VLEN=4,2;  NDS>=8 -> VLEN=2.

module attributes {
  llvm.target_triple = "aarch64-unknown-linux-gnu",
  llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
} {

  // NDS = sizeof(char) = 1, ret=i8, arg=i16 -> min(1,2) = 1
  llvm.func @NDS_is_sizeof_char(%in: i16) -> i8 attributes {
    target_features = #llvm.target_features<["+neon"]>
  } {
    omp.declare_simd notinbranch
    %0 = llvm.trunc %in : i16 to i8
    llvm.return %0 : i8
  }

  // NDS = sizeof(short) = 2, ret=i32, arg=i16 -> min(4,2) = 2
  llvm.func @NDS_is_sizeof_short(%in: i16) -> i32 attributes {
    target_features = #llvm.target_features<["+neon"]>
  } {
    omp.declare_simd notinbranch
    %0 = llvm.sext %in : i16 to i32
    llvm.return %0 : i32
  }

  // NDS = sizeof(float) = 4, linear ptr to float -> pointee size = 4
  // Without linear, ptr NDS would be 8 (pointer size). With linear,
  // NDS uses pointee size via arg_types.
  llvm.func @NDS_is_sizeof_float_with_linear(%in: f64, %sin: !llvm.ptr) attributes {
    target_features = #llvm.target_features<["+neon"]>
  } {
    %c4 = llvm.mlir.constant(4 : i64) : i64
    omp.declare_simd notinbranch
                     linear(%sin : !llvm.ptr = %c4 : i64) {arg_types = [f64, f32]}
    llvm.return
  }

  // NDS = sizeof(float) = 4, ret=f64, arg=f32 -> min(8,4) = 4
  llvm.func @NDS_is_size_of_float(%in: f32) -> f64 attributes {
    target_features = #llvm.target_features<["+neon"]>
  } {
    omp.declare_simd notinbranch
    %0 = llvm.fpext %in : f32 to f64
    llvm.return %0 : f64
  }

  // NDS = sizeof(double) = 8, linear ptr to double -> pointee size = 8
  llvm.func @NDS_is_sizeof_double(%in: f64, %sin: !llvm.ptr) attributes {
    target_features = #llvm.target_features<["+neon"]>
  } {
    %c8 = llvm.mlir.constant(8 : i64) : i64
    omp.declare_simd notinbranch
                     linear(%sin : !llvm.ptr = %c8 : i64) {arg_types = [f64, f64]}
    llvm.return
  }
}

// NDS=1 -> VLEN=16,8
// CHECK-DAG: _ZGVnN16v_NDS_is_sizeof_char
// CHECK-DAG: _ZGVnN8v_NDS_is_sizeof_char
// CHECK-NOT: _ZGV{{.*}}_NDS_is_sizeof_char

// NDS=2 -> VLEN=8,4
// CHECK-DAG: _ZGVnN8v_NDS_is_sizeof_short
// CHECK-DAG: _ZGVnN4v_NDS_is_sizeof_short
// CHECK-NOT: _ZGV{{.*}}_NDS_is_sizeof_short

// NDS=4 (linear float pointee) -> VLEN=4,2
// CHECK-DAG: _ZGVnN4vl4_NDS_is_sizeof_float_with_linear
// CHECK-DAG: _ZGVnN2vl4_NDS_is_sizeof_float_with_linear
// CHECK-NOT: _ZGV{{.*}}_NDS_is_sizeof_float_with_linear

// NDS=4 -> VLEN=4,2
// CHECK-DAG: _ZGVnN4v_NDS_is_size_of_float
// CHECK-DAG: _ZGVnN2v_NDS_is_size_of_float
// CHECK-NOT: _ZGV{{.*}}_NDS_is_size_of_float

// NDS=8 -> VLEN=2
// CHECK-DAG: _ZGVnN2vl8_NDS_is_sizeof_double
// CHECK-NOT: _ZGV{{.*}}_NDS_is_sizeof_double
