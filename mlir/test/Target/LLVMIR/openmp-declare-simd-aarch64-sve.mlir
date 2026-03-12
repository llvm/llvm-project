// RUN: mlir-translate --mlir-to-llvmir %s 2>&1 | FileCheck %s

module attributes {
  llvm.target_triple = "aarch64-unknown-linux-gnu",
  llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
} {
  llvm.func @foo(%x: f32) -> f64 attributes {
    target_features = #llvm.target_features<["+sve"]>
  } {
    omp.declare_simd
    omp.declare_simd notinbranch
    omp.declare_simd simdlen(2)
    omp.declare_simd simdlen(4)
    omp.declare_simd simdlen(5) // not a multiple of 128-bits
    omp.declare_simd simdlen(6)
    omp.declare_simd simdlen(8)
    omp.declare_simd simdlen(32)
    omp.declare_simd simdlen(34) // requires more than 2048 bits
    %0 = llvm.fpext %x : f32 to f64
    llvm.return %0 : f64
  }

  llvm.func @a01_fun(%x: i32) -> i8 attributes {
    target_features = #llvm.target_features<["+sve"]>
  } {
    omp.declare_simd notinbranch
    %0 = llvm.mlir.constant(0 : i8) : i8
    llvm.return %0 : i8
  }
}

// CHECK-DAG: attributes {{#[0-9]+}} = { "_ZGVsM2v_foo" "_ZGVsM32v_foo" "_ZGVsM4v_foo" "_ZGVsM6v_foo" "_ZGVsM8v_foo" "_ZGVsMxv_foo" "target-features"="+sve" }
// CHECK-NOT: _ZGVsN
// CHECK-NOT: _ZGVsM5v_foo
// CHECK-NOT: _ZGVsM34v_foo

// CHECK-DAG: attributes {{#[0-9]+}} = { "_ZGVsMxv_a01_fun" "target-features"="+sve" }
