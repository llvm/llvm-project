// RUN: mlir-translate --mlir-to-llvmir %s | FileCheck %s

module attributes {
  llvm.target_triple = "aarch64-unknown-linux-gnu",
  llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
} {
  llvm.func @"_Z1fd"(%x: f64) -> f64 attributes {
    target_features = #llvm.target_features<["+neon"]>
  } {
    omp.declare_simd
    llvm.return %x : f64
  }

  llvm.func @"_Z1ff"(%x: f32) -> f32 attributes {
    target_features = #llvm.target_features<["+neon"]>
  } {
    omp.declare_simd
    llvm.return %x : f32
  }

  llvm.func @"_Z1gd"(%x: f64) -> f64 attributes {
    target_features = #llvm.target_features<["+sve"]>
  } {
    omp.declare_simd
    llvm.return %x : f64
  }

  llvm.func @"_Z1gf"(%x: f32) -> f32 attributes {
    target_features = #llvm.target_features<["+sve"]>
  } {
    omp.declare_simd
    llvm.return %x : f32
  }
}

// CHECK-DAG: { "_ZGVnM2v__Z1fd" "_ZGVnN2v__Z1fd" "target-features"="+neon" }
// CHECK-DAG: { "_ZGVnM2v__Z1ff" "_ZGVnM4v__Z1ff" "_ZGVnN2v__Z1ff" "_ZGVnN4v__Z1ff" "target-features"="+neon" }
// CHECK-DAG: { "_ZGVsMxv__Z1gd" "target-features"="+sve" }
// CHECK-DAG: { "_ZGVsMxv__Z1gf" "target-features"="+sve" }
