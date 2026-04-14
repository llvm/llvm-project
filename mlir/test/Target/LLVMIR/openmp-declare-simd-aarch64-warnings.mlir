// RUN: mlir-translate --mlir-to-llvmir %s 2>&1 | FileCheck %s

module attributes {
  llvm.target_triple = "aarch64-unknown-linux-gnu",
  llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
} {

  // CHECK: warning: AArch64 Advanced SIMD declare simd requires simdlen to be a power of 2
  llvm.func @advsimd_non_power2(%x: f32) -> f64 attributes {
    target_features = #llvm.target_features<["+neon"]>
  } {
    omp.declare_simd simdlen(6)
    %0 = llvm.fpext %x : f32 to f64
    llvm.return %0 : f64
  }

  // CHECK: warning: simdlen(1) has no effect on AArch64 declare simd
  llvm.func @advsimd_simdlen1(%x: f64) -> f32 attributes {
    target_features = #llvm.target_features<["+neon"]>
  } {
    omp.declare_simd simdlen(1)
    %0 = llvm.fptrunc %x : f64 to f32
    llvm.return %0 : f32
  }

  // CHECK: warning: simdlen(1) has no effect on AArch64 declare simd
  llvm.func @sve_simdlen1(%x: f32) -> f64 attributes {
    target_features = #llvm.target_features<["+sve"]>
  } {
    omp.declare_simd simdlen(1)
    %0 = llvm.fpext %x : f32 to f64
    llvm.return %0 : f64
  }

  // WDS=64 (f64 ret), 5*64=320 which is not a multiple of 128.
  // CHECK: warning: AArch64 SVE fixed-length declare simd simdlen must fit architectural lane limits for element width 64
  llvm.func @sve_not_multiple_128(%x: f32) -> f64 attributes {
    target_features = #llvm.target_features<["+sve"]>
  } {
    omp.declare_simd simdlen(5)
    %0 = llvm.fpext %x : f32 to f64
    llvm.return %0 : f64
  }

  // WDS=64 (f64 ret), 34*64=2176 > 2048.
  // CHECK: warning: AArch64 SVE fixed-length declare simd simdlen must fit architectural lane limits for element width 64
  llvm.func @sve_exceeds_2048(%x: f32) -> f64 attributes {
    target_features = #llvm.target_features<["+sve"]>
  } {
    omp.declare_simd simdlen(34)
    %0 = llvm.fpext %x : f32 to f64
    llvm.return %0 : f64
  }

  // valid AdvSIMD simdlen (power of 2, > 1).
  // CHECK-NOT: warning:{{.*}}advsimd_valid
  llvm.func @advsimd_valid(%x: f32) -> f64 attributes {
    target_features = #llvm.target_features<["+neon"]>
  } {
    omp.declare_simd simdlen(4)
    %0 = llvm.fpext %x : f32 to f64
    llvm.return %0 : f64
  }

  // valid SVE simdlen (2*64=128, multiple of 128 and <= 2048).
  // CHECK-NOT: warning:{{.*}}sve_valid
  llvm.func @sve_valid(%x: f32) -> f64 attributes {
    target_features = #llvm.target_features<["+sve"]>
  } {
    omp.declare_simd simdlen(2)
    %0 = llvm.fpext %x : f32 to f64
    llvm.return %0 : f64
  }
}
