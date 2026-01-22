// RUN: not mlir-translate --mlir-to-llvmir %s 2>&1 | FileCheck %s

// Remove this test when codegen for aarch64 has been done
module attributes {
  llvm.target_triple = "aarch64-unknown-linux-gnu",
  llvm.data_layout = "e-m:e-i64:64-n32:64"
} {
  llvm.func @omp_declare_simd_nonx86(%x: !llvm.ptr, %y: !llvm.ptr) -> i32 {
    omp.declare_simd
    %vx = llvm.load %x : !llvm.ptr -> i32
    %vy = llvm.load %y : !llvm.ptr -> i32
    %sum = llvm.add %vx, %vy : i32
    llvm.return %sum : i32
  }
}

// CHECK: error: 'omp.declare_simd' op to LLVM IR currently only supported on x86
// CHECK-SAME: (got aarch64-unknown-linux-gnu)

