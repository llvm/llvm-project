// RUN: mlir-translate --mlir-to-llvmir -split-input-file %s | FileCheck %s

// CHECK-LABEL: define void @arm_sme_zero
// CHECK: call void @llvm.aarch64.sme.zero(i32 255)
llvm.func @arm_sme_zero() {
  %mask = llvm.mlir.constant(255 : i32) : i32
  "arm_sme.intr.zero"(%mask) : (i32) -> ()
  llvm.return
}

// -----
