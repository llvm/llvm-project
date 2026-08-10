; RUN: mlir-translate %s -import-llvm -mlir-timing 2>&1 | FileCheck %s

; CHECK: Execution time report
; CHECK: Total Execution Time:
; CHECK: Name
; CHECK-NEXT: Translate LLVMIR to MLIR

define void @foo() {
  ret void
}
