; RUN: mlir-translate -import-llvm -split-input-file %s | FileCheck %s

; CHECK-LABEL: llvm.func @tune_cpu()
; CHECK-SAME: tune_cpu = "pentium4"
define void @tune_cpu() #0 {
  ret void
}

attributes #0 = { "tune-cpu"="pentium4" }
