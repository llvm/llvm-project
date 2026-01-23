; RUN: mlir-translate -import-llvm -split-input-file %s | FileCheck %s

; CHECK-LABEL: llvm.func @reciprocal_estimates()
; CHECK-SAME: reciprocal_estimates = "all"
define void @reciprocal_estimates() #0 {
  ret void
}

attributes #0 = { "reciprocal-estimates" = "all" }
