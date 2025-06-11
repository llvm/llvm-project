; RUN: mlir-translate -import-llvm -split-input-file %s | FileCheck %s

; CHECK-LABEL: llvm.func @prefer_vector_width()
; CHECK-SAME: prefer_vector_width = "128"
define void @prefer_vector_width() #0 {
  ret void
}

attributes #0 = { "prefer-vector-width"="128" }
