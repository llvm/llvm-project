; RUN: mlir-translate -import-llvm -split-input-file %s | FileCheck %s

; CHECK-LABEL: llvm.func @target_cpu()
; CHECK-SAME: target_cpu = "gfx90a"
define void @target_cpu() #0 {
  ret void
}

attributes #0 = { "target-cpu"="gfx90a" }
