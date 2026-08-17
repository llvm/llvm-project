; RUN: mlir-translate -import-llvm %s | FileCheck %s

; CHECK-LABEL: llvm.func @disable_tail_calls()
; CHECK-SAME: disable_tail_calls = true
define void @disable_tail_calls() #0 {
  ret void
}

attributes #0 = { "disable-tail-calls"="true" }
