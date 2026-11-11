; RUN: mlir-translate -import-llvm -split-input-file %s 2>&1 | FileCheck %s

; CHECK-LABEL: llvm.func @disable_tail_calls()
; CHECK-SAME: disable_tail_calls = true
define void @disable_tail_calls() #0 {
  ret void
}

; CHECK-LABEL: llvm.func @disable_tail_calls_false()
; CHECK-NOT: disable_tail_calls
define void @disable_tail_calls_false() #1 {
  ret void
}

attributes #0 = { "disable-tail-calls"="true" }
attributes #1 = { "disable-tail-calls"="false" }
