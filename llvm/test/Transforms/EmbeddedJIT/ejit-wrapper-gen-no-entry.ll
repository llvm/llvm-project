; RUN: opt -passes=ejit-wrapper-gen -S %s | FileCheck %s

; CHECK-NOT: jit_entry:
; CHECK-NOT: ejit_compile_or_get

define void @regular_func() {
  ret void
}
