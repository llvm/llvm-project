; RUN: opt -passes=ejit-register-bitcode -S %s | FileCheck %s

; Verify pass does nothing when there are no ejit_entry functions
; CHECK-NOT: @__ejit_bitcode
; CHECK-NOT: ejit_auto_register

define void @regular_func() {
  ret void
}

define void @another_func() {
  call void @regular_func()
  ret void
}
