; RUN: opt -passes=ejit-aot-module -S %s | FileCheck %s

; Verify aot-module does nothing when no ejit metadata at all
; CHECK-NOT: ejit_register_period_array
; CHECK-NOT: ejit_compile_or_get
; CHECK-NOT: ejit_deactivate_array
; CHECK-NOT: ejit_auto_register

define void @regular_func() {
  ret void
}
