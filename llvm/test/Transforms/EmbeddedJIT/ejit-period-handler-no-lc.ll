; RUN: opt -passes=ejit-period-handler -S %s | FileCheck %s

; Verify pass does nothing when no ejit_period_lc functions
; CHECK-NOT: ejit_deactivate_array
; CHECK-NOT: ejit_activate_array

define void @regular_func() {
  ret void
}
