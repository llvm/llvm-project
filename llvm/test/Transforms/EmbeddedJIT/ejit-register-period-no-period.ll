; RUN: opt -passes=ejit-register-period -S %s | FileCheck %s

; Verify PASS2 does nothing when no period variables
; CHECK-NOT: ejit_register_period_array
; CHECK-NOT: ejit_register_static_var
; CHECK-NOT: ejit_auto_register

@regular_data = global [4 x i32] zeroinitializer

define void @dummy() {
  ret void
}
