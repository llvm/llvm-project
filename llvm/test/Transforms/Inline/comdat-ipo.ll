; RUN: opt -inline -S < %s | FileCheck %s

define i32 @caller() {
; CHECK-LABEL: @caller(
; CHECK-NEXT:  %val2 = call i32 @linkonce_callee(i32 42)
; CHECK-NEXT:  ret i32 %val2

  %val = call i32 @odr_callee()
  %val2 = call i32 @linkonce_callee(i32 %val);
  ret i32 %val2
}

define linkonce_odr i32 @odr_callee() {
  ret i32 42
}

define linkonce i32 @linkonce_callee(i32 %val) {
  ret i32 %val
}
