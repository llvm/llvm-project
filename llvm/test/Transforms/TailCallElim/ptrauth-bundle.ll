; RUN: opt < %s -passes=tailcallelim -verify-dom-info -S | FileCheck %s
; Check that the "ptrauth" operand bundle doesn't prevent tail calls.

define i64 @f_1(i64 %x, ptr %f_0) {
; CHECK-LABEL: @f_1(
entry:
; CHECK: tail call i64 %f_0(i64 %x) [ "ptrauth"(i32 42, i64 %x) ]
  %tmp = call i64 %f_0(i64 %x) [ "ptrauth"(i32 42, i64 %x) ]
  ret i64 0
}
