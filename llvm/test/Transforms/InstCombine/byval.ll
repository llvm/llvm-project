; RUN: opt -S -passes=instcombine %s | FileCheck %s

declare void @add_byval_callee(double*)

; CHECK-LABEL: define void @add_byval
; CHECK: [[ARG:%.*]] = bitcast i64* %in to double*
; CHECK: call void @add_byval_callee(double* byval(double) [[ARG]])
define void @add_byval(i64* %in) {
  %tmp = bitcast void (double*)* @add_byval_callee to void (i64*)*
  call void %tmp(i64* byval(i64) %in)
  ret void
}

%t2 = type { i8 }

; CHECK-LABEL: define void @vararg_byval
; CHECK: call void (i8, ...) @vararg_callee(i8 undef, i8* byval(i8) %p)
define void @vararg_byval(i8* %p) {
  %tmp = bitcast i8* %p to %t2*
  call void (i8, ...) @vararg_callee(i8 undef, %t2* byval(%t2) %tmp)
  ret void
}

declare void @vararg_callee(i8, ...)
