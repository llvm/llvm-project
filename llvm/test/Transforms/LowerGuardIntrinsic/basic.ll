; RUN: opt -S -lower-guard-intrinsic < %s | FileCheck %s

declare void @llvm.experimental.guard(i1, ...)

define i8 @f_basic(i1* %c_ptr) {
; CHECK-LABEL: @f_basic(

  %c = load volatile i1, i1* %c_ptr
  call void(i1, ...) @llvm.experimental.guard(i1 %c, i32 1) [ "deopt"(i32 1) ]
  ret i8 5

; CHECK:  br i1 %c, label %guarded, label %deopt
; CHECK: deopt:
; CHECK-NEXT:  %deoptcall = call i8 (...) @llvm.experimental.deoptimize.i8(i32 1) [ "deopt"(i32 1) ]
; CHECK-NEXT:  ret i8 %deoptcall
; CHECK: guarded:
; CHECK-NEXT:  ret i8 5
}

define void @f_void_return_ty(i1* %c_ptr) {
; CHECK-LABEL: @f_void_return_ty(

  %c = load volatile i1, i1* %c_ptr
  call void(i1, ...) @llvm.experimental.guard(i1 %c, i32 1) [ "deopt"() ]
  ret void

; CHECK:  br i1 %c, label %guarded, label %deopt
; CHECK: deopt:
; CHECK-NEXT:  call void (...) @llvm.experimental.deoptimize.isVoid(i32 1) [ "deopt"() ]
; CHECK-NEXT:  ret void
; CHECK: guarded:
; CHECK-NEXT:  ret void
}

define void @f_multiple_args(i1* %c_ptr) {
; CHECK-LABEL: @f_multiple_args(

  %c = load volatile i1, i1* %c_ptr
  call void(i1, ...) @llvm.experimental.guard(i1 %c, i32 1, i32 2, double 500.0) [ "deopt"(i32 2, i32 3) ]
  ret void

; CHECK: br i1 %c, label %guarded, label %deopt
; CHECK: deopt:
; CHECK-NEXT:  call void (...) @llvm.experimental.deoptimize.isVoid(i32 1, i32 2, double 5.000000e+02) [ "deopt"(i32 2, i32 3) ]
; CHECK-NEXT:  ret void
; CHECK: guarded:
; CHECK-NEXT:  ret void
}

define i32 @f_zero_args(i1* %c_ptr) {
; CHECK-LABEL: @f_zero_args(
  %c = load volatile i1, i1* %c_ptr
  call void(i1, ...) @llvm.experimental.guard(i1 %c) [ "deopt"(i32 2, i32 3) ]
  ret i32 500

; CHECK: br i1 %c, label %guarded, label %deopt
; CHECK: deopt:
; CHECK-NEXT:  %deoptcall = call i32 (...) @llvm.experimental.deoptimize.i32() [ "deopt"(i32 2, i32 3) ]
; CHECK-NEXT:  ret i32 %deoptcall
; CHECK: guarded:
; CHECK-NEXT:  ret i32 500
}
