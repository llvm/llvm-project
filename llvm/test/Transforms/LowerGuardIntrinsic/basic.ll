; RUN: opt -S -passes=lower-guard-intrinsic < %s | FileCheck %s
; RUN: opt -S -passes='lower-guard-intrinsic' < %s | FileCheck %s

declare void @llvm.experimental.guard(i1, ...)

define i8 @f_basic(ptr %c_ptr) {
; CHECK-LABEL: @f_basic(

  %c = load volatile i1, ptr %c_ptr
  call void(i1, ...) @llvm.experimental.guard(i1 %c, i32 1) [ "deopt"(i32 1) ]
  ret i8 5

; CHECK:  br i1 %c, label %guarded, label %deopt, !prof !0
; CHECK: deopt:
; CHECK-NEXT:  %deoptcall = call i8 (...) @llvm.experimental.deoptimize.i8(i32 1) [ "deopt"(i32 1) ]
; CHECK-NEXT:  ret i8 %deoptcall
; CHECK: guarded:
; CHECK-NEXT:  ret i8 5
}

define void @f_void_return_ty(ptr %c_ptr) {
; CHECK-LABEL: @f_void_return_ty(

  %c = load volatile i1, ptr %c_ptr
  call void(i1, ...) @llvm.experimental.guard(i1 %c, i32 1) [ "deopt"() ]
  ret void

; CHECK:  br i1 %c, label %guarded, label %deopt, !prof !0
; CHECK: deopt:
; CHECK-NEXT:  call void (...) @llvm.experimental.deoptimize.isVoid(i32 1) [ "deopt"() ]
; CHECK-NEXT:  ret void
; CHECK: guarded:
; CHECK-NEXT:  ret void
}

define void @f_multiple_args(ptr %c_ptr) {
; CHECK-LABEL: @f_multiple_args(

  %c = load volatile i1, ptr %c_ptr
  call void(i1, ...) @llvm.experimental.guard(i1 %c, i32 1, i32 2, double 500.0) [ "deopt"(i32 2, i32 3) ]
  ret void

; CHECK: br i1 %c, label %guarded, label %deopt, !prof !0
; CHECK: deopt:
; CHECK-NEXT:  call void (...) @llvm.experimental.deoptimize.isVoid(i32 1, i32 2, double 5.000000e+02) [ "deopt"(i32 2, i32 3) ]
; CHECK-NEXT:  ret void
; CHECK: guarded:
; CHECK-NEXT:  ret void
}

define i32 @f_zero_args(ptr %c_ptr) {
; CHECK-LABEL: @f_zero_args(
  %c = load volatile i1, ptr %c_ptr
  call void(i1, ...) @llvm.experimental.guard(i1 %c) [ "deopt"(i32 2, i32 3) ]
  ret i32 500

; CHECK: br i1 %c, label %guarded, label %deopt, !prof !0
; CHECK: deopt:
; CHECK-NEXT:  %deoptcall = call i32 (...) @llvm.experimental.deoptimize.i32() [ "deopt"(i32 2, i32 3) ]
; CHECK-NEXT:  ret i32 %deoptcall
; CHECK: guarded:
; CHECK-NEXT:  ret i32 500
}

define i8 @f_with_make_implicit_md(ptr %ptr) {
; CHECK-LABEL: @f_with_make_implicit_md(
; CHECK:  br i1 %notNull, label %guarded, label %deopt, !prof !0, !make.implicit !1
; CHECK: deopt:
; CHECK-NEXT:  %deoptcall = call i8 (...) @llvm.experimental.deoptimize.i8(i32 1) [ "deopt"(i32 1) ]
; CHECK-NEXT:  ret i8 %deoptcall

  %notNull = icmp ne ptr %ptr, null
  call void(i1, ...) @llvm.experimental.guard(i1 %notNull, i32 1) [ "deopt"(i32 1) ], !make.implicit !{}
  ret i8 5
}

!0 = !{!"branch_weights", i32 1048576, i32 1}
