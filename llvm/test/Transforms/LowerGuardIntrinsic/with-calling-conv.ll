; RUN: opt -S -lower-guard-intrinsic < %s | FileCheck %s

declare cc99 void @llvm.experimental.guard(i1, ...)

define i8 @f_basic(ptr %c_ptr) {
; CHECK-LABEL: @f_basic(
; CHECK:  br i1 %c, label %guarded, label %deopt
; CHECK: deopt:
; CHECK-NEXT:  %deoptcall = call cc99 i8 (...) @llvm.experimental.deoptimize.i8() [ "deopt"() ]
; CHECK-NEXT:  ret i8 %deoptcall

  %c = load volatile i1, ptr %c_ptr
  call cc99 void(i1, ...) @llvm.experimental.guard(i1 %c) [ "deopt"() ]
  ret i8 6
}
