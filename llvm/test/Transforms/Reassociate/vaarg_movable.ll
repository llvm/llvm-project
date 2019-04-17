; RUN: opt -S -reassociate -die < %s | FileCheck %s

; The two va_arg instructions depend on the memory/context, are therfore not
; identical and the sub should not be optimized to 0 by reassociate.
;
; CHECK-LABEL: @func(
; ...
; CHECK: %v0 = va_arg i8** %varargs, i32
; CHECK: %v1 = va_arg i8** %varargs, i32
; CHECK: %v0.neg = sub i32 0, %v0
; CHECK: %sub = add i32 %v0.neg, 1
; CHECK: %add = add i32 %sub, %v1
; ...
; CHECK: ret i32 %add
define i32 @func(i32 %dummy, ...) {
  %varargs = alloca i8*, align 8
  %varargs1 = bitcast i8** %varargs to i8*
  call void @llvm.va_start(i8* %varargs1)
  %v0 = va_arg i8** %varargs, i32
  %v1 = va_arg i8** %varargs, i32
  %sub = sub nsw i32 %v1, %v0
  %add = add nsw i32 %sub, 1
  call void @llvm.va_end(i8* %varargs1)
  ret i32 %add
}

declare void @llvm.va_start(i8*)
declare void @llvm.va_end(i8*)
