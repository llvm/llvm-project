; RUN: opt -function-attrs -S < %s | FileCheck %s
; RUN: opt -passes=function-attrs -S < %s | FileCheck %s

; This checks for a previously existing iterator wraparound bug in
; FunctionAttrs, and in the process covers corner cases with varargs.

declare void @llvm.va_start(i8*)
declare void @llvm.va_end(i8*)

define void @va_func(i32* readonly %b, ...) readonly nounwind {
; CHECK-LABEL: define void @va_func(i32* nocapture readonly %b, ...)
 entry:
  %valist = alloca i8
  call void @llvm.va_start(i8* %valist)
  call void @llvm.va_end(i8* %valist)
  %x = call i32 @caller(i32* %b)
  ret void
}

define i32 @caller(i32* %x) {
; CHECK-LABEL: define i32 @caller(i32* nocapture readonly %x)
 entry:
  call void(i32*,...) @va_func(i32* null, i32 0, i32 0, i32 0, i32* %x)
  ret i32 42
}

define void @va_func2(i32* readonly %b, ...) {
; CHECK-LABEL: define void @va_func2(i32* nocapture readonly %b, ...)
 entry:
  %valist = alloca i8
  call void @llvm.va_start(i8* %valist)
  call void @llvm.va_end(i8* %valist)
  %x = call i32 @caller(i32* %b)
  ret void
}

define i32 @caller2(i32* %x, i32* %y) {
; CHECK-LABEL: define i32 @caller2(i32* nocapture readonly %x, i32* %y)
 entry:
  call void(i32*,...) @va_func2(i32* %x, i32 0, i32 0, i32 0, i32* %y)
  ret i32 42
}

