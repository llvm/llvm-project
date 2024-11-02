; RUN: opt -function-attrs -S < %s | FileCheck %s
; RUN: opt -passes=function-attrs -S < %s | FileCheck %s

; This checks for a previously existing iterator wraparound bug in
; FunctionAttrs, and in the process covers corner cases with varargs.

declare void @llvm.va_start(ptr)
declare void @llvm.va_end(ptr)

define void @va_func(ptr readonly %b, ...) readonly nounwind {
; CHECK-LABEL: define void @va_func(ptr nocapture readonly %b, ...)
 entry:
  %valist = alloca i8
  call void @llvm.va_start(ptr %valist)
  call void @llvm.va_end(ptr %valist)
  %x = call i32 @caller(ptr %b)
  ret void
}

define i32 @caller(ptr %x) {
; CHECK-LABEL: define i32 @caller(ptr nocapture readonly %x)
 entry:
  call void(ptr,...) @va_func(ptr null, i32 0, i32 0, i32 0, ptr %x)
  ret i32 42
}

define void @va_func2(ptr readonly %b, ...) {
; CHECK-LABEL: define void @va_func2(ptr nocapture readonly %b, ...)
 entry:
  %valist = alloca i8
  call void @llvm.va_start(ptr %valist)
  call void @llvm.va_end(ptr %valist)
  %x = call i32 @caller(ptr %b)
  ret void
}

define i32 @caller2(ptr %x, ptr %y) {
; CHECK-LABEL: define i32 @caller2(ptr nocapture readonly %x, ptr %y)
 entry:
  call void(ptr,...) @va_func2(ptr %x, i32 0, i32 0, i32 0, ptr %y)
  ret i32 42
}

