; RUN: opt -functionattrs -S < %s | FileCheck %s
; RUN: opt -passes=function-attrs -S < %s | FileCheck %s

; This checks for an iterator wraparound bug in FunctionAttrs.  The previous
; "incorrect" behavior was inferring readonly for the %x argument in @caller.
; Inferring readonly for %x *is* actually correct, since @va_func is marked
; readonly, but FunctionAttrs was inferring readonly for the wrong reasons (and
; we _need_ the readonly on @va_func to trigger the problematic code path).  It
; is possible that in the future FunctionAttrs becomes smart enough to infer
; readonly for %x for the right reasons, and at that point this test will have
; to be marked invalid.

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
; CHECK-LABEL: define i32 @caller(i32* nocapture %x)
 entry:
  call void(i32*,...) @va_func(i32* null, i32 0, i32 0, i32 0, i32* %x)
  ret i32 42
}
