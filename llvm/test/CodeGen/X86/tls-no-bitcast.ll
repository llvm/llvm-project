; RUN: llc -tls-load-hoist=true -stop-after=tlshoist < %s | FileCheck %s 

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@I58561 = external thread_local global ptr

define i32 @I59676() {
entry:
; CHECK: @I59676
; CHECK-NOT: bitcast
; CHECK: tail call ptr @llvm.threadlocal.address.p0(ptr @I58561)
; CHECK-NEXT; tail call ptr @llvm.threadlocal.address.p0(ptr @I58561)
  %0 = tail call ptr @llvm.threadlocal.address.p0(ptr @I58561)
  %1 = tail call ptr @llvm.threadlocal.address.p0(ptr @I58561)
  ret i32 0
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare nonnull ptr @llvm.threadlocal.address.p0(ptr nonnull) #0

; uselistorder directives
uselistorder ptr @llvm.threadlocal.address.p0, { 1, 0 }

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
