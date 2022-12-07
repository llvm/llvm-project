; RUN: opt -S -passes=loop-unroll %s | FileCheck %s
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc18.0.0"

define void @test1() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.inc
  %phi = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  invoke void @callee(i32 %phi)
          to label %for.inc unwind label %ehcleanup

for.inc:                                          ; preds = %for.body
  %inc = add nuw nsw i32 %phi, 1
  %cmp = icmp slt i32 %inc, 3
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.inc
  call void @dtor()
  ret void

ehcleanup:                                        ; preds = %for.body
  %cp = cleanuppad within none []
  call void @dtor() [ "funclet"(token %cp) ]
  cleanupret from %cp unwind to caller
}

; CHECK-LABEL: define void @test1(
; CHECK: invoke void @callee(i32 0

; CHECK: invoke void @callee(i32 1

; CHECK: invoke void @callee(i32 2

declare void @callee(i32)

declare i32 @__CxxFrameHandler3(...)

declare void @dtor()
