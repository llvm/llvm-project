; RUN: opt < %s -loop-reduce -S | FileCheck %s
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

define void @f() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  br label %for.cond.i

for.cond.i:                                       ; preds = %for.inc.i, %entry
  %_First.addr.0.i = phi i32* [ null, %entry ], [ %incdec.ptr.i, %for.inc.i ]
  invoke void @g()
          to label %for.inc.i unwind label %catch.dispatch.i

catch.dispatch.i:                                 ; preds = %for.cond.i
  %cs = catchswitch within none [label %for.cond.1.preheader.i] unwind to caller

for.cond.1.preheader.i:                           ; preds = %catch.dispatch.i
  %0 = catchpad within %cs [i8* null, i32 64, i8* null]
  %cmp.i = icmp eq i32* %_First.addr.0.i, null
  br label %for.cond.1.i

for.cond.1.i:                                     ; preds = %for.body.i, %for.cond.1.preheader.i
  br i1 %cmp.i, label %for.end.i, label %for.body.i

for.body.i:                                       ; preds = %for.cond.1.i
  call void @g()
  br label %for.cond.1.i

for.inc.i:                                        ; preds = %for.cond.i
  %incdec.ptr.i = getelementptr inbounds i32, i32* %_First.addr.0.i, i64 1
  br label %for.cond.i

for.end.i:                                        ; preds = %for.cond.1.i
  catchret from %0 to label %leave

leave:                                            ; preds = %for.end.i
  ret void
}

; CHECK-LABEL: define void @f(
; CHECK: %[[PHI:.*]]  = phi i64 [ %[[IV_NEXT:.*]], {{.*}} ], [ 0, {{.*}} ]
; CHECK: %[[ITOP:.*]] = inttoptr i64 %[[PHI]] to i32*
; CHECK: %[[CMP:.*]]  = icmp eq i32* %[[ITOP]], null
; CHECK: %[[IV_NEXT]] = add i64 %[[PHI]], -4

declare void @g()

declare i32 @__CxxFrameHandler3(...)
