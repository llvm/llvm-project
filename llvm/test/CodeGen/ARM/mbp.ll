; RUN: llc < %s | FileCheck %s
target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv7-unknown-linux-gnueabihf"

%Target = type { ptr, ptr }
%List = type { i32, ptr }

; The entry block should be the first block of the function.
; CHECK-LABEL: foo
; CHECK:       %entry
; CHECK:       %for.body
; CHECK:       %for.inc
; CHECK:       %if.then
; CHECK:       %for.cond.i
; CHECK:       %for.body.i
; CHECK:       %return

define i1 @foo(ptr %ha, i32 %he) !prof !39 {
entry:
  %TargetPtr = load ptr, ptr %ha, align 4
  %cmp1 = icmp eq ptr %TargetPtr, null
  br i1 %cmp1, label %return, label %for.body, !prof !50

for.body:
  %TargetPhi = phi ptr [ %NextPtr, %for.inc ], [ %TargetPtr, %entry ]
  %ListAddr = getelementptr inbounds %Target, ptr %TargetPhi, i32 0, i32 1
  %ListPtr = load ptr, ptr %ListAddr, align 4
  %cmp2 = icmp eq ptr %ListPtr, null
  br i1 %cmp2, label %for.inc, label %if.then, !prof !59

if.then:
  %len = load i32, ptr %ListPtr, align 4
  %ptr = getelementptr inbounds %List, ptr %ListPtr, i32 0, i32 1
  %ptr2 = load ptr, ptr %ptr, align 4
  br label %for.cond.i

for.cond.i:
  %i = phi i32 [ %len, %if.then ], [ %index, %for.body.i ]
  %index = add nsw i32 %i, -1
  %cmp3 = icmp sgt i32 %i, 0
  br i1 %cmp3, label %for.body.i, label %for.inc, !prof !75

for.body.i:
  %ptr3 = getelementptr inbounds i32, ptr %ptr2, i32 %index
  %data = load i32, ptr %ptr3, align 4
  %cmp4 = icmp eq i32 %data, %he
  br i1 %cmp4, label %return, label %for.cond.i, !prof !79

for.inc:
  %NextPtr = load ptr, ptr %TargetPhi, align 4
  %cmp5 = icmp eq ptr %NextPtr, null
  br i1 %cmp5, label %return, label %for.body, !prof !50

return:
  %retval = phi i1 [ false, %entry ], [ true, %for.body.i ], [ false, %for.inc ]
  ret i1 %retval
}

!39 = !{!"function_entry_count", i64 226}
!50 = !{!"branch_weights", i32 451, i32 1}
!59 = !{!"branch_weights", i32 1502, i32 1}
!75 = !{!"branch_weights", i32 301, i32 1}
!79 = !{!"branch_weights", i32 1, i32 301}
