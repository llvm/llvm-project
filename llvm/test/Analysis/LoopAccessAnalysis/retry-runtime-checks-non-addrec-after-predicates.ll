; RUN: opt -passes='print<access-info>' -disable-output %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

define void @retry_do_not_cast_non_addrec(ptr %p) {
; CHECK-LABEL: 'retry_do_not_cast_non_addrec'
; CHECK-NEXT:    loop:
; CHECK-NEXT:      Memory dependences are safe
; CHECK-NEXT:      Dependences:
; CHECK-NEXT:      Run-time memory checks:
; CHECK-NEXT:      Grouped accesses:
; CHECK-NEXT:        Group GRP0:
; CHECK-NEXT:          (Low: %p High: (68719476722 + %p))
; CHECK-NEXT:            Member: {(1 + %p),+,16}<%loop>
; CHECK-NEXT:            Member: {%p,+,16}<%loop>
; CHECK-EMPTY:
; CHECK-NEXT:      Non vectorizable stores to invariant address were not found in loop.
; CHECK-NEXT:      SCEV assumptions:
; CHECK-NEXT:      {0,+,16}<%loop> Added Flags: <nusw>
; CHECK-NEXT:      {%p,+,16}<%loop> Added Flags: <nusw>
; CHECK-NEXT:      {(1 + %p),+,16}<%loop> Added Flags: <nusw>
; CHECK-EMPTY:
; CHECK-NEXT:      Expressions re-written:
; CHECK-NEXT:      [PSE]  %gep.0 = getelementptr i8, ptr %p, i64 %idx.0:
; CHECK-NEXT:        ((zext i32 %iv to i64) + %p)
; CHECK-NEXT:        --> {%p,+,16}<%loop>
; CHECK-NEXT:      [PSE]  %gep.1 = getelementptr i8, ptr %p, i64 %idx.1:
; CHECK-NEXT:        ((zext i32 (1 + %iv)<nsw> to i64) + %p)
; CHECK-NEXT:        --> {(1 + %p),+,16}<%loop>
;
entry:
  br label %loop

loop:
  %count = phi i32 [ %count.next, %loop ], [ 0, %entry ]
  %iv = phi i32 [ %iv.next, %loop ], [ 0, %entry ]
  %idx.0 = zext i32 %iv to i64
  %gep.0 = getelementptr i8, ptr %p, i64 %idx.0
  store i8 0, ptr %gep.0, align 1
  %iv.plus.1 = add i32 %iv, 1
  %idx.1 = zext i32 %iv.plus.1 to i64
  %gep.1 = getelementptr i8, ptr %p, i64 %idx.1
  store i8 0, ptr %gep.1, align 1
  %and = and i32 %iv, 511
  %iv.next = add i32 %and, 16
  %count.next = add i32 %count, 1
  %ec = icmp eq i32 %count.next, 0
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}
