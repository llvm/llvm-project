; RUN: opt < %s -passes='default<O2>' -disable-output
; RUN: opt < %s -passes='require<opt-remark-emit>,loop(loop-unroll-full)' \
; RUN:   -unroll-full-max-cloned-instructions=1 -unroll-peel-count=0 -S \
; RUN:   | FileCheck %s
; RUN: opt < %s -passes=loop-unroll \
; RUN:   -unroll-full-max-cloned-instructions=35 -unroll-peel-count=0 -S \
; RUN:   | FileCheck %s --check-prefix=BELOW
; RUN: opt < %s -passes=loop-unroll \
; RUN:   -unroll-full-max-cloned-instructions=36 -unroll-peel-count=0 -S \
; RUN:   | FileCheck %s --check-prefix=AT
; RUN: opt < %s -passes=loop-unroll \
; RUN:   -unroll-full-max-cloned-instructions=1 -unroll-peel-count=0 -S \
; RUN:   | FileCheck %s --check-prefix=EXPLICIT
; RUN: llvm-extract -func=boundary %s -S \
; RUN:   | opt -passes=loop-unroll -unroll-full-max-cloned-instructions=1 \
; RUN:     -unroll-threshold=4294967295 -unroll-peel-count=0 -S \
; RUN:   | FileCheck %s --check-prefix=NO-THRESHOLD
;
; CHECK-LABEL: define void @nested(
; CHECK: loop.5:
; CHECK: %iv.5 = phi i64
;
; BELOW-LABEL: define i32 @boundary(
; BELOW: loop:
; BELOW: %iv = phi i32
;
; AT-LABEL: define i32 @boundary(
; AT-NOT: phi
; AT: ret i32 10
;
; EXPLICIT-LABEL: define i32 @explicit_enable(
; EXPLICIT-NOT: phi
; EXPLICIT: ret i32 10
;
; NO-THRESHOLD-LABEL: define i32 @boundary(
; NO-THRESHOLD-NOT: phi
; NO-THRESHOLD: ret i32 10

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @boundary() {
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %next, %loop ]
  %next = add i32 %iv, 1
  %continue = icmp ult i32 %next, 10
  br i1 %continue, label %loop, label %exit

exit:
  ret i32 %next
}

define i32 @explicit_enable() {
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %next, %loop ]
  %next = add i32 %iv, 1
  %continue = icmp ult i32 %next, 10
  br i1 %continue, label %loop, label %exit, !llvm.loop !0

exit:
  ret i32 %next
}

define void @nested(i32 %arg) {
entry:
  br label %loop.1

loop.1:
  %iv.1 = phi i64 [ %next.1, %loop.1.latch ], [ 0, %entry ]
  %cmp.1 = icmp slt i64 %iv.1, 10
  br i1 %cmp.1, label %loop.2, label %exit

loop.2:
  %iv.2 = phi i64 [ %next.2, %loop.2.latch ], [ 0, %loop.1 ]
  %value.2 = phi i32 [ %next.value.2, %loop.2.latch ], [ 0, %loop.1 ]
  %cmp.2 = icmp slt i64 %iv.2, 10
  br i1 %cmp.2, label %body.2, label %loop.1.latch

body.2:
  %next.value.2 = add i32 %value.2, 1
  %enter.3 = icmp slt i32 %value.2, 0
  br i1 %enter.3, label %loop.3, label %loop.2.latch

loop.3:
  %iv.3 = phi i64 [ %next.3, %loop.3.latch ], [ 0, %body.2 ]
  %cmp.3 = icmp slt i64 %iv.3, 10
  br i1 %cmp.3, label %loop.4, label %loop.2.latch

loop.4:
  %iv.4 = phi i64 [ %next.4, %loop.4.latch ], [ 0, %loop.3 ]
  %value.4 = phi i32 [ %next.value.4, %loop.4.latch ], [ 0, %loop.3 ]
  %cmp.4 = icmp slt i64 %iv.4, 10
  br i1 %cmp.4, label %body.4, label %loop.3.latch

body.4:
  %next.value.4 = add i32 %value.4, 1
  %enter.5 = icmp slt i32 %value.4, 0
  br i1 %enter.5, label %loop.5, label %loop.4.latch

loop.5:
  %iv.5 = phi i64 [ %next.5, %body.5 ], [ 0, %body.4 ]
  %value.5 = phi i32 [ %arg, %body.5 ], [ 0, %body.4 ]
  %cmp.5 = icmp slt i64 %iv.5, 10
  br i1 %cmp.5, label %body.5, label %loop.4.latch

body.5:
  %negative = icmp slt i32 %value.5, 0
  call void @llvm.assume(i1 %negative)
  call void @llvm.stackrestore.p0(ptr null)
  %next.5 = add i64 %iv.5, 1
  br label %loop.5

loop.4.latch:
  %next.4 = add i64 %iv.4, 1
  br label %loop.4

loop.3.latch:
  %next.3 = add i64 %iv.3, 1
  br label %loop.3

loop.2.latch:
  %next.2 = add i64 %iv.2, 1
  br label %loop.2

loop.1.latch:
  %next.1 = add i64 %iv.1, 1
  br label %loop.1

exit:
  ret void
}

declare void @llvm.stackrestore.p0(ptr) #0
declare void @llvm.assume(i1 noundef) #1

attributes #0 = { nocallback nofree nosync nounwind willreturn }
attributes #1 = {
  nocallback nofree nosync nounwind willreturn
  memory(inaccessiblemem: write)
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.enable"}
