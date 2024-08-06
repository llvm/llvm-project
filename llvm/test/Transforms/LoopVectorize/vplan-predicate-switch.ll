; REQUIRES: asserts
; RUN: opt -p loop-vectorize -force-vector-width=2 -force-vector-interleave=1 -debug -disable-output %s 2>&1 | FileCheck %s

define void @switch4_default_common_dest_with_case(ptr %start, ptr %end) {
; CHECK-NOT: VPlan
;
entry:
  br label %loop.header

loop.header:
  %ptr.iv = phi ptr [ %start, %entry ], [ %ptr.iv.next, %loop.latch ]
  %l = load i8, ptr %ptr.iv, align 1
  switch i8 %l, label %default [
  i8 -12, label %if.then.1
  i8 13, label %if.then.2
  i8 0, label %default
  ]

if.then.1:
  store i8 42, ptr %ptr.iv, align 1
  br label %loop.latch

if.then.2:
  store i8 0, ptr %ptr.iv, align 1
  br label %loop.latch

default:
  store i8 2, ptr %ptr.iv, align 1
  br label %loop.latch

loop.latch:
  %ptr.iv.next = getelementptr inbounds i8, ptr %ptr.iv, i64 1
  %ec = icmp eq ptr %ptr.iv.next, %end
  br i1 %ec, label %exit, label %loop.header

exit:
  ret void
}
