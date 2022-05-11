; RUN: opt -passes=loop-vectorize -force-vector-width=2 -force-vector-interleave=1 -S %s | FileCheck %s

; REQUIRES: asserts
; XFAIL: *

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-linux-gnu"

%pair = type { ptr, ptr }

define void @test_pr55375_interleave_opaque_ptr(ptr %start, ptr %end) {
entry:
  br label %loop

loop:
  %iv = phi ptr [ %start, %entry ], [ %iv.next, %loop ]
  %iv.1 = getelementptr inbounds %pair, ptr %iv, i64 0, i32 1
  store ptr %iv, ptr %iv.1, align 8
  store ptr null, ptr %iv, align 8
  %iv.next = getelementptr inbounds %pair, ptr %iv, i64 1
  %ec = icmp eq ptr %iv.next, %end
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}
