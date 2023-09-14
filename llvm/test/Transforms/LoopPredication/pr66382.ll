; XFAIL: *
; RUN: opt -S -loop-predication-skip-profitability-checks=false -passes='require<scalar-evolution>,loop-mssa(loop-predication)' %s | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nocallback nofree nosync willreturn
declare void @llvm.experimental.guard(i1, ...) #0

define void @foo() {
entry:
  br label %Header

Header:                                           ; preds = %Latch, %entry
  %j2 = phi i64 [ 0, %entry ], [ %j.next, %Latch ]
  call void (i1, ...) @llvm.experimental.guard(i1 false, i32 0) [ "deopt"() ]
  %j.next = add i64 %j2, 1
  br i1 false, label %Latch, label %exit

Latch:                                            ; preds = %Header
  %speculate_trip_count = icmp ult i64 %j2, 0
  br i1 %speculate_trip_count, label %Header, label %common.ret, !prof !0

common.ret:                                       ; preds = %exit, %Latch
  ret void

exit:                                             ; preds = %Header
  br label %common.ret
}

attributes #0 = { nocallback nofree nosync willreturn }

!0 = !{!"branch_weights", i32 0, i32 0}
