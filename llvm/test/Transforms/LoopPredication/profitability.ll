; RUN: opt -S -loop-predication -loop-predication-skip-profitability-checks=false < %s 2>&1 | FileCheck %s
; RUN: opt -S -loop-predication-skip-profitability-checks=false -passes='require<scalar-evolution>,require<branch-prob>,loop(loop-predication)' < %s 2>&1 | FileCheck %s

; latch block exits to a speculation block. BPI already knows (without prof
; data) that deopt is very rarely
; taken. So we do not predicate this loop using that coarse latch check.
; LatchExitProbability: 0x04000000 / 0x80000000 = 3.12%
; ExitingBlockProbability: 0x7ffa572a / 0x80000000 = 99.98%
define i64 @donot_predicate(i64* nocapture readonly %arg, i32 %length, i64* nocapture readonly %arg2, i64* nocapture readonly %n_addr, i64 %i) {
; CHECK-LABEL: donot_predicate(
entry:
  %length.ext = zext i32 %length to i64
  %n.pre = load i64, i64* %n_addr, align 4
  br label %Header

; CHECK-LABEL: Header:
; CHECK:         %within.bounds = icmp ult i64 %j2, %length.ext
; CHECK-NEXT:    call void (i1, ...) @llvm.experimental.guard(i1 %within.bounds, i32 9)
Header:                                          ; preds = %entry, %Latch
  %result.in3 = phi i64* [ %arg2, %entry ], [ %arg, %Latch ]
  %j2 = phi i64 [ 0, %entry ], [ %j.next, %Latch ]
  %within.bounds = icmp ult i64 %j2, %length.ext
  call void (i1, ...) @llvm.experimental.guard(i1 %within.bounds, i32 9) [ "deopt"() ]
  %innercmp = icmp eq i64 %j2, %n.pre
  %j.next = add nuw nsw i64 %j2, 1
  br i1 %innercmp, label %Latch, label %exit, !prof !0

Latch:                                           ; preds = %Header
  %speculate_trip_count = icmp ult i64 %j.next, 1048576
  br i1 %speculate_trip_count, label %Header, label %deopt

deopt:                                            ; preds = %Latch
  %counted_speculation_failed = call i64 (...) @llvm.experimental.deoptimize.i64(i64 30) [ "deopt"(i32 0) ]
  ret i64 %counted_speculation_failed

exit:                                             ; preds = %Header
  %result.in3.lcssa = phi i64* [ %result.in3, %Header ]
  %result.le = load i64, i64* %result.in3.lcssa, align 8
  ret i64 %result.le
}
!0 = !{!"branch_weights", i32 18, i32 104200}

; predicate loop since there's no profile information and BPI concluded all
; exiting blocks have same probability of exiting from loop.
define i64 @predicate(i64* nocapture readonly %arg, i32 %length, i64* nocapture readonly %arg2, i64* nocapture readonly %n_addr, i64 %i) {
; CHECK-LABEL: predicate(
; CHECK-LABEL: entry:
; CHECK:           [[limit_check:[^ ]+]] = icmp ule i64 1048576, %length.ext
; CHECK-NEXT:      [[first_iteration_check:[^ ]+]] = icmp ult i64 0, %length.ext
; CHECK-NEXT: [[wide_cond:[^ ]+]] = and i1 [[first_iteration_check]], [[limit_check]]
entry:
  %length.ext = zext i32 %length to i64
  %n.pre = load i64, i64* %n_addr, align 4
  br label %Header

; CHECK-LABEL: Header:
; CHECK: call void (i1, ...) @llvm.experimental.guard(i1 [[wide_cond]], i32 9) [ "deopt"() ]
Header:                                          ; preds = %entry, %Latch
  %result.in3 = phi i64* [ %arg2, %entry ], [ %arg, %Latch ]
  %j2 = phi i64 [ 0, %entry ], [ %j.next, %Latch ]
  %within.bounds = icmp ult i64 %j2, %length.ext
  call void (i1, ...) @llvm.experimental.guard(i1 %within.bounds, i32 9) [ "deopt"() ]
  %innercmp = icmp eq i64 %j2, %n.pre
  %j.next = add nuw nsw i64 %j2, 1
  br i1 %innercmp, label %Latch, label %exit

Latch:                                           ; preds = %Header
  %speculate_trip_count = icmp ult i64 %j.next, 1048576
  br i1 %speculate_trip_count, label %Header, label %exitLatch

exitLatch:                                            ; preds = %Latch
  ret i64 1

exit:                                             ; preds = %Header
  %result.in3.lcssa = phi i64* [ %result.in3, %Header ]
  %result.le = load i64, i64* %result.in3.lcssa, align 8
  ret i64 %result.le
}

; Same as test above but with profiling data that the most probable exit from
; the loop is the header exiting block (not the latch block). So do not predicate.
; LatchExitProbability: 0x000020e1 / 0x80000000 = 0.00%
; ExitingBlockProbability: 0x7ffcbb86 / 0x80000000 = 99.99%
define i64 @donot_predicate_prof(i64* nocapture readonly %arg, i32 %length, i64* nocapture readonly %arg2, i64* nocapture readonly %n_addr, i64 %i) {
; CHECK-LABEL: donot_predicate_prof(
; CHECK-LABEL: entry:
entry:
  %length.ext = zext i32 %length to i64
  %n.pre = load i64, i64* %n_addr, align 4
  br label %Header

; CHECK-LABEL: Header:
; CHECK:         %within.bounds = icmp ult i64 %j2, %length.ext
; CHECK-NEXT:    call void (i1, ...) @llvm.experimental.guard(i1 %within.bounds, i32 9)
Header:                                          ; preds = %entry, %Latch
  %result.in3 = phi i64* [ %arg2, %entry ], [ %arg, %Latch ]
  %j2 = phi i64 [ 0, %entry ], [ %j.next, %Latch ]
  %within.bounds = icmp ult i64 %j2, %length.ext
  call void (i1, ...) @llvm.experimental.guard(i1 %within.bounds, i32 9) [ "deopt"() ]
  %innercmp = icmp eq i64 %j2, %n.pre
  %j.next = add nuw nsw i64 %j2, 1
  br i1 %innercmp, label %Latch, label %exit, !prof !1

Latch:                                           ; preds = %Header
  %speculate_trip_count = icmp ult i64 %j.next, 1048576
  br i1 %speculate_trip_count, label %Header, label %exitLatch, !prof !2

exitLatch:                                            ; preds = %Latch
  ret i64 1

exit:                                             ; preds = %Header
  %result.in3.lcssa = phi i64* [ %result.in3, %Header ]
  %result.le = load i64, i64* %result.in3.lcssa, align 8
  ret i64 %result.le
}
declare i64 @llvm.experimental.deoptimize.i64(...)
declare void @llvm.experimental.guard(i1, ...)

!1 = !{!"branch_weights", i32 104, i32 1042861}
!2 = !{!"branch_weights", i32 255129, i32 1}
