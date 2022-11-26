; RUN: opt -verify-loop-info -irce-print-changed-loops -irce -S < %s 2>&1 | FileCheck %s
; RUN: opt -verify-loop-info -irce-print-changed-loops -passes='require<branch-prob>,irce' -S < %s 2>&1 | FileCheck %s

; CHECK: irce: in function test_01: constrained Loop at depth 1 containing: %loop<header><exiting>,%in.bounds<latch><exiting>
; CHECK: irce: in function test_01u: constrained Loop at depth 1 containing: %loop<header><exiting>,%in.bounds<latch><exiting>
; CHECK-NOT: irce: in function test_02: constrained Loop at depth 1 containing: %loop<header><exiting>,%in.bounds<latch><exiting>
; CHECK: irce: in function test_03: constrained Loop at depth 1 containing: %loop<header><exiting>,%in.bounds<latch><exiting>
; CHECK: irce: in function test_04: constrained Loop at depth 1 containing: %loop<header><exiting>,%in.bounds<latch><exiting>
; CHECK: irce: in function test_05: constrained Loop at depth 1 containing: %loop<header><exiting>,%in.bounds<latch><exiting>
; CHECK-NOT: irce: in function test_06: constrained Loop at depth 1 containing: %loop<header><exiting>,%in.bounds<latch><exiting>
; CHECK: irce: in function test_07: constrained Loop at depth 1 containing: %loop<header><exiting>,%in.bounds<latch><exiting>
; CHECK-NOT: irce: in function test_08: constrained Loop at depth 1 containing: %loop<header><exiting>,%in.bounds<latch><exiting>

; Show that IRCE can turn 'ne' condition to 'slt' in increasing IV when the IV
; can be negative at some point.
define void @test_01(ptr %arr, ptr %a_len_ptr) #0 {

; CHECK:      test_01
; CHECK:        main.exit.selector:
; CHECK-NEXT:     [[PSEUDO_PHI:%[^ ]+]] = phi i32 [ %idx.next, %in.bounds ]
; CHECK-NEXT:     [[COND:%[^ ]+]] = icmp slt i32 [[PSEUDO_PHI]], 100
; CHECK-NEXT:     br i1 [[COND]]

entry:
  %len = load i32, ptr %a_len_ptr, !range !0
  br label %loop

loop:
  %idx = phi i32 [ -3, %entry ], [ %idx.next, %in.bounds ]
  %idx.next = add i32 %idx, 1
  %abc = icmp slt i32 %idx, %len
  br i1 %abc, label %in.bounds, label %out.of.bounds

in.bounds:
  %addr = getelementptr i32, ptr %arr, i32 %idx
  store i32 0, ptr %addr
  %next = icmp ne i32 %idx.next, 100
  br i1 %next, label %loop, label %exit

out.of.bounds:
  ret void

exit:
  ret void
}

; Show that IRCE can turn 'ne' condition to 'ult' in increasing IV when IV is
; non-negative.
define void @test_01u(ptr %arr, ptr %a_len_ptr) #0 {

; CHECK:      test_01u
; CHECK:        main.exit.selector:
; CHECK-NEXT:     [[PSEUDO_PHI:%[^ ]+]] = phi i32 [ %idx.next, %in.bounds ]
; CHECK-NEXT:     [[COND:%[^ ]+]] = icmp ult i32 [[PSEUDO_PHI]], 100
; CHECK-NEXT:     br i1 [[COND]]

entry:
  %len = load i32, ptr %a_len_ptr, !range !0
  br label %loop

loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %in.bounds ]
  %idx.next = add i32 %idx, 1
  %abc = icmp slt i32 %idx, %len
  br i1 %abc, label %in.bounds, label %out.of.bounds

in.bounds:
  %addr = getelementptr i32, ptr %arr, i32 %idx
  store i32 0, ptr %addr
  %next = icmp ne i32 %idx.next, 100
  br i1 %next, label %loop, label %exit

out.of.bounds:
  ret void

exit:
  ret void
}

; Show that if n is not known to be greater than the starting value, IRCE
; doesn't apply.
define void @test_02(ptr %arr, ptr %a_len_ptr) #0 {

; CHECK: test_02(

entry:
  %len = load i32, ptr %a_len_ptr, !range !0
  br label %loop

loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %in.bounds ]
  %idx.next = add i32 %idx, 1
  %abc = icmp slt i32 %idx, %len
  br i1 %abc, label %in.bounds, label %out.of.bounds

in.bounds:
  %addr = getelementptr i32, ptr %arr, i32 %idx
  store i32 0, ptr %addr
  %next = icmp ne i32 %idx.next, -100
  br i1 %next, label %loop, label %exit

out.of.bounds:
  ret void

exit:
  ret void
}

; Show that IRCE can turn 'eq' condition to 'sge' in increasing IV.
define void @test_03(ptr %arr, ptr %a_len_ptr) #0 {

; CHECK: test_03(
; CHECK:        main.exit.selector:
; CHECK-NEXT:     [[PSEUDO_PHI:%[^ ]+]] = phi i32 [ %idx.next, %in.bounds ]
; CHECK-NEXT:     [[COND:%[^ ]+]] = icmp ult i32 [[PSEUDO_PHI]], 100
; CHECK-NEXT:     br i1 [[COND]]

entry:
  %len = load i32, ptr %a_len_ptr, !range !0
  br label %loop

loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %in.bounds ]
  %idx.next = add i32 %idx, 1
  %abc = icmp slt i32 %idx, %len
  br i1 %abc, label %in.bounds, label %out.of.bounds

in.bounds:
  %addr = getelementptr i32, ptr %arr, i32 %idx
  store i32 0, ptr %addr
  %next = icmp eq i32 %idx.next, 100
  br i1 %next, label %exit, label %loop

out.of.bounds:
  ret void

exit:
  ret void
}

define void @test_04(ptr %arr, ptr %a_len_ptr) #0 {

entry:
  %len = load i32, ptr %a_len_ptr, !range !0
  br label %loop

loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %in.bounds ]
  %idx.next = add i32 %idx, 1
  %abc = icmp slt i32 %idx, %len
  br i1 %abc, label %in.bounds, label %out.of.bounds

in.bounds:
  %addr = getelementptr i32, ptr %arr, i32 %idx
  store i32 0, ptr %addr
  %next = icmp eq i32 %idx.next, -100
  br i1 %next, label %exit, label %loop

out.of.bounds:
  ret void

exit:
  ret void
}

; Show that IRCE can turn 'ne' condition to 'sgt' in decreasing IV.
define void @test_05(ptr %arr, ptr %a_len_ptr) #0 {

; CHECK: test_05(
; CHECK:        preloop.exit.selector:
; CHECK-NEXT:     [[PSEUDO_PHI:%[^ ]+]] = phi i32 [ %idx.next.preloop, %in.bounds.preloop ]
; CHECK-NEXT:     [[COND:%[^ ]+]] = icmp sgt i32 [[PSEUDO_PHI]], 0
; CHECK-NEXT:     br i1 [[COND]]

entry:
  %len = load i32, ptr %a_len_ptr, !range !0
  br label %loop

loop:
  %idx = phi i32 [ 100, %entry ], [ %idx.next, %in.bounds ]
  %idx.next = add i32 %idx, -1
  %abc = icmp slt i32 %idx, %len
  br i1 %abc, label %in.bounds, label %out.of.bounds

in.bounds:
  %addr = getelementptr i32, ptr %arr, i32 %idx
  store i32 0, ptr %addr
  %next = icmp ne i32 %idx.next, 0
  br i1 %next, label %loop, label %exit

out.of.bounds:
  ret void

exit:
  ret void
}

; Show that IRCE cannot turn 'ne' condition to 'sgt' in decreasing IV if the end
; value is not proved to be less than the start value.
define void @test_06(ptr %arr, ptr %a_len_ptr) #0 {

; CHECK: test_06(

entry:
  %len = load i32, ptr %a_len_ptr, !range !0
  br label %loop

loop:
  %idx = phi i32 [ 100, %entry ], [ %idx.next, %in.bounds ]
  %idx.next = add i32 %idx, -1
  %abc = icmp slt i32 %idx, %len
  br i1 %abc, label %in.bounds, label %out.of.bounds

in.bounds:
  %addr = getelementptr i32, ptr %arr, i32 %idx
  store i32 0, ptr %addr
  %next = icmp ne i32 %idx.next, 120
  br i1 %next, label %loop, label %exit

out.of.bounds:
  ret void

exit:
  ret void
}

; Show that IRCE can turn 'eq' condition to 'slt' in decreasing IV.
define void @test_07(ptr %arr, ptr %a_len_ptr) #0 {

; CHECK: test_07(
; CHECK:        preloop.exit.selector:
; CHECK-NEXT:     [[PSEUDO_PHI:%[^ ]+]] = phi i32 [ %idx.next.preloop, %in.bounds.preloop ]
; CHECK-NEXT:     [[COND:%[^ ]+]] = icmp sgt i32 [[PSEUDO_PHI]], 0
; CHECK-NEXT:     br i1 [[COND]]

entry:
  %len = load i32, ptr %a_len_ptr, !range !0
  br label %loop

loop:
  %idx = phi i32 [ 100, %entry ], [ %idx.next, %in.bounds ]
  %idx.next = add i32 %idx, -1
  %abc = icmp slt i32 %idx, %len
  br i1 %abc, label %in.bounds, label %out.of.bounds

in.bounds:
  %addr = getelementptr i32, ptr %arr, i32 %idx
  store i32 0, ptr %addr
  %next = icmp eq i32 %idx.next, 0
  br i1 %next, label %exit, label %loop

out.of.bounds:
  ret void

exit:
  ret void
}

; Show that IRCE cannot turn 'eq' condition to 'slt' in decreasing IV if the end
; value is not proved to be less than the start value.
define void @test_08(ptr %arr, ptr %a_len_ptr) #0 {

; CHECK: test_08(

entry:
  %len = load i32, ptr %a_len_ptr, !range !0
  br label %loop

loop:
  %idx = phi i32 [ 100, %entry ], [ %idx.next, %in.bounds ]
  %idx.next = add i32 %idx, -1
  %abc = icmp slt i32 %idx, %len
  br i1 %abc, label %in.bounds, label %out.of.bounds

in.bounds:
  %addr = getelementptr i32, ptr %arr, i32 %idx
  store i32 0, ptr %addr
  %next = icmp eq i32 %idx.next, 120
  br i1 %next, label %exit, label %loop

out.of.bounds:
  ret void

exit:
  ret void
}

!0 = !{i32 0, i32 50}
