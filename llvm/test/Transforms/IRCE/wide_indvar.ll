; RUN: opt -verify-loop-info -irce-print-changed-loops -irce -irce-allow-narrow-latch=true -S < %s 2>&1 | FileCheck %s
; RUN: opt -verify-loop-info -irce-print-changed-loops -passes='require<branch-prob>,irce' -irce-allow-narrow-latch=true -S < %s 2>&1 | FileCheck %s

; Check that we can remove trivially non-failing range check.
define i32 @test_increasing_slt_slt_wide_simple_no_postloop() {

; CHECK-LABEL: @test_increasing_slt_slt_wide_simple_no_postloop(
; CHECK-NOT:   preloop
; CHECK-NOT:   postloop
; CHECK:       loop:
; CHECK:       br i1 true, label %backedge, label %check_failed

entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %backedge ]
  %rc = icmp slt i64 %iv, 100
  br i1 %rc, label %backedge, label %check_failed

backedge:
  %iv.next = add i64 %iv, 1
  %narrow.iv = trunc i64 %iv.next to i32
  %latch.cond = icmp slt i32 %narrow.iv, 100
  br i1 %latch.cond, label %loop, label %exit

exit:
  ret i32 %narrow.iv

check_failed:
  ret i32 -1
}

; This range check fails on the last iteration, so it needs a postloop.
define i32 @test_increasing_slt_slt_wide_simple_postloop() {

; CHECK-LABEL: @test_increasing_slt_slt_wide_simple_postloop(
; CHECK-NOT:   preloop
; CHECK:       loop:
; CHECK:       br i1 true, label %backedge, label %check_failed
; CHECK:       backedge
; CHECK:       [[COND:%[^ ]+]] = icmp slt i64 %wide.narrow.iv, 99
; CHECK:       br i1 [[COND]], label %loop, label %main.exit.selector
; CHECK:       postloop

entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %backedge ]
  %rc = icmp slt i64 %iv, 99
  br i1 %rc, label %backedge, label %check_failed

backedge:
  %iv.next = add i64 %iv, 1
  %narrow.iv = trunc i64 %iv.next to i32
  %latch.cond = icmp slt i32 %narrow.iv, 100
  br i1 %latch.cond, label %loop, label %exit

exit:
  ret i32 %narrow.iv

check_failed:
  ret i32 -1
}

; General case. If both %N and %M are non-negative, we do not need a preloop.
define i32 @test_increasing_slt_slt_wide_non-negative(ptr %n_ptr, ptr %m_ptr) {

; CHECK-LABEL: @test_increasing_slt_slt_wide_non-negative(
; CHECK-NOT:   preloop
; CHECK:       loop:
; CHECK:       br i1 true, label %backedge, label %check_failed
; CHECK:       backedge
; CHECK:       [[COND:%[^ ]+]] = icmp slt i64 %wide.narrow.iv, %exit.mainloop.at
; CHECK:       br i1 [[COND]], label %loop, label %main.exit.selector
; CHECK:       postloop

entry:
  %N = load i32, ptr %n_ptr, !range !2
  %M = load i64, ptr %m_ptr, !range !1
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %backedge ]
  %rc = icmp slt i64 %iv, %M
  br i1 %rc, label %backedge, label %check_failed

backedge:
  %iv.next = add i64 %iv, 1
  %narrow.iv = trunc i64 %iv.next to i32
  %latch.cond = icmp slt i32 %narrow.iv, %N
  br i1 %latch.cond, label %loop, label %exit

exit:
  ret i32 %narrow.iv

check_failed:
  ret i32 -1
}

; General case. Even though %M may be negative, we do not need a preloop because
; we make a non-negativity runtime check against M and do not go to main loop if
; M was negative.
define i32 @test_increasing_slt_slt_wide_general(ptr %n_ptr, ptr %m_ptr) {

; CHECK-LABEL: @test_increasing_slt_slt_wide_general(
; CHECK-NOT:   preloop
; CHECK:       loop:
; CHECK:       br i1 true, label %backedge, label %check_failed
; CHECK:       backedge
; CHECK:       [[COND:%[^ ]+]] = icmp slt i64
; CHECK:       br i1 [[COND]], label %loop, label %main.exit.selector
; CHECK:       postloop

entry:
  %N = load i32, ptr %n_ptr, !range !2
  %M = load i64, ptr %m_ptr
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %backedge ]
  %rc = icmp slt i64 %iv, %M
  br i1 %rc, label %backedge, label %check_failed

backedge:
  %iv.next = add i64 %iv, 1
  %narrow.iv = trunc i64 %iv.next to i32
  %latch.cond = icmp slt i32 %narrow.iv, %N
  br i1 %latch.cond, label %loop, label %exit

exit:
  ret i32 %narrow.iv

check_failed:
  ret i32 -1
}

; General case with preloop.
define i32 @test_increasing_slt_slt_wide_general_preloop(ptr %n_ptr, ptr %m_ptr) {

; CHECK-LABEL: @test_increasing_slt_slt_wide_general_preloop(
; CHECK:       loop:
; CHECK:       br i1 true, label %backedge, label %check_failed
; CHECK:       backedge
; CHECK:       [[COND:%[^ ]+]] = icmp slt i64
; CHECK:       br i1 [[COND]], label %loop, label %main.exit.selector
; CHECK:       preloop
; CHECK:       postloop

entry:
  %N = load i32, ptr %n_ptr, !range !2
  %M = load i64, ptr %m_ptr
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %backedge ]
  %rc = icmp slt i64 %iv, %M
  br i1 %rc, label %backedge, label %check_failed

backedge:
  %iv.next = add i64 %iv, 1
  %narrow.iv = trunc i64 %iv to i32
  %latch.cond = icmp slt i32 %narrow.iv, %N
  br i1 %latch.cond, label %loop, label %exit

exit:
  ret i32 %narrow.iv

check_failed:
  ret i32 -1
}

; Same as above, multiple checks.
define i32 @test_increasing_slt_slt_wide_multiple_checks(ptr %n_ptr, ptr %m1_ptr, ptr %m2_ptr, ptr %m3_ptr, ptr %m4_ptr) {
; CHECK-LABEL: @test_increasing_slt_slt_wide_multiple_checks(
; CHECK-NOT:   preloop
; CHECK:       loop:
; CHECK:       %c1 = and i1 true, true
; CHECK:       %c2 = and i1 %c1, true
; CHECK:       %rc = and i1 %c2, true
; CHECK:       br i1 %rc, label %backedge, label %check_failed.loopexit
; CHECK:       backedge
; CHECK:       [[COND:%[^ ]+]] = icmp slt i64
; CHECK:       br i1 [[COND]], label %loop, label %main.exit.selector
; CHECK:       postloop

entry:
  %N = load i32, ptr %n_ptr, !range !2
  %M1 = load i64, ptr %m1_ptr
  %M2 = load i64, ptr %m2_ptr
  %M3 = load i64, ptr %m3_ptr
  %M4 = load i64, ptr %m4_ptr
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %backedge ]
  %rc1 = icmp slt i64 %iv, %M1
  %rc2 = icmp slt i64 %iv, %M2
  %rc3 = icmp slt i64 %iv, %M3
  %rc4 = icmp slt i64 %iv, %M4
  %c1 = and i1 %rc1, %rc2
  %c2 = and i1 %c1, %rc3
  %rc = and i1 %c2, %rc4
  br i1 %rc, label %backedge, label %check_failed

backedge:
  %iv.next = add i64 %iv, 1
  %narrow.iv = trunc i64 %iv.next to i32
  %latch.cond = icmp slt i32 %narrow.iv, %N
  br i1 %latch.cond, label %loop, label %exit

exit:
  ret i32 %narrow.iv

check_failed:
  ret i32 -1
}

; Wide IV against narrow range check. We don't currently support it.
define i32 @test_increasing_slt_slt_wide_simple_negtest_narrow_rc() {

; CHECK-LABEL: @test_increasing_slt_slt_wide_simple_negtest_narrow_rc(
; CHECK-NOT:   i1 true
; CHECK-NOT:   main

entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %backedge ]
  %narrow.iv = trunc i64 %iv to i32
  %rc = icmp slt i32 %narrow.iv, 101
  br i1 %rc, label %backedge, label %check_failed

backedge:
  %iv.next = add i64 %iv, 1
  %latch.cond = icmp slt i64 %iv, 100
  br i1 %latch.cond, label %loop, label %exit

exit:
  ret i32 %narrow.iv

check_failed:
  ret i32 -1
}

; Check that we can remove trivially non-failing range check.
define i32 @test_increasing_ult_ult_wide_simple_no_postloop() {

; CHECK-LABEL: @test_increasing_ult_ult_wide_simple_no_postloop(
; CHECK-NOT:   preloop
; CHECK-NOT:   postloop
; CHECK:       loop:
; CHECK:       br i1 true, label %backedge, label %check_failed

entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %backedge ]
  %rc = icmp ult i64 %iv, 100
  br i1 %rc, label %backedge, label %check_failed

backedge:
  %iv.next = add i64 %iv, 1
  %narrow.iv = trunc i64 %iv.next to i32
  %latch.cond = icmp ult i32 %narrow.iv, 100
  br i1 %latch.cond, label %loop, label %exit

exit:
  ret i32 %narrow.iv

check_failed:
  ret i32 -1
}

; This range check fails on the last iteration, so it needs a postloop.
define i32 @test_increasing_ult_ult_wide_simple_postloop() {

; CHECK-LABEL: @test_increasing_ult_ult_wide_simple_postloop(
; CHECK-NOT:   preloop
; CHECK:       loop:
; CHECK:       br i1 true, label %backedge, label %check_failed
; CHECK:       backedge
; CHECK:       [[COND:%[^ ]+]] = icmp ult i64 %wide.narrow.iv, 99
; CHECK:       br i1 [[COND]], label %loop, label %main.exit.selector
; CHECK:       postloop

entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %backedge ]
  %rc = icmp ult i64 %iv, 99
  br i1 %rc, label %backedge, label %check_failed

backedge:
  %iv.next = add i64 %iv, 1
  %narrow.iv = trunc i64 %iv.next to i32
  %latch.cond = icmp ult i32 %narrow.iv, 100
  br i1 %latch.cond, label %loop, label %exit

exit:
  ret i32 %narrow.iv

check_failed:
  ret i32 -1
}

; General case. If both %N and %M are non-negative, we do not need a preloop.
define i32 @test_increasing_ult_ult_wide_non-negative(ptr %n_ptr, ptr %m_ptr) {

; CHECK-LABEL: @test_increasing_ult_ult_wide_non-negative(
; CHECK-NOT:   preloop
; CHECK:       loop:
; CHECK:       br i1 true, label %backedge, label %check_failed
; CHECK:       backedge
; CHECK:       [[COND:%[^ ]+]] = icmp ult i64 %wide.narrow.iv, %exit.mainloop.at
; CHECK:       br i1 [[COND]], label %loop, label %main.exit.selector
; CHECK:       postloop

entry:
  %N = load i32, ptr %n_ptr, !range !2
  %M = load i64, ptr %m_ptr, !range !1
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %backedge ]
  %rc = icmp ult i64 %iv, %M
  br i1 %rc, label %backedge, label %check_failed

backedge:
  %iv.next = add i64 %iv, 1
  %narrow.iv = trunc i64 %iv.next to i32
  %latch.cond = icmp ult i32 %narrow.iv, %N
  br i1 %latch.cond, label %loop, label %exit

exit:
  ret i32 %narrow.iv

check_failed:
  ret i32 -1
}

; General case. Even though %M may be negative, we do not need a preloop because
; we make a non-negativity runtime check against M and do not go to main loop if
; M was negative.
define i32 @test_increasing_ult_ult_wide_general(ptr %n_ptr, ptr %m_ptr) {

; CHECK-LABEL: @test_increasing_ult_ult_wide_general(
; CHECK-NOT:   preloop
; CHECK:       loop:
; CHECK:       br i1 true, label %backedge, label %check_failed
; CHECK:       backedge
; CHECK:       [[COND:%[^ ]+]] = icmp ult i64
; CHECK:       br i1 [[COND]], label %loop, label %main.exit.selector
; CHECK:       postloop

entry:
  %N = load i32, ptr %n_ptr, !range !2
  %M = load i64, ptr %m_ptr
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %backedge ]
  %rc = icmp ult i64 %iv, %M
  br i1 %rc, label %backedge, label %check_failed

backedge:
  %iv.next = add i64 %iv, 1
  %narrow.iv = trunc i64 %iv.next to i32
  %latch.cond = icmp ult i32 %narrow.iv, %N
  br i1 %latch.cond, label %loop, label %exit

exit:
  ret i32 %narrow.iv

check_failed:
  ret i32 -1
}

; Same as above, multiple checks.
define i32 @test_increasing_ult_ult_wide_multiple_checks(ptr %n_ptr, ptr %m1_ptr, ptr %m2_ptr, ptr %m3_ptr, ptr %m4_ptr) {
; CHECK-LABEL: @test_increasing_ult_ult_wide_multiple_checks(
; CHECK-NOT:   preloop
; CHECK:       loop:
; CHECK:       %c1 = and i1 true, true
; CHECK:       %c2 = and i1 %c1, true
; CHECK:       %rc = and i1 %c2, true
; CHECK:       br i1 %rc, label %backedge, label %check_failed.loopexit
; CHECK:       backedge
; CHECK:       [[COND:%[^ ]+]] = icmp ult i64
; CHECK:       br i1 [[COND]], label %loop, label %main.exit.selector
; CHECK:       postloop

entry:
  %N = load i32, ptr %n_ptr, !range !2
  %M1 = load i64, ptr %m1_ptr
  %M2 = load i64, ptr %m2_ptr
  %M3 = load i64, ptr %m3_ptr
  %M4 = load i64, ptr %m4_ptr
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %backedge ]
  %rc1 = icmp ult i64 %iv, %M1
  %rc2 = icmp ult i64 %iv, %M2
  %rc3 = icmp ult i64 %iv, %M3
  %rc4 = icmp ult i64 %iv, %M4
  %c1 = and i1 %rc1, %rc2
  %c2 = and i1 %c1, %rc3
  %rc = and i1 %c2, %rc4
  br i1 %rc, label %backedge, label %check_failed

backedge:
  %iv.next = add i64 %iv, 1
  %narrow.iv = trunc i64 %iv.next to i32
  %latch.cond = icmp ult i32 %narrow.iv, %N
  br i1 %latch.cond, label %loop, label %exit

exit:
  ret i32 %narrow.iv

check_failed:
  ret i32 -1
}

; Wide IV against narrow range check. We don't currently support it.
define i32 @test_increasing_ult_ult_wide_simple_negtest_narrow_rc() {

; CHECK-LABEL: @test_increasing_ult_ult_wide_simple_negtest_narrow_rc(
; CHECK-NOT:   i1 true
; CHECK-NOT:   main

entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %backedge ]
  %narrow.iv = trunc i64 %iv to i32
  %rc = icmp ult i32 %narrow.iv, 101
  br i1 %rc, label %backedge, label %check_failed

backedge:
  %iv.next = add i64 %iv, 1
  %latch.cond = icmp ult i64 %iv, 100
  br i1 %latch.cond, label %loop, label %exit

exit:
  ret i32 %narrow.iv

check_failed:
  ret i32 -1
}

!0 = !{i32 0, i32 2147483647}
!1 = !{i64 0, i64 9223372036854775807}
!2 = !{i32 1, i32 2147483647}
