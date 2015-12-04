; RUN: llc -mtriple=aarch64-apple-darwin -fast-isel -fast-isel-abort=1 -verify-machineinstrs < %s | FileCheck %s

; CHECK-LABEL: test_or
; CHECK:       cbnz w0, {{LBB[0-9]+_2}}
; CHECK:       cbz w1, {{LBB[0-9]+_1}}
define i64 @test_or(i32 %a, i32 %b) {
bb1:
  %0 = icmp eq i32 %a, 0
  %1 = icmp eq i32 %b, 0
  %or.cond = or i1 %0, %1
  br i1 %or.cond, label %bb3, label %bb4, !prof !0

bb3:
  ret i64 0

bb4:
  %2 = call i64 @bar()
  ret i64 %2
}

; CHECK-LABEL: test_and
; CHECK:       cbz w0, {{LBB[0-9]+_2}}
; CHECK:       cbnz w1, {{LBB[0-9]+_3}}
define i64 @test_and(i32 %a, i32 %b) {
bb1:
  %0 = icmp ne i32 %a, 0
  %1 = icmp ne i32 %b, 0
  %or.cond = and i1 %0, %1
  br i1 %or.cond, label %bb4, label %bb3, !prof !1

bb3:
  ret i64 0

bb4:
  %2 = call i64 @bar()
  ret i64 %2
}

; If the branch is unpredictable, don't add another branch.

; CHECK-LABEL: test_or_unpredictable
; CHECK:       cmp   w0, #0
; CHECK-NEXT:  cset  w8, eq
; CHECK-NEXT:  cmp   w1, #0
; CHECK-NEXT:  cset  w9, eq
; CHECK-NEXT:  orr   w8, w8, w9
; CHECK-NEXT:  tbnz w8, #0,
define i64 @test_or_unpredictable(i32 %a, i32 %b) {
bb1:
  %0 = icmp eq i32 %a, 0
  %1 = icmp eq i32 %b, 0
  %or.cond = or i1 %0, %1
  br i1 %or.cond, label %bb3, label %bb4, !unpredictable !2

bb3:
  ret i64 0

bb4:
  %2 = call i64 @bar()
  ret i64 %2
}

; CHECK-LABEL: test_and_unpredictable
; CHECK:       cmp   w0, #0
; CHECK-NEXT:  cset  w8, ne
; CHECK-NEXT:  cmp   w1, #0
; CHECK-NEXT:  cset  w9, ne
; CHECK-NEXT:  and   w8, w8, w9
; CHECK-NEXT:  tbz w8, #0,
define i64 @test_and_unpredictable(i32 %a, i32 %b) {
bb1:
  %0 = icmp ne i32 %a, 0
  %1 = icmp ne i32 %b, 0
  %or.cond = and i1 %0, %1
  br i1 %or.cond, label %bb4, label %bb3, !unpredictable !2

bb3:
  ret i64 0

bb4:
  %2 = call i64 @bar()
  ret i64 %2
}

declare i64 @bar()

!0 = !{!"branch_weights", i32 5128, i32 32}
!1 = !{!"branch_weights", i32 1024, i32 4136}
!2 = !{}

