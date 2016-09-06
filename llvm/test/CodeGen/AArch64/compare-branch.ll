; RUN: llc -verify-machineinstrs -o - %s -mtriple=aarch64-linux-gnu | FileCheck %s

@var32 = global i32 0
@var64 = global i64 0

define void @foo() {
; CHECK-LABEL: foo:

  %val1 = load volatile i32, i32* @var32
  %tst1 = icmp eq i32 %val1, 0
  br i1 %tst1, label %end, label %test2, !prof !1
; CHECK: cbz {{w[0-9]+}}, .LBB

test2:
  %val2 = load volatile i32, i32* @var32
  %tst2 = icmp ne i32 %val2, 0
  br i1 %tst2, label %end, label %test3, !prof !1
; CHECK: cbnz {{w[0-9]+}}, .LBB

test3:
  %val3 = load volatile i64, i64* @var64
  %tst3 = icmp eq i64 %val3, 0
  br i1 %tst3, label %end, label %test4, !prof !1
; CHECK: cbz {{x[0-9]+}}, .LBB

test4:
  %val4 = load volatile i64, i64* @var64
  %tst4 = icmp ne i64 %val4, 0
  br i1 %tst4, label %end, label %test5, !prof !1
; CHECK: cbnz {{x[0-9]+}}, .LBB

test5:
  store volatile i64 %val4, i64* @var64
  ret void

end:
  ret void
}


!1 = !{!"branch_weights", i32 1, i32 1}
