; RUN: llc -disable-peephole -aarch64-enable-atomic-cfg-tidy=0 -verify-machineinstrs -mtriple=aarch64-apple-darwin < %s | FileCheck %s
; RUN: llc -disable-peephole -fast-isel -fast-isel-abort=1 -aarch64-enable-atomic-cfg-tidy=0 -verify-machineinstrs -mtriple=aarch64-apple-darwin < %s | FileCheck --check-prefix=CHECK --check-prefix=FAST %s

define i32 @icmp_eq_i8(i8 zeroext %a) {
; CHECK-LABEL: icmp_eq_i8
; CHECK:       tbz {{w[0-9]+}}, #0, {{LBB.+_2}}
  %1 = and i8 %a, 1
  %2 = icmp eq i8 %1, 0
  br i1 %2, label %bb1, label %bb2, !prof !0
bb1:
  ret i32 1
bb2:
  ret i32 0
}

define i32 @icmp_eq_i16(i16 zeroext %a) {
; CHECK-LABEL: icmp_eq_i16
; CHECK:       tbz w0, #1, {{LBB.+_2}}
  %1 = and i16 %a, 2
  %2 = icmp eq i16 %1, 0
  br i1 %2, label %bb1, label %bb2, !prof !0
bb1:
  ret i32 1
bb2:
  ret i32 0
}

define i32 @icmp_eq_i32(i32 %a) {
; CHECK-LABEL: icmp_eq_i32
; CHECK:       tbz w0, #2, {{LBB.+_2}}
  %1 = and i32 %a, 4
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %bb1, label %bb2, !prof !0
bb1:
  ret i32 1
bb2:
  ret i32 0
}

define i32 @icmp_eq_i64_1(i64 %a) {
; CHECK-LABEL: icmp_eq_i64_1
; CHECK:       tbz w0, #3, {{LBB.+_2}}
  %1 = and i64 %a, 8
  %2 = icmp eq i64 %1, 0
  br i1 %2, label %bb1, label %bb2, !prof !0
bb1:
  ret i32 1
bb2:
  ret i32 0
}

define i32 @icmp_eq_i64_2(i64 %a) {
; CHECK-LABEL: icmp_eq_i64_2
; CHECK:       tbz x0, #32, {{LBB.+_2}}
  %1 = and i64 %a, 4294967296
  %2 = icmp eq i64 %1, 0
  br i1 %2, label %bb1, label %bb2, !prof !0
bb1:
  ret i32 1
bb2:
  ret i32 0
}

define i32 @icmp_ne_i8(i8 zeroext %a) {
; CHECK-LABEL: icmp_ne_i8
; CHECK:       tbnz w0, #0, {{LBB.+_2}}
  %1 = and i8 %a, 1
  %2 = icmp ne i8 %1, 0
  br i1 %2, label %bb1, label %bb2, !prof !0
bb1:
  ret i32 1
bb2:
  ret i32 0
}

define i32 @icmp_ne_i16(i16 zeroext %a) {
; CHECK-LABEL: icmp_ne_i16
; CHECK:       tbnz w0, #1, {{LBB.+_2}}
  %1 = and i16 %a, 2
  %2 = icmp ne i16 %1, 0
  br i1 %2, label %bb1, label %bb2, !prof !0
bb1:
  ret i32 1
bb2:
  ret i32 0
}

define i32 @icmp_ne_i32(i32 %a) {
; CHECK-LABEL: icmp_ne_i32
; CHECK:       tbnz w0, #2, {{LBB.+_2}}
  %1 = and i32 %a, 4
  %2 = icmp ne i32 %1, 0
  br i1 %2, label %bb1, label %bb2, !prof !0
bb1:
  ret i32 1
bb2:
  ret i32 0
}

define i32 @icmp_ne_i64_1(i64 %a) {
; CHECK-LABEL: icmp_ne_i64_1
; CHECK:       tbnz w0, #3, {{LBB.+_2}}
  %1 = and i64 %a, 8
  %2 = icmp ne i64 %1, 0
  br i1 %2, label %bb1, label %bb2, !prof !0
bb1:
  ret i32 1
bb2:
  ret i32 0
}

define i32 @icmp_ne_i64_2(i64 %a) {
; CHECK-LABEL: icmp_ne_i64_2
; CHECK:       tbnz x0, #32, {{LBB.+_2}}
  %1 = and i64 %a, 4294967296
  %2 = icmp ne i64 %1, 0
  br i1 %2, label %bb1, label %bb2, !prof !0
bb1:
  ret i32 1
bb2:
  ret i32 0
}

define i32 @icmp_slt_i8(i8 zeroext %a) {
; FAST-LABEL: icmp_slt_i8
; FAST:       tbnz w0, #7, {{LBB.+_2}}
  %1 = icmp slt i8 %a, 0
  br i1 %1, label %bb1, label %bb2, !prof !0
bb1:
  ret i32 1
bb2:
  ret i32 0
}

define i32 @icmp_slt_i16(i16 zeroext %a) {
; FAST-LABEL: icmp_slt_i16
; FAST:       tbnz w0, #15, {{LBB.+_2}}
  %1 = icmp slt i16 %a, 0
  br i1 %1, label %bb1, label %bb2, !prof !0
bb1:
  ret i32 1
bb2:
  ret i32 0
}

define i32 @icmp_slt_i32(i32 %a) {
; CHECK-LABEL: icmp_slt_i32
; CHECK:       tbnz w0, #31, {{LBB.+_2}}
  %1 = icmp slt i32 %a, 0
  br i1 %1, label %bb1, label %bb2, !prof !0
bb1:
  ret i32 1
bb2:
  ret i32 0
}

define i32 @icmp_slt_i64(i64 %a) {
; CHECK-LABEL: icmp_slt_i64
; CHECK:       tbnz x0, #63, {{LBB.+_2}}
  %1 = icmp slt i64 %a, 0
  br i1 %1, label %bb1, label %bb2, !prof !0
bb1:
  ret i32 1
bb2:
  ret i32 0
}

define i32 @icmp_sge_i8(i8 zeroext %a) {
; FAST-LABEL: icmp_sge_i8
; FAST:       tbz w0, #7, {{LBB.+_2}}
  %1 = icmp sge i8 %a, 0
  br i1 %1, label %bb1, label %bb2, !prof !0
bb1:
  ret i32 1
bb2:
  ret i32 0
}

define i32 @icmp_sge_i16(i16 zeroext %a) {
; FAST-LABEL: icmp_sge_i16
; FAST:       tbz w0, #15, {{LBB.+_2}}
  %1 = icmp sge i16 %a, 0
  br i1 %1, label %bb1, label %bb2, !prof !0
bb1:
  ret i32 1
bb2:
  ret i32 0
}

define i32 @icmp_sle_i8(i8 zeroext %a) {
; FAST-LABEL: icmp_sle_i8
; FAST:       tbnz w0, #7, {{LBB.+_2}}
  %1 = icmp sle i8 %a, -1
  br i1 %1, label %bb1, label %bb2, !prof !0
bb1:
  ret i32 1
bb2:
  ret i32 0
}

define i32 @icmp_sle_i16(i16 zeroext %a) {
; FAST-LABEL: icmp_sle_i16
; FAST:       tbnz w0, #15, {{LBB.+_2}}
  %1 = icmp sle i16 %a, -1
  br i1 %1, label %bb1, label %bb2, !prof !0
bb1:
  ret i32 1
bb2:
  ret i32 0
}

define i32 @icmp_sle_i32(i32 %a) {
; CHECK-LABEL: icmp_sle_i32
; CHECK:       tbnz w0, #31, {{LBB.+_2}}
  %1 = icmp sle i32 %a, -1
  br i1 %1, label %bb1, label %bb2, !prof !0
bb1:
  ret i32 1
bb2:
  ret i32 0
}

define i32 @icmp_sle_i64(i64 %a) {
; CHECK-LABEL: icmp_sle_i64
; CHECK:       tbnz x0, #63, {{LBB.+_2}}
  %1 = icmp sle i64 %a, -1
  br i1 %1, label %bb1, label %bb2, !prof !0
bb1:
  ret i32 1
bb2:
  ret i32 0
}

define i32 @icmp_sgt_i8(i8 zeroext %a) {
; FAST-LABEL: icmp_sgt_i8
; FAST:       tbz w0, #7, {{LBB.+_2}}
  %1 = icmp sgt i8 %a, -1
  br i1 %1, label %bb1, label %bb2, !prof !0
bb1:
  ret i32 1
bb2:
  ret i32 0
}

define i32 @icmp_sgt_i16(i16 zeroext %a) {
; FAST-LABEL: icmp_sgt_i16
; FAST:       tbz w0, #15, {{LBB.+_2}}
  %1 = icmp sgt i16 %a, -1
  br i1 %1, label %bb1, label %bb2, !prof !0
bb1:
  ret i32 1
bb2:
  ret i32 0
}

define i32 @icmp_sgt_i32(i32 %a) {
; CHECK-LABEL: icmp_sgt_i32
; CHECK:       tbz w0, #31, {{LBB.+_2}}
  %1 = icmp sgt i32 %a, -1
  br i1 %1, label %bb1, label %bb2, !prof !0
bb1:
  ret i32 1
bb2:
  ret i32 0
}

define i32 @icmp_sgt_i64(i64 %a) {
; FAST-LABEL: icmp_sgt_i64
; FAST:       tbz x0, #63, {{LBB.+_2}}
  %1 = icmp sgt i64 %a, -1
  br i1 %1, label %bb1, label %bb2, !prof !0
bb1:
  ret i32 1
bb2:
  ret i32 0
}

; Test that we don't fold the 'and' instruction into the compare.
define i32 @icmp_eq_and_i32(i32 %a, i1 %c) {
; CHECK-LABEL: icmp_eq_and_i32
; CHECK:       and  [[REG:w[0-9]+]], w0, #0x4
; CHECK-NEXT:  cbz  [[REG]], {{LBB.+_3}}
  %1 = and i32 %a, 4
  br i1 %c, label %bb0, label %bb2
bb0:
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %bb1, label %bb2, !prof !0
bb1:
  ret i32 1
bb2:
  ret i32 0
}

!0 = !{!"branch_weights", i32 0, i32 2147483647}
!1 = !{!"branch_weights", i32 2147483647, i32 0}
