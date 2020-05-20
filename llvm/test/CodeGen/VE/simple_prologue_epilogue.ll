; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

define void @func() {
; CHECK-LABEL: func:
; CHECK:       # %bb.0:
; CHECK-NEXT:  st %s9, (, %s11)
; CHECK-NEXT:  st %s10, 8(, %s11)
; CHECK-NEXT:  st %s15, 24(, %s11)
; CHECK-NEXT:  st %s16, 32(, %s11)
; CHECK-NEXT:  or %s9, 0, %s11
; CHECK-NEXT:  lea %s13, -176
; CHECK-NEXT:  and %s13, %s13, (32)0
; CHECK-NEXT:  lea.sl %s11, -1(%s13, %s11)
; CHECK-NEXT:  brge.l.t %s11, %s8, .LBB0_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:  ld %s61, 24(, %s14)
; CHECK-NEXT:  or %s62, 0, %s0
; CHECK-NEXT:  lea %s63, 315
; CHECK-NEXT:  shm.l %s63, (%s61)
; CHECK-NEXT:  shm.l %s8, 8(%s61)
; CHECK-NEXT:  shm.l %s11, 16(%s61)
; CHECK-NEXT:  monc
; CHECK-NEXT:  or %s0, 0, %s62
; CHECK-NEXT: .LBB0_2:
; CHECK-NEXT:  or %s11, 0, %s9
; CHECK-NEXT:  ld %s16, 32(, %s11)
; CHECK-NEXT:  ld %s15, 24(, %s11)
; CHECK-NEXT:  ld %s10, 8(, %s11)
; CHECK-NEXT:  ld %s9, (, %s11)
; CHECK-NEXT:  b.l.t (, %s10)
  ret void
}

define i64 @func1(i64) {
; CHECK-LABEL: func1:
; CHECK:       # %bb.0:
; CHECK-NEXT:  st %s9, (, %s11)
; CHECK-NEXT:  st %s10, 8(, %s11)
; CHECK-NEXT:  st %s15, 24(, %s11)
; CHECK-NEXT:  st %s16, 32(, %s11)
; CHECK-NEXT:  or %s9, 0, %s11
; CHECK-NEXT:  lea %s13, -176
; CHECK-NEXT:  and %s13, %s13, (32)0
; CHECK-NEXT:  lea.sl %s11, -1(%s13, %s11)
; CHECK-NEXT:  brge.l.t %s11, %s8, .LBB1_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:  ld %s61, 24(, %s14)
; CHECK-NEXT:  or %s62, 0, %s0
; CHECK-NEXT:  lea %s63, 315
; CHECK-NEXT:  shm.l %s63, (%s61)
; CHECK-NEXT:  shm.l %s8, 8(%s61)
; CHECK-NEXT:  shm.l %s11, 16(%s61)
; CHECK-NEXT:  monc
; CHECK-NEXT:  or %s0, 0, %s62
; CHECK-NEXT: .LBB1_2:
; CHECK-NEXT:  or %s11, 0, %s9
; CHECK-NEXT:  ld %s16, 32(, %s11)
; CHECK-NEXT:  ld %s15, 24(, %s11)
; CHECK-NEXT:  ld %s10, 8(, %s11)
; CHECK-NEXT:  ld %s9, (, %s11)
; CHECK-NEXT:  b.l.t (, %s10)
  ret i64 %0
}

define i64 @func2(i64, i64, i64, i64, i64) {
; CHECK-LABEL: func2:
; CHECK:       # %bb.0:
; CHECK-NEXT:  st %s9, (, %s11)
; CHECK-NEXT:  st %s10, 8(, %s11)
; CHECK-NEXT:  st %s15, 24(, %s11)
; CHECK-NEXT:  st %s16, 32(, %s11)
; CHECK-NEXT:  or %s9, 0, %s11
; CHECK-NEXT:  lea %s13, -176
; CHECK-NEXT:  and %s13, %s13, (32)0
; CHECK-NEXT:  lea.sl %s11, -1(%s13, %s11)
; CHECK-NEXT:  brge.l.t %s11, %s8, .LBB2_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:  ld %s61, 24(, %s14)
; CHECK-NEXT:  or %s62, 0, %s0
; CHECK-NEXT:  lea %s63, 315
; CHECK-NEXT:  shm.l %s63, (%s61)
; CHECK-NEXT:  shm.l %s8, 8(%s61)
; CHECK-NEXT:  shm.l %s11, 16(%s61)
; CHECK-NEXT:  monc
; CHECK-NEXT:  or %s0, 0, %s62
; CHECK-NEXT: .LBB2_2:
; CHECK-NEXT:  or %s0, 0, %s4
; CHECK-NEXT:  or %s11, 0, %s9
; CHECK-NEXT:  ld %s16, 32(, %s11)
; CHECK-NEXT:  ld %s15, 24(, %s11)
; CHECK-NEXT:  ld %s10, 8(, %s11)
; CHECK-NEXT:  ld %s9, (, %s11)
; CHECK-NEXT:  b.l.t (, %s10)
  ret i64 %4
}
