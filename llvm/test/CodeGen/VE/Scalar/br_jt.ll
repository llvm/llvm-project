; RUN: llc < %s -mtriple=ve | FileCheck %s
; RUN: llc < %s -mtriple=ve -relocation-model=pic \
; RUN:     | FileCheck %s -check-prefix=PIC

; Function Attrs: norecurse nounwind readnone
define signext i32 @br_jt(i32 signext %0) {
; CHECK-LABEL: br_jt:
; CHECK:       # %bb.0:
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    adds.w.sx %s1, -1, %s0
; CHECK-NEXT:    cmpu.w %s2, 3, %s1
; CHECK-NEXT:    brgt.w 0, %s2, .LBB{{[0-9]+}}_5
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    sll %s0, %s0, 3
; CHECK-NEXT:    lea %s1, .LJTI0_0@lo
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s1, .LJTI0_0@hi(, %s1)
; CHECK-NEXT:    ld %s1, (%s1, %s0)
; CHECK-NEXT:    or %s0, 3, (0)1
; CHECK-NEXT:    b.l.t (, %s1)
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
; CHECK-NEXT:  .LBB{{[0-9]+}}_3:
; CHECK-NEXT:    or %s0, 4, (0)1
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
; CHECK-NEXT:  .LBB{{[0-9]+}}_4:
; CHECK-NEXT:    or %s0, 7, (0)1
; CHECK-NEXT:  .LBB{{[0-9]+}}_5:
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
;
; PIC-LABEL: br_jt:
; PIC:       # %bb.0:
; PIC-NEXT:    st %s9, (, %s11)
; PIC-NEXT:    st %s10, 8(, %s11)
; PIC-NEXT:    st %s15, 24(, %s11)
; PIC-NEXT:    st %s16, 32(, %s11)
; PIC-NEXT:    or %s9, 0, %s11
; PIC-NEXT:    lea %s13, -176
; PIC-NEXT:    and %s13, %s13, (32)0
; PIC-NEXT:    lea.sl %s11, -1(%s13, %s11)
; PIC-NEXT:    brge.l %s11, %s8, .LBB0_7
; PIC-NEXT:  # %bb.6:
; PIC-NEXT:    ld %s61, 24(, %s14)
; PIC-NEXT:    or %s62, 0, %s0
; PIC-NEXT:    lea %s63, 315
; PIC-NEXT:    shm.l %s63, (%s61)
; PIC-NEXT:    shm.l %s8, 8(%s61)
; PIC-NEXT:    shm.l %s11, 16(%s61)
; PIC-NEXT:    monc
; PIC-NEXT:    or %s0, 0, %s62
; PIC-NEXT:  .LBB0_7:
; PIC-NEXT:    adds.w.sx %s0, %s0, (0)1
; PIC-NEXT:    adds.w.sx %s1, -1, %s0
; PIC-NEXT:    cmpu.w %s2, 3, %s1
; PIC-NEXT:    lea %s15, _GLOBAL_OFFSET_TABLE_@pc_lo(-24)
; PIC-NEXT:    and %s15, %s15, (32)0
; PIC-NEXT:    sic %s16
; PIC-NEXT:    lea.sl %s15, _GLOBAL_OFFSET_TABLE_@pc_hi(%s16, %s15)
; PIC-NEXT:    brgt.w 0, %s2, .LBB0_5
; PIC-NEXT:  # %bb.1:
; PIC-NEXT:    adds.w.zx %s0, %s1, (0)1
; PIC-NEXT:    sll %s0, %s0, 2
; PIC-NEXT:    lea %s1, .LJTI0_0@gotoff_lo
; PIC-NEXT:    and %s1, %s1, (32)0
; PIC-NEXT:    lea.sl %s1, .LJTI0_0@gotoff_hi(%s1, %s15)
; PIC-NEXT:    ldl.sx %s0, (%s1, %s0)
; PIC-NEXT:    lea %s1, br_jt@gotoff_lo
; PIC-NEXT:    and %s1, %s1, (32)0
; PIC-NEXT:    lea.sl %s1, br_jt@gotoff_hi(%s1, %s15)
; PIC-NEXT:    adds.l %s1, %s0, %s1
; PIC-NEXT:    or %s0, 3, (0)1
; PIC-NEXT:    b.l.t (, %s1)
; PIC-NEXT:  .LBB0_2:
; PIC-NEXT:    or %s0, 0, (0)1
; PIC-NEXT:    br.l.t .LBB0_5
; PIC-NEXT:  .LBB0_3:
; PIC-NEXT:    or %s0, 4, (0)1
; PIC-NEXT:    br.l.t .LBB0_5
; PIC-NEXT:  .LBB0_4:
; PIC-NEXT:    or %s0, 7, (0)1
; PIC-NEXT:  .LBB0_5:
; PIC-NEXT:    adds.w.sx %s0, %s0, (0)1
; PIC-NEXT:    or %s11, 0, %s9
; PIC-NEXT:    ld %s16, 32(, %s11)
; PIC-NEXT:    ld %s15, 24(, %s11)
; PIC-NEXT:    ld %s10, 8(, %s11)
; PIC-NEXT:    ld %s9, (, %s11)
; PIC-NEXT:    b.l.t (, %s10)
  switch i32 %0, label %5 [
    i32 1, label %6
    i32 2, label %2
    i32 3, label %3
    i32 4, label %4
  ]

2:                                                ; preds = %1
  br label %6

3:                                                ; preds = %1
  br label %6

4:                                                ; preds = %1
  br label %6

5:                                                ; preds = %1
  br label %6

6:                                                ; preds = %1, %5, %4, %3, %2
  %7 = phi i32 [ %0, %5 ], [ 7, %4 ], [ 4, %3 ], [ 0, %2 ], [ 3, %1 ]
  ret i32 %7
}
