; RUN: llc < %s -mtriple=ve | FileCheck %s
; RUN: llc < %s -mtriple=ve -relocation-model=pic \
; RUN:     | FileCheck %s -check-prefix=PIC

@switch.table.br_jt4 = private unnamed_addr constant [4 x i32] [i32 3, i32 0, i32 4, i32 7], align 4
@switch.table.br_jt7 = private unnamed_addr constant [9 x i32] [i32 3, i32 0, i32 4, i32 7, i32 3, i32 3, i32 5, i32 11, i32 10], align 4
@switch.table.br_jt8 = private unnamed_addr constant [9 x i32] [i32 3, i32 0, i32 4, i32 7, i32 3, i32 1, i32 5, i32 11, i32 10], align 4

; Function Attrs: norecurse nounwind readnone
define signext i32 @br_jt3(i32 signext %0) {
; CHECK-LABEL: br_jt3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    breq.w 1, %s0, .LBB{{[0-9]+}}_1
; CHECK-NEXT:  # %bb.2:
; CHECK-NEXT:    breq.w 4, %s0, .LBB{{[0-9]+}}_5
; CHECK-NEXT:  # %bb.3:
; CHECK-NEXT:    brne.w 2, %s0, .LBB{{[0-9]+}}_6
; CHECK-NEXT:  # %bb.4:
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
; CHECK-NEXT:  .LBB{{[0-9]+}}_1:
; CHECK-NEXT:    or %s0, 3, (0)1
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
; CHECK-NEXT:  .LBB{{[0-9]+}}_5:
; CHECK-NEXT:    or %s0, 7, (0)1
; CHECK-NEXT:  .LBB{{[0-9]+}}_6:
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
;
; PIC-LABEL: br_jt3:
; PIC:       # %bb.0:
; PIC-NEXT:    and %s0, %s0, (32)0
; PIC-NEXT:    breq.w 1, %s0, .LBB0_1
; PIC-NEXT:  # %bb.2:
; PIC-NEXT:    breq.w 4, %s0, .LBB0_5
; PIC-NEXT:  # %bb.3:
; PIC-NEXT:    brne.w 2, %s0, .LBB0_6
; PIC-NEXT:  # %bb.4:
; PIC-NEXT:    or %s0, 0, (0)1
; PIC-NEXT:    adds.w.sx %s0, %s0, (0)1
; PIC-NEXT:    b.l.t (, %s10)
; PIC-NEXT:  .LBB0_1:
; PIC-NEXT:    or %s0, 3, (0)1
; PIC-NEXT:    adds.w.sx %s0, %s0, (0)1
; PIC-NEXT:    b.l.t (, %s10)
; PIC-NEXT:  .LBB0_5:
; PIC-NEXT:    or %s0, 7, (0)1
; PIC-NEXT:  .LBB0_6:
; PIC-NEXT:    adds.w.sx %s0, %s0, (0)1
; PIC-NEXT:    b.l.t (, %s10)
  switch i32 %0, label %4 [
    i32 1, label %5
    i32 2, label %2
    i32 4, label %3
  ]

2:                                                ; preds = %1
  br label %5

3:                                                ; preds = %1
  br label %5

4:                                                ; preds = %1
  br label %5

5:                                                ; preds = %1, %4, %3, %2
  %6 = phi i32 [ %0, %4 ], [ 7, %3 ], [ 0, %2 ], [ 3, %1 ]
  ret i32 %6
}

; Function Attrs: norecurse nounwind readnone
define signext i32 @br_jt4(i32 signext %0) {
; CHECK-LABEL: br_jt4:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    adds.w.sx %s1, -1, %s0
; CHECK-NEXT:    cmpu.w %s2, 3, %s1
; CHECK-NEXT:    brgt.w 0, %s2, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    adds.w.sx %s0, %s1, (0)1
; CHECK-NEXT:    sll %s0, %s0, 2
; CHECK-NEXT:    lea %s1, .Lswitch.table.br_jt4@lo
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s1, .Lswitch.table.br_jt4@hi(, %s1)
; CHECK-NEXT:    ldl.sx %s0, (%s0, %s1)
; CHECK-NEXT:    b.l.t (, %s10)
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
;
; PIC-LABEL: br_jt4:
; PIC:       # %bb.0:
; PIC-NEXT:    st %s15, 24(, %s11)
; PIC-NEXT:    st %s16, 32(, %s11)
; PIC-NEXT:    and %s0, %s0, (32)0
; PIC-NEXT:    adds.w.sx %s1, -1, %s0
; PIC-NEXT:    cmpu.w %s2, 3, %s1
; PIC-NEXT:    lea %s15, _GLOBAL_OFFSET_TABLE_@pc_lo(-24)
; PIC-NEXT:    and %s15, %s15, (32)0
; PIC-NEXT:    sic %s16
; PIC-NEXT:    lea.sl %s15, _GLOBAL_OFFSET_TABLE_@pc_hi(%s16, %s15)
; PIC-NEXT:    brgt.w 0, %s2, .LBB1_2
; PIC-NEXT:  # %bb.1:
; PIC-NEXT:    adds.w.sx %s0, %s1, (0)1
; PIC-NEXT:    sll %s0, %s0, 2
; PIC-NEXT:    lea %s1, .Lswitch.table.br_jt4@gotoff_lo
; PIC-NEXT:    and %s1, %s1, (32)0
; PIC-NEXT:    lea.sl %s1, .Lswitch.table.br_jt4@gotoff_hi(%s1, %s15)
; PIC-NEXT:    ldl.sx %s0, (%s0, %s1)
; PIC-NEXT:    br.l.t .LBB1_3
; PIC-NEXT:  .LBB1_2:
; PIC-NEXT:    adds.w.sx %s0, %s0, (0)1
; PIC-NEXT:  .LBB1_3:
; PIC-NEXT:    ld %s16, 32(, %s11)
; PIC-NEXT:    ld %s15, 24(, %s11)
; PIC-NEXT:    b.l.t (, %s10)
  %2 = add i32 %0, -1
  %3 = icmp ult i32 %2, 4
  br i1 %3, label %4, label %8

4:                                                ; preds = %1
  %5 = sext i32 %2 to i64
  %6 = getelementptr inbounds [4 x i32], ptr @switch.table.br_jt4, i64 0, i64 %5
  %7 = load i32, ptr %6, align 4
  ret i32 %7

8:                                                ; preds = %1
  ret i32 %0
}

; Function Attrs: norecurse nounwind readnone
define signext i32 @br_jt7(i32 signext %0) {
; CHECK-LABEL: br_jt7:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    adds.w.sx %s1, -1, %s0
; CHECK-NEXT:    cmpu.w %s2, 8, %s1
; CHECK-NEXT:    brgt.w 0, %s2, .LBB{{[0-9]+}}_3
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    and %s2, %s1, (48)0
; CHECK-NEXT:    lea %s3, 463
; CHECK-NEXT:    and %s3, %s3, (32)0
; CHECK-NEXT:    srl %s2, %s3, %s2
; CHECK-NEXT:    and %s2, 1, %s2
; CHECK-NEXT:    brne.w 0, %s2, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  .LBB{{[0-9]+}}_3:
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s0, %s1, (0)1
; CHECK-NEXT:    sll %s0, %s0, 2
; CHECK-NEXT:    lea %s1, .Lswitch.table.br_jt7@lo
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s1, .Lswitch.table.br_jt7@hi(, %s1)
; CHECK-NEXT:    ldl.sx %s0, (%s0, %s1)
; CHECK-NEXT:    b.l.t (, %s10)
;
; PIC-LABEL: br_jt7:
; PIC:       # %bb.0:
; PIC-NEXT:    st %s15, 24(, %s11)
; PIC-NEXT:    st %s16, 32(, %s11)
; PIC-NEXT:    and %s0, %s0, (32)0
; PIC-NEXT:    adds.w.sx %s1, -1, %s0
; PIC-NEXT:    cmpu.w %s2, 8, %s1
; PIC-NEXT:    lea %s15, _GLOBAL_OFFSET_TABLE_@pc_lo(-24)
; PIC-NEXT:    and %s15, %s15, (32)0
; PIC-NEXT:    sic %s16
; PIC-NEXT:    lea.sl %s15, _GLOBAL_OFFSET_TABLE_@pc_hi(%s16, %s15)
; PIC-NEXT:    brgt.w 0, %s2, .LBB2_3
; PIC-NEXT:  # %bb.1:
; PIC-NEXT:    and %s2, %s1, (48)0
; PIC-NEXT:    lea %s3, 463
; PIC-NEXT:    and %s3, %s3, (32)0
; PIC-NEXT:    srl %s2, %s3, %s2
; PIC-NEXT:    and %s2, 1, %s2
; PIC-NEXT:    brne.w 0, %s2, .LBB2_2
; PIC-NEXT:  .LBB2_3:
; PIC-NEXT:    adds.w.sx %s0, %s0, (0)1
; PIC-NEXT:    br.l.t .LBB2_4
; PIC-NEXT:  .LBB2_2:
; PIC-NEXT:    adds.w.sx %s0, %s1, (0)1
; PIC-NEXT:    sll %s0, %s0, 2
; PIC-NEXT:    lea %s1, .Lswitch.table.br_jt7@gotoff_lo
; PIC-NEXT:    and %s1, %s1, (32)0
; PIC-NEXT:    lea.sl %s1, .Lswitch.table.br_jt7@gotoff_hi(%s1, %s15)
; PIC-NEXT:    ldl.sx %s0, (%s0, %s1)
; PIC-NEXT:  .LBB2_4:
; PIC-NEXT:    ld %s16, 32(, %s11)
; PIC-NEXT:    ld %s15, 24(, %s11)
; PIC-NEXT:    b.l.t (, %s10)
  %2 = add i32 %0, -1
  %3 = icmp ult i32 %2, 9
  br i1 %3, label %4, label %13

4:                                                ; preds = %1
  %5 = trunc i32 %2 to i16
  %6 = lshr i16 463, %5
  %7 = and i16 %6, 1
  %8 = icmp eq i16 %7, 0
  br i1 %8, label %13, label %9

9:                                                ; preds = %4
  %10 = sext i32 %2 to i64
  %11 = getelementptr inbounds [9 x i32], ptr @switch.table.br_jt7, i64 0, i64 %10
  %12 = load i32, ptr %11, align 4
  ret i32 %12

13:                                               ; preds = %1, %4
  ret i32 %0
}

; Function Attrs: norecurse nounwind readnone
define signext i32 @br_jt8(i32 signext %0) {
; CHECK-LABEL: br_jt8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    adds.w.sx %s1, -1, %s0
; CHECK-NEXT:    cmpu.w %s2, 8, %s1
; CHECK-NEXT:    brgt.w 0, %s2, .LBB{{[0-9]+}}_3
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    and %s2, %s1, (48)0
; CHECK-NEXT:    lea %s3, 495
; CHECK-NEXT:    and %s3, %s3, (32)0
; CHECK-NEXT:    srl %s2, %s3, %s2
; CHECK-NEXT:    and %s2, 1, %s2
; CHECK-NEXT:    brne.w 0, %s2, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  .LBB{{[0-9]+}}_3:
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s0, %s1, (0)1
; CHECK-NEXT:    sll %s0, %s0, 2
; CHECK-NEXT:    lea %s1, .Lswitch.table.br_jt8@lo
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s1, .Lswitch.table.br_jt8@hi(, %s1)
; CHECK-NEXT:    ldl.sx %s0, (%s0, %s1)
; CHECK-NEXT:    b.l.t (, %s10)
;
; PIC-LABEL: br_jt8:
; PIC:       # %bb.0:
; PIC-NEXT:    st %s15, 24(, %s11)
; PIC-NEXT:    st %s16, 32(, %s11)
; PIC-NEXT:    and %s0, %s0, (32)0
; PIC-NEXT:    adds.w.sx %s1, -1, %s0
; PIC-NEXT:    cmpu.w %s2, 8, %s1
; PIC-NEXT:    lea %s15, _GLOBAL_OFFSET_TABLE_@pc_lo(-24)
; PIC-NEXT:    and %s15, %s15, (32)0
; PIC-NEXT:    sic %s16
; PIC-NEXT:    lea.sl %s15, _GLOBAL_OFFSET_TABLE_@pc_hi(%s16, %s15)
; PIC-NEXT:    brgt.w 0, %s2, .LBB3_3
; PIC-NEXT:  # %bb.1:
; PIC-NEXT:    and %s2, %s1, (48)0
; PIC-NEXT:    lea %s3, 495
; PIC-NEXT:    and %s3, %s3, (32)0
; PIC-NEXT:    srl %s2, %s3, %s2
; PIC-NEXT:    and %s2, 1, %s2
; PIC-NEXT:    brne.w 0, %s2, .LBB3_2
; PIC-NEXT:  .LBB3_3:
; PIC-NEXT:    adds.w.sx %s0, %s0, (0)1
; PIC-NEXT:    br.l.t .LBB3_4
; PIC-NEXT:  .LBB3_2:
; PIC-NEXT:    adds.w.sx %s0, %s1, (0)1
; PIC-NEXT:    sll %s0, %s0, 2
; PIC-NEXT:    lea %s1, .Lswitch.table.br_jt8@gotoff_lo
; PIC-NEXT:    and %s1, %s1, (32)0
; PIC-NEXT:    lea.sl %s1, .Lswitch.table.br_jt8@gotoff_hi(%s1, %s15)
; PIC-NEXT:    ldl.sx %s0, (%s0, %s1)
; PIC-NEXT:  .LBB3_4:
; PIC-NEXT:    ld %s16, 32(, %s11)
; PIC-NEXT:    ld %s15, 24(, %s11)
; PIC-NEXT:    b.l.t (, %s10)
  %2 = add i32 %0, -1
  %3 = icmp ult i32 %2, 9
  br i1 %3, label %4, label %13

4:                                                ; preds = %1
  %5 = trunc i32 %2 to i16
  %6 = lshr i16 495, %5
  %7 = and i16 %6, 1
  %8 = icmp eq i16 %7, 0
  br i1 %8, label %13, label %9

9:                                                ; preds = %4
  %10 = sext i32 %2 to i64
  %11 = getelementptr inbounds [9 x i32], ptr @switch.table.br_jt8, i64 0, i64 %10
  %12 = load i32, ptr %11, align 4
  ret i32 %12

13:                                               ; preds = %1, %4
  ret i32 %0
}

; Function Attrs: norecurse nounwind readnone
define signext i32 @br_jt3_m(i32 signext %0, i32 signext %1) {
; CHECK-LABEL: br_jt3_m:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    breq.w 1, %s0, .LBB{{[0-9]+}}_1
; CHECK-NEXT:  # %bb.2:
; CHECK-NEXT:    breq.w 4, %s0, .LBB{{[0-9]+}}_5
; CHECK-NEXT:  # %bb.3:
; CHECK-NEXT:    brne.w 2, %s0, .LBB{{[0-9]+}}_6
; CHECK-NEXT:  # %bb.4:
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
; CHECK-NEXT:  .LBB{{[0-9]+}}_1:
; CHECK-NEXT:    or %s0, 3, (0)1
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
; CHECK-NEXT:  .LBB{{[0-9]+}}_5:
; CHECK-NEXT:    and %s0, %s1, (32)0
; CHECK-NEXT:    adds.w.sx %s0, 3, %s0
; CHECK-NEXT:  .LBB{{[0-9]+}}_6:
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
;
; PIC-LABEL: br_jt3_m:
; PIC:       # %bb.0:
; PIC-NEXT:    and %s0, %s0, (32)0
; PIC-NEXT:    breq.w 1, %s0, .LBB4_1
; PIC-NEXT:  # %bb.2:
; PIC-NEXT:    breq.w 4, %s0, .LBB4_5
; PIC-NEXT:  # %bb.3:
; PIC-NEXT:    brne.w 2, %s0, .LBB4_6
; PIC-NEXT:  # %bb.4:
; PIC-NEXT:    or %s0, 0, (0)1
; PIC-NEXT:    adds.w.sx %s0, %s0, (0)1
; PIC-NEXT:    b.l.t (, %s10)
; PIC-NEXT:  .LBB4_1:
; PIC-NEXT:    or %s0, 3, (0)1
; PIC-NEXT:    adds.w.sx %s0, %s0, (0)1
; PIC-NEXT:    b.l.t (, %s10)
; PIC-NEXT:  .LBB4_5:
; PIC-NEXT:    and %s0, %s1, (32)0
; PIC-NEXT:    adds.w.sx %s0, 3, %s0
; PIC-NEXT:  .LBB4_6:
; PIC-NEXT:    adds.w.sx %s0, %s0, (0)1
; PIC-NEXT:    b.l.t (, %s10)
  switch i32 %0, label %6 [
    i32 1, label %7
    i32 2, label %3
    i32 4, label %4
  ]

3:                                                ; preds = %2
  br label %7

4:                                                ; preds = %2
  %5 = add nsw i32 %1, 3
  br label %7

6:                                                ; preds = %2
  br label %7

7:                                                ; preds = %2, %6, %4, %3
  %8 = phi i32 [ %0, %6 ], [ %5, %4 ], [ 0, %3 ], [ 3, %2 ]
  ret i32 %8
}

; Function Attrs: norecurse nounwind readnone
define signext i32 @br_jt4_m(i32 signext %0, i32 signext %1) {
; CHECK-LABEL: br_jt4_m:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    adds.w.sx %s2, -1, %s0
; CHECK-NEXT:    cmpu.w %s3, 3, %s2
; CHECK-NEXT:    brgt.w 0, %s3, .LBB{{[0-9]+}}_5
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    adds.w.zx %s0, %s2, (0)1
; CHECK-NEXT:    sll %s0, %s0, 3
; CHECK-NEXT:    lea %s2, .LJTI5_0@lo
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lea.sl %s2, .LJTI5_0@hi(, %s2)
; CHECK-NEXT:    ld %s2, (%s2, %s0)
; CHECK-NEXT:    or %s0, 3, (0)1
; CHECK-NEXT:    b.l.t (, %s2)
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
; CHECK-NEXT:  .LBB{{[0-9]+}}_3:
; CHECK-NEXT:    or %s0, 4, (0)1
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
; CHECK-NEXT:  .LBB{{[0-9]+}}_4:
; CHECK-NEXT:    and %s0, %s1, (32)0
; CHECK-NEXT:    adds.w.sx %s0, 3, %s0
; CHECK-NEXT:  .LBB{{[0-9]+}}_5:
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
;
; PIC-LABEL: br_jt4_m:
; PIC:       # %bb.0:
; PIC-NEXT:    and %s0, %s0, (32)0
; PIC-NEXT:    brlt.w 2, %s0, .LBB5_4
; PIC-NEXT:  # %bb.1:
; PIC-NEXT:    breq.w 1, %s0, .LBB5_8
; PIC-NEXT:  # %bb.2:
; PIC-NEXT:    brne.w 2, %s0, .LBB5_7
; PIC-NEXT:  # %bb.3:
; PIC-NEXT:    or %s0, 0, (0)1
; PIC-NEXT:    adds.w.sx %s0, %s0, (0)1
; PIC-NEXT:    b.l.t (, %s10)
; PIC-NEXT:  .LBB5_4:
; PIC-NEXT:    breq.w 3, %s0, .LBB5_9
; PIC-NEXT:  # %bb.5:
; PIC-NEXT:    brne.w 4, %s0, .LBB5_7
; PIC-NEXT:  # %bb.6:
; PIC-NEXT:    and %s0, %s1, (32)0
; PIC-NEXT:    adds.w.sx %s0, 3, %s0
; PIC-NEXT:  .LBB5_7:
; PIC-NEXT:    adds.w.sx %s0, %s0, (0)1
; PIC-NEXT:    b.l.t (, %s10)
; PIC-NEXT:  .LBB5_8:
; PIC-NEXT:    or %s0, 3, (0)1
; PIC-NEXT:    adds.w.sx %s0, %s0, (0)1
; PIC-NEXT:    b.l.t (, %s10)
; PIC-NEXT:  .LBB5_9:
; PIC-NEXT:    or %s0, 4, (0)1
; PIC-NEXT:    adds.w.sx %s0, %s0, (0)1
; PIC-NEXT:    b.l.t (, %s10)
  switch i32 %0, label %7 [
    i32 1, label %8
    i32 2, label %3
    i32 3, label %4
    i32 4, label %5
  ]

3:                                                ; preds = %2
  br label %8

4:                                                ; preds = %2
  br label %8

5:                                                ; preds = %2
  %6 = add nsw i32 %1, 3
  br label %8

7:                                                ; preds = %2
  br label %8

8:                                                ; preds = %2, %7, %5, %4, %3
  %9 = phi i32 [ %0, %7 ], [ %6, %5 ], [ 4, %4 ], [ 0, %3 ], [ 3, %2 ]
  ret i32 %9
}

; Function Attrs: norecurse nounwind readnone
define signext i32 @br_jt7_m(i32 signext %0, i32 signext %1) {
; CHECK-LABEL: br_jt7_m:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s2, %s0, (32)0
; CHECK-NEXT:    adds.w.sx %s0, -1, %s2
; CHECK-NEXT:    cmpu.w %s3, 8, %s0
; CHECK-NEXT:    brgt.w 0, %s3, .LBB{{[0-9]+}}_8
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    sll %s0, %s0, 3
; CHECK-NEXT:    lea %s3, .LJTI6_0@lo
; CHECK-NEXT:    and %s3, %s3, (32)0
; CHECK-NEXT:    lea.sl %s3, .LJTI6_0@hi(, %s3)
; CHECK-NEXT:    ld %s3, (%s3, %s0)
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    or %s0, 3, (0)1
; CHECK-NEXT:    b.l.t (, %s3)
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
; CHECK-NEXT:  .LBB{{[0-9]+}}_3:
; CHECK-NEXT:    or %s0, 4, (0)1
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
; CHECK-NEXT:  .LBB{{[0-9]+}}_4:
; CHECK-NEXT:    adds.w.sx %s0, 3, %s1
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
; CHECK-NEXT:  .LBB{{[0-9]+}}_8:
; CHECK-NEXT:    or %s0, 0, %s2
; CHECK-NEXT:  .LBB{{[0-9]+}}_9:
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
; CHECK-NEXT:  .LBB{{[0-9]+}}_7:
; CHECK-NEXT:    or %s0, 11, (0)1
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
; CHECK-NEXT:  .LBB{{[0-9]+}}_6:
; CHECK-NEXT:    or %s0, 10, (0)1
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
; CHECK-NEXT:  .LBB{{[0-9]+}}_5:
; CHECK-NEXT:    adds.w.sx %s0, -2, %s1
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
;
; PIC-LABEL: br_jt7_m:
; PIC:       # %bb.0:
; PIC-NEXT:    and %s0, %s0, (32)0
; PIC-NEXT:    brge.w 3, %s0, .LBB6_1
; PIC-NEXT:  # %bb.6:
; PIC-NEXT:    brlt.w 7, %s0, .LBB6_10
; PIC-NEXT:  # %bb.7:
; PIC-NEXT:    and %s1, %s1, (32)0
; PIC-NEXT:    breq.w 4, %s0, .LBB6_14
; PIC-NEXT:  # %bb.8:
; PIC-NEXT:    brne.w 7, %s0, .LBB6_16
; PIC-NEXT:  # %bb.9:
; PIC-NEXT:    adds.w.sx %s0, -2, %s1
; PIC-NEXT:    adds.w.sx %s0, %s0, (0)1
; PIC-NEXT:    b.l.t (, %s10)
; PIC-NEXT:  .LBB6_1:
; PIC-NEXT:    breq.w 1, %s0, .LBB6_2
; PIC-NEXT:  # %bb.3:
; PIC-NEXT:    breq.w 2, %s0, .LBB6_13
; PIC-NEXT:  # %bb.4:
; PIC-NEXT:    brne.w 3, %s0, .LBB6_16
; PIC-NEXT:  # %bb.5:
; PIC-NEXT:    or %s0, 4, (0)1
; PIC-NEXT:    adds.w.sx %s0, %s0, (0)1
; PIC-NEXT:    b.l.t (, %s10)
; PIC-NEXT:  .LBB6_10:
; PIC-NEXT:    breq.w 8, %s0, .LBB6_15
; PIC-NEXT:  # %bb.11:
; PIC-NEXT:    brne.w 9, %s0, .LBB6_16
; PIC-NEXT:  # %bb.12:
; PIC-NEXT:    or %s0, 10, (0)1
; PIC-NEXT:    adds.w.sx %s0, %s0, (0)1
; PIC-NEXT:    b.l.t (, %s10)
; PIC-NEXT:  .LBB6_14:
; PIC-NEXT:    adds.w.sx %s0, 3, %s1
; PIC-NEXT:    adds.w.sx %s0, %s0, (0)1
; PIC-NEXT:    b.l.t (, %s10)
; PIC-NEXT:  .LBB6_2:
; PIC-NEXT:    or %s0, 3, (0)1
; PIC-NEXT:    adds.w.sx %s0, %s0, (0)1
; PIC-NEXT:    b.l.t (, %s10)
; PIC-NEXT:  .LBB6_15:
; PIC-NEXT:    or %s0, 11, (0)1
; PIC-NEXT:  .LBB6_16:
; PIC-NEXT:    adds.w.sx %s0, %s0, (0)1
; PIC-NEXT:    b.l.t (, %s10)
; PIC-NEXT:  .LBB6_13:
; PIC-NEXT:    or %s0, 0, (0)1
; PIC-NEXT:    adds.w.sx %s0, %s0, (0)1
; PIC-NEXT:    b.l.t (, %s10)
  switch i32 %0, label %11 [
    i32 1, label %12
    i32 2, label %3
    i32 3, label %4
    i32 4, label %5
    i32 7, label %7
    i32 9, label %9
    i32 8, label %10
  ]

3:                                                ; preds = %2
  br label %12

4:                                                ; preds = %2
  br label %12

5:                                                ; preds = %2
  %6 = add nsw i32 %1, 3
  br label %12

7:                                                ; preds = %2
  %8 = add nsw i32 %1, -2
  br label %12

9:                                                ; preds = %2
  br label %12

10:                                               ; preds = %2
  br label %12

11:                                               ; preds = %2
  br label %12

12:                                               ; preds = %2, %11, %10, %9, %7, %5, %4, %3
  %13 = phi i32 [ %0, %11 ], [ 11, %10 ], [ 10, %9 ], [ %8, %7 ], [ %6, %5 ], [ 4, %4 ], [ 0, %3 ], [ 3, %2 ]
  ret i32 %13
}

; Function Attrs: norecurse nounwind readnone
define signext i32 @br_jt8_m(i32 signext %0, i32 signext %1) {
; CHECK-LABEL: br_jt8_m:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s2, %s0, (32)0
; CHECK-NEXT:    adds.w.sx %s0, -1, %s2
; CHECK-NEXT:    cmpu.w %s3, 8, %s0
; CHECK-NEXT:    brgt.w 0, %s3, .LBB{{[0-9]+}}_9
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    sll %s0, %s0, 3
; CHECK-NEXT:    lea %s3, .LJTI7_0@lo
; CHECK-NEXT:    and %s3, %s3, (32)0
; CHECK-NEXT:    lea.sl %s3, .LJTI7_0@hi(, %s3)
; CHECK-NEXT:    ld %s3, (%s3, %s0)
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    or %s0, 3, (0)1
; CHECK-NEXT:    b.l.t (, %s3)
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
; CHECK-NEXT:  .LBB{{[0-9]+}}_3:
; CHECK-NEXT:    or %s0, 4, (0)1
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
; CHECK-NEXT:  .LBB{{[0-9]+}}_4:
; CHECK-NEXT:    adds.w.sx %s0, 3, %s1
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
; CHECK-NEXT:  .LBB{{[0-9]+}}_9:
; CHECK-NEXT:    or %s0, 0, %s2
; CHECK-NEXT:  .LBB{{[0-9]+}}_10:
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
; CHECK-NEXT:  .LBB{{[0-9]+}}_5:
; CHECK-NEXT:    adds.w.sx %s0, -5, %s1
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
; CHECK-NEXT:  .LBB{{[0-9]+}}_6:
; CHECK-NEXT:    adds.w.sx %s0, -2, %s1
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
; CHECK-NEXT:  .LBB{{[0-9]+}}_8:
; CHECK-NEXT:    or %s0, 11, (0)1
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
; CHECK-NEXT:  .LBB{{[0-9]+}}_7:
; CHECK-NEXT:    or %s0, 10, (0)1
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
;
; PIC-LABEL: br_jt8_m:
; PIC:       # %bb.0:
; PIC-NEXT:    st %s15, 24(, %s11)
; PIC-NEXT:    st %s16, 32(, %s11)
; PIC-NEXT:    and %s2, %s0, (32)0
; PIC-NEXT:    adds.w.sx %s0, -1, %s2
; PIC-NEXT:    cmpu.w %s3, 8, %s0
; PIC-NEXT:    lea %s15, _GLOBAL_OFFSET_TABLE_@pc_lo(-24)
; PIC-NEXT:    and %s15, %s15, (32)0
; PIC-NEXT:    sic %s16
; PIC-NEXT:    lea.sl %s15, _GLOBAL_OFFSET_TABLE_@pc_hi(%s16, %s15)
; PIC-NEXT:    brgt.w 0, %s3, .LBB7_9
; PIC-NEXT:  # %bb.1:
; PIC-NEXT:    and %s1, %s1, (32)0
; PIC-NEXT:    adds.w.zx %s0, %s0, (0)1
; PIC-NEXT:    sll %s0, %s0, 2
; PIC-NEXT:    lea %s3, .LJTI7_0@gotoff_lo
; PIC-NEXT:    and %s3, %s3, (32)0
; PIC-NEXT:    lea.sl %s3, .LJTI7_0@gotoff_hi(%s3, %s15)
; PIC-NEXT:    ldl.sx %s0, (%s3, %s0)
; PIC-NEXT:    lea %s3, br_jt8_m@gotoff_lo
; PIC-NEXT:    and %s3, %s3, (32)0
; PIC-NEXT:    lea.sl %s3, br_jt8_m@gotoff_hi(%s3, %s15)
; PIC-NEXT:    adds.l %s3, %s0, %s3
; PIC-NEXT:    or %s0, 3, (0)1
; PIC-NEXT:    b.l.t (, %s3)
; PIC-NEXT:  .LBB7_2:
; PIC-NEXT:    or %s0, 0, (0)1
; PIC-NEXT:    br.l.t .LBB7_10
; PIC-NEXT:  .LBB7_3:
; PIC-NEXT:    or %s0, 4, (0)1
; PIC-NEXT:    br.l.t .LBB7_10
; PIC-NEXT:  .LBB7_4:
; PIC-NEXT:    adds.w.sx %s0, 3, %s1
; PIC-NEXT:    br.l.t .LBB7_10
; PIC-NEXT:  .LBB7_9:
; PIC-NEXT:    or %s0, 0, %s2
; PIC-NEXT:    br.l.t .LBB7_10
; PIC-NEXT:  .LBB7_5:
; PIC-NEXT:    adds.w.sx %s0, -5, %s1
; PIC-NEXT:    br.l.t .LBB7_10
; PIC-NEXT:  .LBB7_6:
; PIC-NEXT:    adds.w.sx %s0, -2, %s1
; PIC-NEXT:    br.l.t .LBB7_10
; PIC-NEXT:  .LBB7_8:
; PIC-NEXT:    or %s0, 11, (0)1
; PIC-NEXT:    br.l.t .LBB7_10
; PIC-NEXT:  .LBB7_7:
; PIC-NEXT:    or %s0, 10, (0)1
; PIC-NEXT:  .LBB7_10:
; PIC-NEXT:    adds.w.sx %s0, %s0, (0)1
; PIC-NEXT:    ld %s16, 32(, %s11)
; PIC-NEXT:    ld %s15, 24(, %s11)
; PIC-NEXT:    b.l.t (, %s10)
  switch i32 %0, label %13 [
    i32 1, label %14
    i32 2, label %3
    i32 3, label %4
    i32 4, label %5
    i32 6, label %7
    i32 7, label %9
    i32 9, label %11
    i32 8, label %12
  ]

3:                                                ; preds = %2
  br label %14

4:                                                ; preds = %2
  br label %14

5:                                                ; preds = %2
  %6 = add nsw i32 %1, 3
  br label %14

7:                                                ; preds = %2
  %8 = add nsw i32 %1, -5
  br label %14

9:                                                ; preds = %2
  %10 = add nsw i32 %1, -2
  br label %14

11:                                               ; preds = %2
  br label %14

12:                                               ; preds = %2
  br label %14

13:                                               ; preds = %2
  br label %14

14:                                               ; preds = %2, %13, %12, %11, %9, %7, %5, %4, %3
  %15 = phi i32 [ %0, %13 ], [ 11, %12 ], [ 10, %11 ], [ %10, %9 ], [ %8, %7 ], [ %6, %5 ], [ 4, %4 ], [ 0, %3 ], [ 3, %2 ]
  ret i32 %15
}
