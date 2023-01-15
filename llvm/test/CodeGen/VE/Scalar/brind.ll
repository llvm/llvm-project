; RUN: llc < %s -mtriple=ve | FileCheck %s

; Function Attrs: norecurse nounwind readnone
define signext i32 @brind(i32 signext %0) {
; CHECK-LABEL: brind:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmpu.w %s1, %s0, (63)0
; CHECK-NEXT:    lea %s2, .Ltmp0@lo
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lea.sl %s2, .Ltmp0@hi(, %s2)
; CHECK-NEXT:    lea %s3, .Ltmp1@lo
; CHECK-NEXT:    and %s3, %s3, (32)0
; CHECK-NEXT:    lea.sl %s3, .Ltmp1@hi(, %s3)
; CHECK-NEXT:    cmov.w.eq %s2, %s3, %s1
; CHECK-NEXT:    lea %s1, .Ltmp2@lo
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s1, .Ltmp2@hi(, %s1)
; CHECK-NEXT:    cmov.w.eq %s1, %s2, %s0
; CHECK-NEXT:    b.l.t (, %s1)
; CHECK-NEXT:  .Ltmp0: # Block address taken
; CHECK-NEXT:  .LBB{{[0-9]+}}_3:
; CHECK-NEXT:    or %s0, -1, (0)1
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
; CHECK-NEXT:  .Ltmp2: # Block address taken
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s0, 2, (0)1
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
; CHECK-NEXT:  .Ltmp1: # Block address taken
; CHECK-NEXT:  .LBB{{[0-9]+}}_1:
; CHECK-NEXT:    or %s0, 1, (0)1
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = icmp eq i32 %0, 1
  %3 = select i1 %2, ptr blockaddress(@brind, %6), ptr blockaddress(@brind, %8)
  %4 = icmp eq i32 %0, 0
  %5 = select i1 %4, ptr %3, ptr blockaddress(@brind, %7)
  indirectbr ptr %5, [label %8, label %6, label %7]

6:                                                ; preds = %1
  br label %8

7:                                                ; preds = %1
  br label %8

8:                                                ; preds = %1, %7, %6
  %9 = phi i32 [ 2, %7 ], [ 1, %6 ], [ -1, %1 ]
  ret i32 %9
}
