; RUN: opt < %s -scalar-evolution-huge-expr-threshold=1000000 -loop-reduce -S | FileCheck %s

target datalayout = "e-m:e-i32:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Show that the b^2 is expanded correctly.
define i32 @test_01(i32 %a) {
; CHECK-LABEL: @test_01
; CHECK:       entry:
; CHECK-NEXT:  br label %loop
; CHECK:       loop:
; CHECK-NEXT:  [[IV:[^ ]+]] = phi i32 [ [[IV_INC:[^ ]+]], %loop ], [ 0, %entry ]
; CHECK-NEXT:  [[IV_INC]] = add nsw i32 [[IV]], -1
; CHECK-NEXT:  [[EXITCOND:[^ ]+]] = icmp eq i32 [[IV_INC]], -80
; CHECK-NEXT:  br i1 [[EXITCOND]], label %exit, label %loop
; CHECK:       exit:
; CHECK-NEXT:  [[B:[^ ]+]] = add i32 %a, 1
; CHECK-NEXT:  [[B2:[^ ]+]] = mul i32 [[B]], [[B]]
; CHECK-NEXT:  [[R1:[^ ]+]] = add i32 [[B2]], -1
; CHECK-NEXT:  [[R2:[^ ]+]] = sub i32 [[R1]], [[IV_INC]]
; CHECK-NEXT:  ret i32 [[R2]]

entry:
  br label %loop

loop:                                           ; preds = %loop, %entry
  %indvars.iv = phi i32 [ 0, %entry ], [ %indvars.iv.next, %loop ]
  %b = add i32 %a, 1
  %b.pow.2 = mul i32 %b, %b
  %result = add i32 %b.pow.2, %indvars.iv
  %indvars.iv.next = add nuw nsw i32 %indvars.iv, 1
  %exitcond = icmp eq i32 %indvars.iv.next, 80
  br i1 %exitcond, label %exit, label %loop

exit:                                             ; preds = %loop
  ret i32 %result
}

; Show that b^8 is expanded correctly.
define i32 @test_02(i32 %a) {
; CHECK-LABEL: @test_02
; CHECK:       entry:
; CHECK-NEXT:  br label %loop
; CHECK:       loop:
; CHECK-NEXT:  [[IV:[^ ]+]] = phi i32 [ [[IV_INC:[^ ]+]], %loop ], [ 0, %entry ]
; CHECK-NEXT:  [[IV_INC]] = add nsw i32 [[IV]], -1
; CHECK-NEXT:  [[EXITCOND:[^ ]+]] = icmp eq i32 [[IV_INC]], -80
; CHECK-NEXT:  br i1 [[EXITCOND]], label %exit, label %loop
; CHECK:       exit:
; CHECK-NEXT:  [[B:[^ ]+]] = add i32 %a, 1
; CHECK-NEXT:  [[B2:[^ ]+]] = mul i32 [[B]], [[B]]
; CHECK-NEXT:  [[B4:[^ ]+]] = mul i32 [[B2]], [[B2]]
; CHECK-NEXT:  [[B8:[^ ]+]] = mul i32 [[B4]], [[B4]]
; CHECK-NEXT:  [[R1:[^ ]+]] = add i32 [[B8]], -1
; CHECK-NEXT:  [[R2:[^ ]+]] = sub i32 [[R1]], [[IV_INC]]
; CHECK-NEXT:  ret i32 [[R2]]
entry:
  br label %loop

loop:                                           ; preds = %loop, %entry
  %indvars.iv = phi i32 [ 0, %entry ], [ %indvars.iv.next, %loop ]
  %b = add i32 %a, 1
  %b.pow.2 = mul i32 %b, %b
  %b.pow.4 = mul i32 %b.pow.2, %b.pow.2
  %b.pow.8 = mul i32 %b.pow.4, %b.pow.4
  %result = add i32 %b.pow.8, %indvars.iv
  %indvars.iv.next = add nuw nsw i32 %indvars.iv, 1
  %exitcond = icmp eq i32 %indvars.iv.next, 80
  br i1 %exitcond, label %exit, label %loop

exit:                                             ; preds = %loop
  ret i32 %result
}

; Show that b^27 (27 = 1 + 2 + 8 + 16) is expanded correctly.
define i32 @test_03(i32 %a) {
; CHECK-LABEL: @test_03
; CHECK:       entry:
; CHECK-NEXT:  br label %loop
; CHECK:       loop:
; CHECK-NEXT:  [[IV:[^ ]+]] = phi i32 [ [[IV_INC:[^ ]+]], %loop ], [ 0, %entry ]
; CHECK-NEXT:  [[IV_INC]] = add nsw i32 [[IV]], -1
; CHECK-NEXT:  [[EXITCOND:[^ ]+]] = icmp eq i32 [[IV_INC]], -80
; CHECK-NEXT:  br i1 [[EXITCOND]], label %exit, label %loop
; CHECK:       exit:
; CHECK-NEXT:  [[B:[^ ]+]] = add i32 %a, 1
; CHECK-NEXT:  [[B2:[^ ]+]] = mul i32 [[B]], [[B]]
; CHECK-NEXT:  [[B3:[^ ]+]] = mul i32 [[B]], [[B2]]
; CHECK-NEXT:  [[B4:[^ ]+]] = mul i32 [[B2]], [[B2]]
; CHECK-NEXT:  [[B8:[^ ]+]] = mul i32 [[B4]], [[B4]]
; CHECK-NEXT:  [[B11:[^ ]+]] = mul i32 [[B3]], [[B8]]
; CHECK-NEXT:  [[B16:[^ ]+]] = mul i32 [[B8]], [[B8]]
; CHECK-NEXT:  [[B27:[^ ]+]] = mul i32 [[B11]], [[B16]]
; CHECK-NEXT:  [[R1:[^ ]+]] = add i32 [[B27]], -1
; CHECK-NEXT:  [[R2:[^ ]+]] = sub i32 [[R1]], [[IV_INC]]
; CHECK-NEXT:  ret i32 [[R2]]
entry:
  br label %loop

loop:                                           ; preds = %loop, %entry
  %indvars.iv = phi i32 [ 0, %entry ], [ %indvars.iv.next, %loop ]
  %b = add i32 %a, 1
  %b.pow.2 = mul i32 %b, %b
  %b.pow.4 = mul i32 %b.pow.2, %b.pow.2
  %b.pow.8 = mul i32 %b.pow.4, %b.pow.4
  %b.pow.16 = mul i32 %b.pow.8, %b.pow.8
  %b.pow.24 = mul i32 %b.pow.16, %b.pow.8
  %b.pow.25 = mul i32 %b.pow.24, %b
  %b.pow.26 = mul i32 %b.pow.25, %b
  %b.pow.27 = mul i32 %b.pow.26, %b
  %result = add i32 %b.pow.27, %indvars.iv
  %indvars.iv.next = add nuw nsw i32 %indvars.iv, 1
  %exitcond = icmp eq i32 %indvars.iv.next, 80
  br i1 %exitcond, label %exit, label %loop

exit:                                             ; preds = %loop
  ret i32 %result
}

; Show how linear calculation of b^16 is turned into logarithmic.
define i32 @test_04(i32 %a) {
; CHECK-LABEL: @test_04
; CHECK:       entry:
; CHECK-NEXT:  br label %loop
; CHECK:       loop:
; CHECK-NEXT:  [[IV:[^ ]+]] = phi i32 [ [[IV_INC:[^ ]+]], %loop ], [ 0, %entry ]
; CHECK-NEXT:  [[IV_INC]] = add nsw i32 [[IV]], -1
; CHECK-NEXT:  [[EXITCOND:[^ ]+]] = icmp eq i32 [[IV_INC]], -80
; CHECK-NEXT:  br i1 [[EXITCOND]], label %exit, label %loop
; CHECK:       exit:
; CHECK-NEXT:  [[B:[^ ]+]] = add i32 %a, 1
; CHECK-NEXT:  [[B2:[^ ]+]] = mul i32 [[B]], [[B]]
; CHECK-NEXT:  [[B4:[^ ]+]] = mul i32 [[B2]], [[B2]]
; CHECK-NEXT:  [[B8:[^ ]+]] = mul i32 [[B4]], [[B4]]
; CHECK-NEXT:  [[B16:[^ ]+]] = mul i32 [[B8]], [[B8]]
; CHECK-NEXT:  [[R1:[^ ]+]] = add i32 [[B16]], -1
; CHECK-NEXT:  [[R2:[^ ]+]] = sub i32 [[R1]], [[IV_INC]]
; CHECK-NEXT:  ret i32 [[R2]]
entry:
  br label %loop

loop:                                           ; preds = %loop, %entry
  %indvars.iv = phi i32 [ 0, %entry ], [ %indvars.iv.next, %loop ]
  %b = add i32 %a, 1
  %b.pow.2 = mul i32 %b, %b
  %b.pow.3 = mul i32 %b.pow.2, %b
  %b.pow.4 = mul i32 %b.pow.3, %b
  %b.pow.5 = mul i32 %b.pow.4, %b
  %b.pow.6 = mul i32 %b.pow.5, %b
  %b.pow.7 = mul i32 %b.pow.6, %b
  %b.pow.8 = mul i32 %b.pow.7, %b
  %b.pow.9 = mul i32 %b.pow.8, %b
  %b.pow.10 = mul i32 %b.pow.9, %b
  %b.pow.11 = mul i32 %b.pow.10, %b
  %b.pow.12 = mul i32 %b.pow.11, %b
  %b.pow.13 = mul i32 %b.pow.12, %b
  %b.pow.14 = mul i32 %b.pow.13, %b
  %b.pow.15 = mul i32 %b.pow.14, %b
  %b.pow.16 = mul i32 %b.pow.15, %b
  %result = add i32 %b.pow.16, %indvars.iv
  %indvars.iv.next = add nuw nsw i32 %indvars.iv, 1
  %exitcond = icmp eq i32 %indvars.iv.next, 80
  br i1 %exitcond, label %exit, label %loop

exit:                                             ; preds = %loop
  ret i32 %result
}

; The output here is reasonably big, we just check that the amount of expanded
; instructions is sane.
define i32 @test_05(i32 %a) {
; CHECK-LABEL: @test_05
; CHECK:       entry:
; CHECK-NEXT:  br label %loop
; CHECK:       loop:
; CHECK-NEXT:  [[IV:[^ ]+]] = phi i32 [ [[IV_INC:[^ ]+]], %loop ], [ 0, %entry ]
; CHECK-NEXT:  [[IV_INC]] = add nsw i32 [[IV]], -1
; CHECK-NEXT:  [[EXITCOND:[^ ]+]] = icmp eq i32 [[IV_INC]], -80
; CHECK-NEXT:  br i1 [[EXITCOND]], label %exit, label %loop
; CHECK:       exit:
; CHECK:       %100
; CHECK-NOT:   %150

entry:
  br label %loop

loop:                                           ; preds = %loop, %entry
  %indvars.iv = phi i32 [ 0, %entry ], [ %indvars.iv.next, %loop ]
  %tmp3 = add i32 %a, 1
  %tmp4 = mul i32 %tmp3, %tmp3
  %tmp5 = mul i32 %tmp4, %tmp4
  %tmp6 = mul i32 %tmp5, %tmp5
  %tmp7 = mul i32 %tmp6, %tmp6
  %tmp8 = mul i32 %tmp7, %tmp7
  %tmp9 = mul i32 %tmp8, %tmp8
  %tmp10 = mul i32 %tmp9, %tmp9
  %tmp11 = mul i32 %tmp10, %tmp10
  %tmp12 = mul i32 %tmp11, %tmp11
  %tmp13 = mul i32 %tmp12, %tmp12
  %tmp14 = mul i32 %tmp13, %tmp13
  %tmp15 = mul i32 %tmp14, %tmp14
  %tmp16 = mul i32 %tmp15, %tmp15
  %tmp17 = mul i32 %tmp16, %tmp16
  %tmp18 = mul i32 %tmp17, %tmp17
  %tmp19 = mul i32 %tmp18, %tmp18
  %tmp20 = mul i32 %tmp19, %tmp19
  %tmp22 = add i32 %tmp20, %indvars.iv
  %indvars.iv.next = add nuw nsw i32 %indvars.iv, 1
  %exitcond = icmp eq i32 %indvars.iv.next, 80
  br i1 %exitcond, label %exit, label %loop

exit:                                             ; preds = %loop
  ret i32 %tmp22
}

; Show that the transformation works even if the calculation involves different
; values inside.
define i32 @test_06(i32 %a, i32 %c) {
; CHECK-LABEL: @test_06
; CHECK:       entry:
; CHECK-NEXT:  br label %loop
; CHECK:       loop:
; CHECK-NEXT:  [[IV:[^ ]+]] = phi i32 [ [[IV_INC:[^ ]+]], %loop ], [ 0, %entry ]
; CHECK-NEXT:  [[IV_INC]] = add nsw i32 [[IV]], -1
; CHECK-NEXT:  [[EXITCOND:[^ ]+]] = icmp eq i32 [[IV_INC]], -80
; CHECK-NEXT:  br i1 [[EXITCOND]], label %exit, label %loop
; CHECK:       exit:
; CHECK:       [[B:[^ ]+]] = add i32 %a, 1
; CHECK-NEXT:  [[B2:[^ ]+]] = mul i32 [[B]], [[B]]
; CHECK-NEXT:  [[B4:[^ ]+]] = mul i32 [[B2]], [[B2]]
; CHECK-NEXT:  [[B8:[^ ]+]] = mul i32 [[B4]], [[B4]]
; CHECK-NEXT:  [[B16:[^ ]+]] = mul i32 [[B8]], [[B8]]
entry:
  br label %loop

loop:                                           ; preds = %loop, %entry
  %indvars.iv = phi i32 [ 0, %entry ], [ %indvars.iv.next, %loop ]
  %b = add i32 %a, 1
  %b.pow.2.tmp = mul i32 %b, %b
  %b.pow.2 = mul i32 %b.pow.2.tmp, %c
  %b.pow.3 = mul i32 %b.pow.2, %b
  %b.pow.4 = mul i32 %b.pow.3, %b
  %b.pow.5 = mul i32 %b.pow.4, %b
  %b.pow.6.tmp = mul i32 %b.pow.5, %b
  %b.pow.6 = mul i32 %b.pow.6.tmp, %c
  %b.pow.7 = mul i32 %b.pow.6, %b
  %b.pow.8 = mul i32 %b.pow.7, %b
  %b.pow.9 = mul i32 %b.pow.8, %b
  %b.pow.10 = mul i32 %b.pow.9, %b
  %b.pow.11 = mul i32 %b.pow.10, %b
  %b.pow.12.tmp = mul i32 %b.pow.11, %b
  %b.pow.12 = mul i32 %c, %b.pow.12.tmp
  %b.pow.13 = mul i32 %b.pow.12, %b
  %b.pow.14 = mul i32 %b.pow.13, %b
  %b.pow.15 = mul i32 %b.pow.14, %b
  %b.pow.16 = mul i32 %b.pow.15, %b
  %result = add i32 %b.pow.16, %indvars.iv
  %indvars.iv.next = add nuw nsw i32 %indvars.iv, 1
  %exitcond = icmp eq i32 %indvars.iv.next, 80
  br i1 %exitcond, label %exit, label %loop

exit:                                             ; preds = %loop
  ret i32 %result
}
