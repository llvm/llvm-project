; RUN: opt < %s -loop-unroll -unroll-runtime=true -unroll-runtime-epilog=false -unroll-runtime-multi-exit=true -unroll-count=4  -verify-dom-info -S | FileCheck %s

; REQUIRES: asserts
; The tests below are for verifying dom tree after runtime unrolling
; with multiple exit/exiting blocks.

; We explicitly set the unroll count so that expensiveTripCount computation is allowed.

; mergedexit block has edges from loop exit blocks.
define i64 @test1() {
; CHECK-LABEL: test1(
; CHECK-LABEL: headerexit:
; CHECK-NEXT:    %addphi = phi i64 [ %add.iv, %header ], [ %add.iv.1, %header.1 ], [ %add.iv.2, %header.2 ], [ %add.iv.3, %header.3 ]
; CHECK-NEXT:    br label %mergedexit
; CHECK-LABEL: latchexit:
; CHECK-NEXT:    %shftphi = phi i64 [ %shft, %latch ], [ %shft.1, %latch.1 ], [ %shft.2, %latch.2 ], [ %shft.3, %latch.3 ]
; CHECK-NEXT:    br label %mergedexit
; CHECK-LABEL: mergedexit:
; CHECK-NEXT:    %retval = phi i64 [ %addphi, %headerexit ], [ %shftphi, %latchexit ]
; CHECK-NEXT:    ret i64 %retval
entry:
  br label %preheader

preheader:                                              ; preds = %bb
  %trip = zext i32 undef to i64
  br label %header

header:                                              ; preds = %latch, %preheader
  %iv = phi i64 [ 2, %preheader ], [ %add.iv, %latch ]
  %add.iv = add nuw nsw i64 %iv, 2
  %cmp1 = icmp ult i64 %add.iv, %trip
  br i1 %cmp1, label %latch, label %headerexit

latch:                                             ; preds = %header
  %shft = ashr i64 %add.iv, 1
  %cmp2 = icmp ult i64 %shft, %trip
  br i1 %cmp2, label %header, label %latchexit

headerexit:                                              ; preds = %header
  %addphi = phi i64 [ %add.iv, %header ]
  br label %mergedexit

latchexit:                                              ; preds = %latch
 %shftphi = phi i64 [ %shft, %latch ]
  br label %mergedexit

mergedexit:                                              ; preds = %latchexit, %headerexit
  %retval = phi i64 [ %addphi, %headerexit ], [ %shftphi, %latchexit ]
  ret i64 %retval
}

; mergedexit has edges from loop exit blocks and a block outside the loop.
define  void @test2(i1 %cond, i32 %n) {
; CHECK-LABEL: header.1:
; CHECK-NEXT:    %add.iv.1 = add nuw nsw i64 %add.iv, 2
; CHECK:         br i1 %cmp1.1, label %latch.1, label %headerexit
; CHECK-LABEL: latch.3:
; CHECK:         %cmp2.3 = icmp ult i64 %shft.3, %trip
; CHECK-NEXT:    br i1 %cmp2.3, label %header, label %latchexit, !llvm.loop
entry:
  br i1 %cond, label %preheader, label %mergedexit

preheader:                                              ; preds = %entry
  %trip = zext i32 %n to i64
  br label %header

header:                                              ; preds = %latch, %preheader
  %iv = phi i64 [ 2, %preheader ], [ %add.iv, %latch ]
  %add.iv = add nuw nsw i64 %iv, 2
  %cmp1 = icmp ult i64 %add.iv, %trip
  br i1 %cmp1, label %latch, label %headerexit

latch:                                             ; preds = %header
  %shft = ashr i64 %add.iv, 1
  %cmp2 = icmp ult i64 %shft, %trip
  br i1 %cmp2, label %header, label %latchexit

headerexit:                                              ; preds = %header
  br label %mergedexit

latchexit:                                              ; preds = %latch
  br label %mergedexit

mergedexit:                                              ; preds = %latchexit, %headerexit, %entry
  ret void
}


; exitsucc is from loop exit block only.
define i64 @test3(i32 %n) {
; CHECK-LABEL: test3(
; CHECK-LABEL:  headerexit:
; CHECK-NEXT:     br label %exitsucc
; CHECK-LABEL:  latchexit:
; CHECK-NEXT:     %shftphi = phi i64 [ %shft, %latch ], [ %shft.1, %latch.1 ], [ %shft.2, %latch.2 ], [ %shft.3, %latch.3 ]
; CHECK-NEXT:     ret i64 %shftphi
; CHECK-LABEL:  exitsucc:
; CHECK-NEXT:     ret i64 96
entry:
  br label %preheader

preheader:                                              ; preds = %bb
  %trip = zext i32 %n to i64
  br label %header

header:                                              ; preds = %latch, %preheader
  %iv = phi i64 [ 2, %preheader ], [ %add.iv, %latch ]
  %add.iv = add nuw nsw i64 %iv, 2
  %cmp1 = icmp ult i64 %add.iv, %trip
  br i1 %cmp1, label %latch, label %headerexit

latch:                                             ; preds = %header
  %shft = ashr i64 %add.iv, 1
  %cmp2 = icmp ult i64 %shft, %trip
  br i1 %cmp2, label %header, label %latchexit

headerexit:                                              ; preds = %header
  br label %exitsucc

latchexit:                                              ; preds = %latch
  %shftphi = phi i64 [ %shft, %latch ]
  ret i64 %shftphi

exitsucc:                                              ; preds = %headerexit
  ret i64 96
}
