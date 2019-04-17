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

; exit block (%default) has an exiting block and another exit block as predecessors.
define void @test4(i16 %c3) {
; CHECK-LABEL: test4

; CHECK-LABEL: exiting.prol:
; CHECK-NEXT:   switch i16 %c3, label %default.loopexit.loopexit1 [

; CHECK-LABEL: exiting:
; CHECK-NEXT:   switch i16 %c3, label %default.loopexit.loopexit [

; CHECK-LABEL: default.loopexit.loopexit:
; CHECK-NEXT:   br label %default.loopexit

; CHECK-LABEL: default.loopexit.loopexit1:
; CHECK-NEXT:   br label %default.loopexit

; CHECK-LABEL: default.loopexit:
; CHECK-NEXT:   br label %default
preheader:
  %c1 = zext i32 undef to i64
  br label %header

header:                                       ; preds = %latch, %preheader
  %indvars.iv = phi i64 [ 0, %preheader ], [ %indvars.iv.next, %latch ]
  br label %exiting

exiting:                                           ; preds = %header
  switch i16 %c3, label %default [
    i16 45, label %otherexit
    i16 95, label %latch
  ]

latch:                                          ; preds = %exiting
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %c2 = icmp ult i64 %indvars.iv.next, %c1
  br i1 %c2, label %header, label %latchexit

latchexit:                                          ; preds = %latch
  ret void

default:                                          ; preds = %otherexit, %exiting
  ret void

otherexit:                                           ; preds = %exiting
  br label %default
}

; exit block (%exitB) has an exiting block and another exit block as predecessors.
; exiting block comes from inner loop.
define void @test5() {
; CHECK-LABEL: test5
; CHECK-LABEL: bb1:
; CHECK-NEXT:   br i1 false, label %outerH.prol.preheader, label %outerH.prol.loopexit

; CHECK-LABEL: outerH.prol.preheader:
; CHECK-NEXT:   br label %outerH.prol

; CHECK-LABEL: outerH.prol:
; CHECK-NEXT:   %tmp4.prol = phi i32 [ %tmp6.prol, %outerLatch.prol ], [ undef, %outerH.prol.preheader ]
; CHECK-NEXT:   %prol.iter = phi i32 [ 0, %outerH.prol.preheader ], [ %prol.iter.sub, %outerLatch.prol ]
; CHECK-NEXT:   br label %innerH.prol
bb:
  %tmp = icmp sgt i32 undef, 79
  br i1 %tmp, label %outerLatchExit, label %bb1

bb1:                                              ; preds = %bb
  br label %outerH

outerH:                                              ; preds = %outerLatch, %bb1
  %tmp4 = phi i32 [ %tmp6, %outerLatch ], [ undef, %bb1 ]
  br label %innerH

innerH:                                              ; preds = %innerLatch, %outerH
  br i1 undef, label %innerexiting, label %otherexitB

innerexiting:                                             ; preds = %innerH
  br i1 undef, label %innerLatch, label %exitB

innerLatch:                                             ; preds = %innerexiting
  %tmp13 = fcmp olt double undef, 2.000000e+00
  br i1 %tmp13, label %innerH, label %outerLatch

outerLatch:                                              ; preds = %innerLatch
  %tmp6 = add i32 %tmp4, 1
  %tmp7 = icmp sgt i32 %tmp6, 79
  br i1 %tmp7, label %outerLatchExit, label %outerH

outerLatchExit:                                              ; preds = %outerLatch, %bb
  ret void

exitB:                                             ; preds = %innerexiting, %otherexitB
  ret void

otherexitB:                                              ; preds = %innerH
  br label %exitB

}

; Blocks reachable from exits (not_zero44) have the IDom as the block within the loop (Header).
; Update the IDom to the preheader.
define void @test6() {
; CHECK-LABEL: test6
; CHECK-LABEL: header.prol.preheader:
; CHECK-NEXT:    br label %header.prol

; CHECK-LABEL: header.prol:
; CHECK-NEXT:    %indvars.iv.prol = phi i64 [ undef, %header.prol.preheader ], [ %indvars.iv.next.prol, %latch.prol ]
; CHECK-NEXT:    %prol.iter = phi i64 [ 1, %header.prol.preheader ], [ %prol.iter.sub, %latch.prol ]
; CHECK-NEXT:    br i1 false, label %latch.prol, label %otherexit.loopexit1

; CHECK-LABEL: header.prol.loopexit.unr-lcssa:
; CHECK-NEXT:    %indvars.iv.unr.ph = phi i64 [ %indvars.iv.next.prol, %latch.prol ]
; CHECK-NEXT:    br label %header.prol.loopexit

; CHECK-LABEL: header.prol.loopexit:
; CHECK-NEXT:    %indvars.iv.unr = phi i64 [ undef, %entry ], [ %indvars.iv.unr.ph, %header.prol.loopexit.unr-lcssa ]
; CHECK-NEXT:    br i1 true, label %latchexit, label %entry.new

; CHECK-LABEL: entry.new:
; CHECK-NEXT:    br label %header
entry:
  br label %header

header:                                          ; preds = %latch, %entry
  %indvars.iv = phi i64 [ undef, %entry ], [ %indvars.iv.next, %latch ]
  br i1 undef, label %latch, label %otherexit

latch:                                         ; preds = %header
  %indvars.iv.next = add nsw i64 %indvars.iv, 2
  %0 = icmp slt i64 %indvars.iv.next, 616
  br i1 %0, label %header, label %latchexit

latchexit:                                          ; preds = %latch
  br label %latchexitsucc

otherexit:                                 ; preds = %header
  br label %otherexitsucc

otherexitsucc:                                          ; preds = %otherexit
  br label %not_zero44

not_zero44:                                       ; preds = %latchexitsucc, %otherexitsucc
  unreachable

latchexitsucc:                                      ; preds = %latchexit
  br label %not_zero44
}

