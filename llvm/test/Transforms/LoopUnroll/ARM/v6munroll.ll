; RUN: opt -mtriple=arm-none-none-eabi -mcpu=cortex-m23 -passes=loop-unroll -unroll-runtime-multi-exit -S %s -o - | FileCheck %s

; This loop has too many live outs, and should not be unrolled under v6m.
; CHECK-LABEL: multiple_liveouts
; CHECK: for.body
; CHECK: br i1 %cmp.not, label %for.cond.cleanup.loopexit, label %for.body
define void @multiple_liveouts(i32* %x, i32* %y, i32* %d, i32 %n) {
entry:
  %0 = load i32, i32* %d, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %d, i32 1
  %1 = load i32, i32* %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds i32, i32* %d, i32 2
  %2 = load i32, i32* %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds i32, i32* %d, i32 3
  %3 = load i32, i32* %arrayidx3, align 4
  %cmp.not58 = icmp eq i32 %n, 0
  br i1 %cmp.not58, label %for.cond.cleanup, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %x.addr.065 = phi i32* [ %incdec.ptr, %for.body ], [ %x, %for.body.preheader ]
  %y.addr.064 = phi i32* [ %incdec.ptr25, %for.body ], [ %y, %for.body.preheader ]
  %res00.063 = phi i32 [ %add, %for.body ], [ %0, %for.body.preheader ]
  %rhs_cols_idx.062 = phi i32 [ %dec, %for.body ], [ %n, %for.body.preheader ]
  %res11.061 = phi i32 [ %add24, %for.body ], [ %3, %for.body.preheader ]
  %res10.060 = phi i32 [ %add20, %for.body ], [ %2, %for.body.preheader ]
  %res01.059 = phi i32 [ %add14, %for.body ], [ %1, %for.body.preheader ]
  %4 = load i32, i32* %x.addr.065, align 4
  %arrayidx5 = getelementptr inbounds i32, i32* %x.addr.065, i32 %n
  %5 = load i32, i32* %arrayidx5, align 4
  %6 = load i32, i32* %y.addr.064, align 4
  %conv9 = and i32 %6, 255
  %conv10 = and i32 %4, 255
  %mul = mul nuw nsw i32 %conv9, %conv10
  %add = add nsw i32 %mul, %res00.063
  %conv12 = and i32 %5, 255
  %mul13 = mul nuw nsw i32 %conv9, %conv12
  %add14 = add nsw i32 %mul13, %res01.059
  %arrayidx15 = getelementptr inbounds i32, i32* %y.addr.064, i32 %n
  %7 = load i32, i32* %arrayidx15, align 4
  %conv17 = and i32 %7, 255
  %mul19 = mul nuw nsw i32 %conv17, %conv10
  %add20 = add nsw i32 %mul19, %res10.060
  %mul23 = mul nuw nsw i32 %conv17, %conv12
  %add24 = add nsw i32 %mul23, %res11.061
  %incdec.ptr = getelementptr inbounds i32, i32* %x.addr.065, i32 1
  %incdec.ptr25 = getelementptr inbounds i32, i32* %y.addr.064, i32 1
  %dec = add nsw i32 %rhs_cols_idx.062, -1
  %cmp.not = icmp eq i32 %dec, 0
  br i1 %cmp.not, label %for.cond.cleanup.loopexit, label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.body
  %add.lcssa = phi i32 [ %add, %for.body ]
  %add14.lcssa = phi i32 [ %add14, %for.body ]
  %add20.lcssa = phi i32 [ %add20, %for.body ]
  %add24.lcssa = phi i32 [ %add24, %for.body ]
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  %res01.0 = phi i32 [ %1, %entry ], [ %add14.lcssa, %for.cond.cleanup.loopexit ]
  %res10.0 = phi i32 [ %2, %entry ], [ %add20.lcssa, %for.cond.cleanup.loopexit ]
  %res11.0 = phi i32 [ %3, %entry ], [ %add24.lcssa, %for.cond.cleanup.loopexit ]
  %res00.0 = phi i32 [ %0, %entry ], [ %add.lcssa, %for.cond.cleanup.loopexit ]
  store i32 %res00.0, i32* %d, align 4
  store i32 %res01.0, i32* %arrayidx1, align 4
  store i32 %res10.0, i32* %arrayidx2, align 4
  store i32 %res11.0, i32* %arrayidx3, align 4
  ret void
}

; This loop has both multiple exit blocks and multiple liveouts
; CHECK-LABEL: multiple_liveouts_doubleexit
; CHECK: for.body
; CHECK: br i1 %cmp.not, label %cleanup22.loopexit2, label %for.body
define void @multiple_liveouts_doubleexit(i32 %n, i32* %x, i32* %y, i32* %z) {
entry:
  %cmp.not55 = icmp eq i32 %n, 0
  br i1 %cmp.not55, label %cleanup22, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.inc
  %x.addr.062 = phi i32* [ %incdec.ptr, %for.inc ], [ %x, %for.body.preheader ]
  %y.addr.061 = phi i32* [ %incdec.ptr19, %for.inc ], [ %y, %for.body.preheader ]
  %rhs_cols_idx.060 = phi i32 [ %dec, %for.inc ], [ %n, %for.body.preheader ]
  %res11.059 = phi i32 [ %add18, %for.inc ], [ 0, %for.body.preheader ]
  %res10.058 = phi i32 [ %add16, %for.inc ], [ 0, %for.body.preheader ]
  %res01.057 = phi i32 [ %add8, %for.inc ], [ 0, %for.body.preheader ]
  %res00.056 = phi i32 [ %add, %for.inc ], [ 0, %for.body.preheader ]
  %0 = load i32, i32* %x.addr.062, align 4
  %1 = load i32, i32* %y.addr.061, align 4
  %conv5 = and i32 %1, 255
  %conv6 = and i32 %0, 255
  %mul = mul nuw nsw i32 %conv5, %conv6
  %add = add nuw nsw i32 %mul, %res00.056
  %add8 = add nuw nsw i32 %conv5, %res01.057
  %cmp9 = icmp ugt i32 %add8, 100
  br i1 %cmp9, label %cleanup22.loopexit, label %for.inc

for.inc:                                          ; preds = %for.body
  %arrayidx11 = getelementptr inbounds i32, i32* %y.addr.061, i32 %n
  %2 = load i32, i32* %arrayidx11, align 4
  %conv13 = and i32 %2, 255
  %mul15 = mul nuw nsw i32 %conv13, %conv6
  %add16 = add nuw nsw i32 %mul15, %res10.058
  %add18 = add nuw nsw i32 %conv13, %res11.059
  %incdec.ptr = getelementptr inbounds i32, i32* %x.addr.062, i32 1
  %incdec.ptr19 = getelementptr inbounds i32, i32* %y.addr.061, i32 1
  %dec = add nsw i32 %rhs_cols_idx.060, -1
  %cmp.not = icmp eq i32 %dec, 0
  br i1 %cmp.not, label %cleanup22.loopexit2, label %for.body

cleanup22.loopexit:                               ; preds = %for.body
  %add.lcssa = phi i32 [ %add, %for.body ]
  %add8.lcssa = phi i32 [ %add8, %for.body ]
  %res10.0.lcssa.ph = phi i32 [ %res10.058, %for.body ]
  %res11.0.lcssa.ph = phi i32 [ %res11.059, %for.body ]
  br label %cleanup22

cleanup22.loopexit2:                               ; preds = %for.inc
  %add.lcssa2 = phi i32 [ %add, %for.inc ]
  %add8.lcssa2 = phi i32 [ %add8, %for.inc ]
  %res10.0.lcssa.ph2 = phi i32 [ %add16, %for.inc ]
  %res11.0.lcssa.ph2 = phi i32 [ %add18, %for.inc ]
  br label %cleanup22

cleanup22:                                        ; preds = %cleanup22.loopexit, %entry
  %res10.0.lcssa = phi i32 [ 0, %entry ], [ %res10.0.lcssa.ph, %cleanup22.loopexit ], [ %res10.0.lcssa.ph2, %cleanup22.loopexit2 ]
  %res11.0.lcssa = phi i32 [ 0, %entry ], [ %res11.0.lcssa.ph, %cleanup22.loopexit ], [ %res11.0.lcssa.ph2, %cleanup22.loopexit2 ]
  %res00.1 = phi i32 [ 0, %entry ], [ %add.lcssa, %cleanup22.loopexit ], [ %add.lcssa2, %cleanup22.loopexit2 ]
  %res01.1 = phi i32 [ 0, %entry ], [ %add8.lcssa, %cleanup22.loopexit ], [ %add8.lcssa2, %cleanup22.loopexit2 ]
  store i32 %res00.1, i32* %z, align 4
  %arrayidx24 = getelementptr inbounds i32, i32* %z, i32 1
  store i32 %res01.1, i32* %arrayidx24, align 4
  %arrayidx25 = getelementptr inbounds i32, i32* %z, i32 2
  store i32 %res10.0.lcssa, i32* %arrayidx25, align 4
  %arrayidx26 = getelementptr inbounds i32, i32* %z, i32 3
  store i32 %res11.0.lcssa, i32* %arrayidx26, align 4
  ret void
}
