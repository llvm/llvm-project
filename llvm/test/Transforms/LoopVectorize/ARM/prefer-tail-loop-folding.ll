; RUN: opt -mtriple=thumbv8.1m.main-arm-eabihf < %s -loop-vectorize -S | \
; RUN:  FileCheck %s -check-prefixes=CHECK,NO-FOLDING

; RUN: opt -mtriple=thumbv8.1m.main-arm-eabihf -mattr=-mve < %s -loop-vectorize -enable-arm-maskedldst=true -S | \
; RUN:  FileCheck %s -check-prefixes=CHECK,NO-FOLDING

; RUN: opt -mtriple=thumbv8.1m.main-arm-eabihf -mattr=+mve < %s -loop-vectorize -enable-arm-maskedldst=false -S | \
; RUN:  FileCheck %s -check-prefixes=CHECK,NO-FOLDING

; Disabling the low-overhead branch extension will make
; 'isHardwareLoopProfitable' return false, so that we test avoiding folding for
; these cases.
; RUN: opt -mtriple=thumbv8.1m.main-arm-eabihf -mattr=+mve,-lob < %s -loop-vectorize -enable-arm-maskedldst=true -S | \
; RUN:  FileCheck %s -check-prefixes=CHECK,NO-FOLDING

; RUN: opt -mtriple=thumbv8.1m.main-arm-eabihf -mattr=+mve < %s -loop-vectorize -enable-arm-maskedldst=true -S | \
; RUN:  FileCheck %s -check-prefixes=CHECK,PREFER-FOLDING

define dso_local void @tail_folding(i32* noalias nocapture %A, i32* noalias nocapture readonly %B, i32* noalias nocapture readonly %C) {
; CHECK-LABEL: tail_folding(
;
; NO-FOLDING-NOT:  call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(
; NO-FOLDING-NOT:  call void @llvm.masked.store.v4i32.p0v4i32(
;
; TODO: this needs implementation of TTI::preferPredicateOverEpilogue,
; then this will be tail-folded too:
;
; PREFER-FOLDING-NOT:  call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(
; PREFER-FOLDING-NOT:  call void @llvm.masked.store.v4i32.p0v4i32(
;
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %B, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, i32* %C, i64 %indvars.iv
  %1 = load i32, i32* %arrayidx2, align 4
  %add = add nsw i32 %1, %0
  %arrayidx4 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  store i32 %add, i32* %arrayidx4, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 430
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}
