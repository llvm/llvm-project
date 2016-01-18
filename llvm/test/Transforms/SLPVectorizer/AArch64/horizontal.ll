; RUN: opt -slp-vectorizer -slp-threshold=-6 -slp-vectorize-hor -S <  %s | FileCheck %s

; FIXME: The threshold is changed to keep this test case a bit smaller.
; The AArch64 cost model should not give such high costs to select statements.

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux"

; CHECK-LABEL: test_select
; CHECK: load <4 x i32>
; CHECK: load <4 x i32>
; CHECK: select <4 x i1>
define i32 @test_select(i32* noalias nocapture readonly %blk1, i32* noalias nocapture readonly %blk2, i32 %lx, i32 %h) {
entry:
  %cmp.22 = icmp sgt i32 %h, 0
  br i1 %cmp.22, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  %idx.ext = sext i32 %lx to i64
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.lr.ph
  %s.026 = phi i32 [ 0, %for.body.lr.ph ], [ %add27, %for.body ]
  %j.025 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %p2.024 = phi i32* [ %blk2, %for.body.lr.ph ], [ %add.ptr29, %for.body ]
  %p1.023 = phi i32* [ %blk1, %for.body.lr.ph ], [ %add.ptr, %for.body ]
  %0 = load i32, i32* %p1.023, align 4
  %1 = load i32, i32* %p2.024, align 4
  %sub = sub nsw i32 %0, %1
  %cmp2 = icmp slt i32 %sub, 0
  %sub3 = sub nsw i32 0, %sub
  %sub3.sub = select i1 %cmp2, i32 %sub3, i32 %sub
  %add = add nsw i32 %sub3.sub, %s.026
  %arrayidx4 = getelementptr inbounds i32, i32* %p1.023, i64 1
  %2 = load i32, i32* %arrayidx4, align 4
  %arrayidx5 = getelementptr inbounds i32, i32* %p2.024, i64 1
  %3 = load i32, i32* %arrayidx5, align 4
  %sub6 = sub nsw i32 %2, %3
  %cmp7 = icmp slt i32 %sub6, 0
  %sub9 = sub nsw i32 0, %sub6
  %v.1 = select i1 %cmp7, i32 %sub9, i32 %sub6
  %add11 = add nsw i32 %add, %v.1
  %arrayidx12 = getelementptr inbounds i32, i32* %p1.023, i64 2
  %4 = load i32, i32* %arrayidx12, align 4
  %arrayidx13 = getelementptr inbounds i32, i32* %p2.024, i64 2
  %5 = load i32, i32* %arrayidx13, align 4
  %sub14 = sub nsw i32 %4, %5
  %cmp15 = icmp slt i32 %sub14, 0
  %sub17 = sub nsw i32 0, %sub14
  %sub17.sub14 = select i1 %cmp15, i32 %sub17, i32 %sub14
  %add19 = add nsw i32 %add11, %sub17.sub14
  %arrayidx20 = getelementptr inbounds i32, i32* %p1.023, i64 3
  %6 = load i32, i32* %arrayidx20, align 4
  %arrayidx21 = getelementptr inbounds i32, i32* %p2.024, i64 3
  %7 = load i32, i32* %arrayidx21, align 4
  %sub22 = sub nsw i32 %6, %7
  %cmp23 = icmp slt i32 %sub22, 0
  %sub25 = sub nsw i32 0, %sub22
  %v.3 = select i1 %cmp23, i32 %sub25, i32 %sub22
  %add27 = add nsw i32 %add19, %v.3
  %add.ptr = getelementptr inbounds i32, i32* %p1.023, i64 %idx.ext
  %add.ptr29 = getelementptr inbounds i32, i32* %p2.024, i64 %idx.ext
  %inc = add nuw nsw i32 %j.025, 1
  %exitcond = icmp eq i32 %inc, %h
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  %s.0.lcssa = phi i32 [ 0, %entry ], [ %add27, %for.end.loopexit ]
  ret i32 %s.0.lcssa
}

;; Check whether SLP can find a reduction phi whose incoming blocks are not
;; the same as the block containing the phi.
;;
;; Came from code like,
;;
;; int s = 0;
;; for (int j = 0; j < h; j++) {
;;   s += p1[0] * p2[0]
;;   s += p1[1] * p2[1];
;;   s += p1[2] * p2[2];
;;   s += p1[3] * p2[3];
;;   if (s >= lim)
;;      break;
;;   p1 += lx;
;;   p2 += lx;
;; }
define i32 @reduction_with_br(i32* noalias nocapture readonly %blk1, i32* noalias nocapture readonly %blk2, i32 %lx, i32 %h, i32 %lim) {
; CHECK-LABEL: reduction_with_br
; CHECK: load <4 x i32>
; CHECK: load <4 x i32>
; CHECK: mul nsw <4 x i32>
entry:
  %cmp.16 = icmp sgt i32 %h, 0
  br i1 %cmp.16, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  %idx.ext = sext i32 %lx to i64
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %if.end
  %s.020 = phi i32 [ 0, %for.body.lr.ph ], [ %add13, %if.end ]
  %j.019 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %if.end ]
  %p2.018 = phi i32* [ %blk2, %for.body.lr.ph ], [ %add.ptr16, %if.end ]
  %p1.017 = phi i32* [ %blk1, %for.body.lr.ph ], [ %add.ptr, %if.end ]
  %0 = load i32, i32* %p1.017, align 4
  %1 = load i32, i32* %p2.018, align 4
  %mul = mul nsw i32 %1, %0
  %add = add nsw i32 %mul, %s.020
  %arrayidx2 = getelementptr inbounds i32, i32* %p1.017, i64 1
  %2 = load i32, i32* %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds i32, i32* %p2.018, i64 1
  %3 = load i32, i32* %arrayidx3, align 4
  %mul4 = mul nsw i32 %3, %2
  %add5 = add nsw i32 %add, %mul4
  %arrayidx6 = getelementptr inbounds i32, i32* %p1.017, i64 2
  %4 = load i32, i32* %arrayidx6, align 4
  %arrayidx7 = getelementptr inbounds i32, i32* %p2.018, i64 2
  %5 = load i32, i32* %arrayidx7, align 4
  %mul8 = mul nsw i32 %5, %4
  %add9 = add nsw i32 %add5, %mul8
  %arrayidx10 = getelementptr inbounds i32, i32* %p1.017, i64 3
  %6 = load i32, i32* %arrayidx10, align 4
  %arrayidx11 = getelementptr inbounds i32, i32* %p2.018, i64 3
  %7 = load i32, i32* %arrayidx11, align 4
  %mul12 = mul nsw i32 %7, %6
  %add13 = add nsw i32 %add9, %mul12
  %cmp14 = icmp slt i32 %add13, %lim
  br i1 %cmp14, label %if.end, label %for.end.loopexit

if.end:                                           ; preds = %for.body
  %add.ptr = getelementptr inbounds i32, i32* %p1.017, i64 %idx.ext
  %add.ptr16 = getelementptr inbounds i32, i32* %p2.018, i64 %idx.ext
  %inc = add nuw nsw i32 %j.019, 1
  %cmp = icmp slt i32 %inc, %h
  br i1 %cmp, label %for.body, label %for.end.loopexit

for.end.loopexit:                                 ; preds = %for.body, %if.end
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  %s.1 = phi i32 [ 0, %entry ], [ %add13, %for.end.loopexit ]
  ret i32 %s.1
}

; CHECK: test_unrolled_select
; CHECK: load <8 x i8>
; CHECK: load <8 x i8>
; CHECK: select <8 x i1>
define i32 @test_unrolled_select(i8* noalias nocapture readonly %blk1, i8* noalias nocapture readonly %blk2, i32 %lx, i32 %h, i32 %lim) #0 {
entry:
  %cmp.43 = icmp sgt i32 %h, 0
  br i1 %cmp.43, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  %idx.ext = sext i32 %lx to i64
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %if.end.86
  %s.047 = phi i32 [ 0, %for.body.lr.ph ], [ %add82, %if.end.86 ]
  %j.046 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %if.end.86 ]
  %p2.045 = phi i8* [ %blk2, %for.body.lr.ph ], [ %add.ptr88, %if.end.86 ]
  %p1.044 = phi i8* [ %blk1, %for.body.lr.ph ], [ %add.ptr, %if.end.86 ]
  %0 = load i8, i8* %p1.044, align 1
  %conv = zext i8 %0 to i32
  %1 = load i8, i8* %p2.045, align 1
  %conv2 = zext i8 %1 to i32
  %sub = sub nsw i32 %conv, %conv2
  %cmp3 = icmp slt i32 %sub, 0
  %sub5 = sub nsw i32 0, %sub
  %sub5.sub = select i1 %cmp3, i32 %sub5, i32 %sub
  %add = add nsw i32 %sub5.sub, %s.047
  %arrayidx6 = getelementptr inbounds i8, i8* %p1.044, i64 1
  %2 = load i8, i8* %arrayidx6, align 1
  %conv7 = zext i8 %2 to i32
  %arrayidx8 = getelementptr inbounds i8, i8* %p2.045, i64 1
  %3 = load i8, i8* %arrayidx8, align 1
  %conv9 = zext i8 %3 to i32
  %sub10 = sub nsw i32 %conv7, %conv9
  %cmp11 = icmp slt i32 %sub10, 0
  %sub14 = sub nsw i32 0, %sub10
  %v.1 = select i1 %cmp11, i32 %sub14, i32 %sub10
  %add16 = add nsw i32 %add, %v.1
  %arrayidx17 = getelementptr inbounds i8, i8* %p1.044, i64 2
  %4 = load i8, i8* %arrayidx17, align 1
  %conv18 = zext i8 %4 to i32
  %arrayidx19 = getelementptr inbounds i8, i8* %p2.045, i64 2
  %5 = load i8, i8* %arrayidx19, align 1
  %conv20 = zext i8 %5 to i32
  %sub21 = sub nsw i32 %conv18, %conv20
  %cmp22 = icmp slt i32 %sub21, 0
  %sub25 = sub nsw i32 0, %sub21
  %sub25.sub21 = select i1 %cmp22, i32 %sub25, i32 %sub21
  %add27 = add nsw i32 %add16, %sub25.sub21
  %arrayidx28 = getelementptr inbounds i8, i8* %p1.044, i64 3
  %6 = load i8, i8* %arrayidx28, align 1
  %conv29 = zext i8 %6 to i32
  %arrayidx30 = getelementptr inbounds i8, i8* %p2.045, i64 3
  %7 = load i8, i8* %arrayidx30, align 1
  %conv31 = zext i8 %7 to i32
  %sub32 = sub nsw i32 %conv29, %conv31
  %cmp33 = icmp slt i32 %sub32, 0
  %sub36 = sub nsw i32 0, %sub32
  %v.3 = select i1 %cmp33, i32 %sub36, i32 %sub32
  %add38 = add nsw i32 %add27, %v.3
  %arrayidx39 = getelementptr inbounds i8, i8* %p1.044, i64 4
  %8 = load i8, i8* %arrayidx39, align 1
  %conv40 = zext i8 %8 to i32
  %arrayidx41 = getelementptr inbounds i8, i8* %p2.045, i64 4
  %9 = load i8, i8* %arrayidx41, align 1
  %conv42 = zext i8 %9 to i32
  %sub43 = sub nsw i32 %conv40, %conv42
  %cmp44 = icmp slt i32 %sub43, 0
  %sub47 = sub nsw i32 0, %sub43
  %sub47.sub43 = select i1 %cmp44, i32 %sub47, i32 %sub43
  %add49 = add nsw i32 %add38, %sub47.sub43
  %arrayidx50 = getelementptr inbounds i8, i8* %p1.044, i64 5
  %10 = load i8, i8* %arrayidx50, align 1
  %conv51 = zext i8 %10 to i32
  %arrayidx52 = getelementptr inbounds i8, i8* %p2.045, i64 5
  %11 = load i8, i8* %arrayidx52, align 1
  %conv53 = zext i8 %11 to i32
  %sub54 = sub nsw i32 %conv51, %conv53
  %cmp55 = icmp slt i32 %sub54, 0
  %sub58 = sub nsw i32 0, %sub54
  %v.5 = select i1 %cmp55, i32 %sub58, i32 %sub54
  %add60 = add nsw i32 %add49, %v.5
  %arrayidx61 = getelementptr inbounds i8, i8* %p1.044, i64 6
  %12 = load i8, i8* %arrayidx61, align 1
  %conv62 = zext i8 %12 to i32
  %arrayidx63 = getelementptr inbounds i8, i8* %p2.045, i64 6
  %13 = load i8, i8* %arrayidx63, align 1
  %conv64 = zext i8 %13 to i32
  %sub65 = sub nsw i32 %conv62, %conv64
  %cmp66 = icmp slt i32 %sub65, 0
  %sub69 = sub nsw i32 0, %sub65
  %sub69.sub65 = select i1 %cmp66, i32 %sub69, i32 %sub65
  %add71 = add nsw i32 %add60, %sub69.sub65
  %arrayidx72 = getelementptr inbounds i8, i8* %p1.044, i64 7
  %14 = load i8, i8* %arrayidx72, align 1
  %conv73 = zext i8 %14 to i32
  %arrayidx74 = getelementptr inbounds i8, i8* %p2.045, i64 7
  %15 = load i8, i8* %arrayidx74, align 1
  %conv75 = zext i8 %15 to i32
  %sub76 = sub nsw i32 %conv73, %conv75
  %cmp77 = icmp slt i32 %sub76, 0
  %sub80 = sub nsw i32 0, %sub76
  %v.7 = select i1 %cmp77, i32 %sub80, i32 %sub76
  %add82 = add nsw i32 %add71, %v.7
  %cmp83 = icmp slt i32 %add82, %lim
  br i1 %cmp83, label %if.end.86, label %for.end.loopexit

if.end.86:                                        ; preds = %for.body
  %add.ptr = getelementptr inbounds i8, i8* %p1.044, i64 %idx.ext
  %add.ptr88 = getelementptr inbounds i8, i8* %p2.045, i64 %idx.ext
  %inc = add nuw nsw i32 %j.046, 1
  %cmp = icmp slt i32 %inc, %h
  br i1 %cmp, label %for.body, label %for.end.loopexit

for.end.loopexit:                                 ; preds = %for.body, %if.end.86
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  %s.1 = phi i32 [ 0, %entry ], [ %add82, %for.end.loopexit ]
  ret i32 %s.1
}

