; RUN: opt < %s -analyze -basicaa -indvars -da | FileCheck %s

; This series of tests is more interesting when debugging is enabled.

; ModuleID = 'Preliminary.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.6.0"


;; may alias
;; int p0(int n, int *A, int *B) {
;;  A[0] = n;
;;  return B[1];

define i32 @p0(i32 %n, i32* %A, i32* %B) nounwind uwtable ssp {
entry:
  store i32 %n, i32* %A, align 4
  %arrayidx1 = getelementptr inbounds i32* %B, i64 1
  %0 = load i32* %arrayidx1, align 4
; CHECK: da analyze - confused!
  ret i32 %0
}


;; no alias
;; int p1(int n, int *restrict A, int *restrict B) {
;;  A[0] = n;
;;  return B[1];

define i32 @p1(i32 %n, i32* noalias %A, i32* noalias %B) nounwind uwtable ssp {
entry:
  store i32 %n, i32* %A, align 4
  %arrayidx1 = getelementptr inbounds i32* %B, i64 1
  %0 = load i32* %arrayidx1, align 4
; CHECK: da analyze - none!
  ret i32 %0
}

;; check loop nesting levels
;;  for (long int i = 0; i < n; i++)
;;    for (long int j = 0; j < n; j++)
;;      for (long int k = 0; k < n; k++)
;;        A[i][j][k] = ...
;;      for (long int k = 0; k < n; k++)
;;        ... = A[i + 3][j + 2][k + 1];

define void @p2(i64 %n, [100 x [100 x i64]]* %A, i64* %B) nounwind uwtable ssp {
entry:
  %cmp10 = icmp sgt i64 %n, 0
  br i1 %cmp10, label %for.cond1.preheader, label %for.end26

for.cond1.preheader:                              ; preds = %for.inc24, %entry
  %B.addr.012 = phi i64* [ %B.addr.1.lcssa, %for.inc24 ], [ %B, %entry ]
  %i.011 = phi i64 [ %inc25, %for.inc24 ], [ 0, %entry ]
  %cmp26 = icmp sgt i64 %n, 0
  br i1 %cmp26, label %for.cond4.preheader, label %for.inc24

for.cond4.preheader:                              ; preds = %for.inc21, %for.cond1.preheader
  %B.addr.18 = phi i64* [ %B.addr.2.lcssa, %for.inc21 ], [ %B.addr.012, %for.cond1.preheader ]
  %j.07 = phi i64 [ %inc22, %for.inc21 ], [ 0, %for.cond1.preheader ]
  %cmp51 = icmp sgt i64 %n, 0
  br i1 %cmp51, label %for.body6, label %for.cond10.loopexit

for.body6:                                        ; preds = %for.body6, %for.cond4.preheader
  %k.02 = phi i64 [ %inc, %for.body6 ], [ 0, %for.cond4.preheader ]
  %arrayidx8 = getelementptr inbounds [100 x [100 x i64]]* %A, i64 %i.011, i64 %j.07, i64 %k.02
  store i64 %i.011, i64* %arrayidx8, align 8
  %inc = add nsw i64 %k.02, 1
  %cmp5 = icmp slt i64 %inc, %n
  br i1 %cmp5, label %for.body6, label %for.cond10.loopexit

for.cond10.loopexit:                              ; preds = %for.body6, %for.cond4.preheader
  %cmp113 = icmp sgt i64 %n, 0
  br i1 %cmp113, label %for.body12, label %for.inc21

for.body12:                                       ; preds = %for.body12, %for.cond10.loopexit
  %k9.05 = phi i64 [ %inc19, %for.body12 ], [ 0, %for.cond10.loopexit ]
  %B.addr.24 = phi i64* [ %incdec.ptr, %for.body12 ], [ %B.addr.18, %for.cond10.loopexit ]
  %add = add nsw i64 %k9.05, 1
  %add13 = add nsw i64 %j.07, 2
  %add14 = add nsw i64 %i.011, 3
  %arrayidx17 = getelementptr inbounds [100 x [100 x i64]]* %A, i64 %add14, i64 %add13, i64 %add
  %0 = load i64* %arrayidx17, align 8
; CHECK: da analyze - flow [-3 -2]!
  %incdec.ptr = getelementptr inbounds i64* %B.addr.24, i64 1
  store i64 %0, i64* %B.addr.24, align 8
  %inc19 = add nsw i64 %k9.05, 1
  %cmp11 = icmp slt i64 %inc19, %n
  br i1 %cmp11, label %for.body12, label %for.inc21

for.inc21:                                        ; preds = %for.body12, %for.cond10.loopexit
  %B.addr.2.lcssa = phi i64* [ %B.addr.18, %for.cond10.loopexit ], [ %incdec.ptr, %for.body12 ]
  %inc22 = add nsw i64 %j.07, 1
  %cmp2 = icmp slt i64 %inc22, %n
  br i1 %cmp2, label %for.cond4.preheader, label %for.inc24

for.inc24:                                        ; preds = %for.inc21, %for.cond1.preheader
  %B.addr.1.lcssa = phi i64* [ %B.addr.012, %for.cond1.preheader ], [ %B.addr.2.lcssa, %for.inc21 ]
  %inc25 = add nsw i64 %i.011, 1
  %cmp = icmp slt i64 %inc25, %n
  br i1 %cmp, label %for.cond1.preheader, label %for.end26

for.end26:                                        ; preds = %for.inc24, %entry
  ret void
}


;; classify subscripts
;;  for (long int i = 0; i < n; i++)
;;  for (long int j = 0; j < n; j++)
;;  for (long int k = 0; k < n; k++)
;;  for (long int l = 0; l < n; l++)
;;  for (long int m = 0; m < n; m++)
;;  for (long int o = 0; o < n; o++)
;;  for (long int p = 0; p < n; p++)
;;  for (long int q = 0; q < n; q++)
;;  for (long int r = 0; r < n; r++)
;;  for (long int s = 0; s < n; s++)
;;  for (long int u = 0; u < n; u++)
;;  for (long int t = 0; t < n; t++) {
;;          A[i - 3] [j] [2] [k-1] [2*l + 1] [m] [p + q] [r + s] = ...
;;    ... = A[i + 3] [2] [u] [1-k] [3*l - 1] [o] [1 + n] [t + 2];

define void @p3(i64 %n, [100 x [100 x [100 x [100 x [100 x [100 x [100 x i64]]]]]]]* %A, i64* %B) nounwind uwtable ssp {
entry:
  %cmp44 = icmp sgt i64 %n, 0
  br i1 %cmp44, label %for.cond1.preheader, label %for.end90

for.cond1.preheader:                              ; preds = %for.inc88, %entry
  %B.addr.046 = phi i64* [ %B.addr.1.lcssa, %for.inc88 ], [ %B, %entry ]
  %i.045 = phi i64 [ %inc89, %for.inc88 ], [ 0, %entry ]
  %cmp240 = icmp sgt i64 %n, 0
  br i1 %cmp240, label %for.cond4.preheader, label %for.inc88

for.cond4.preheader:                              ; preds = %for.inc85, %for.cond1.preheader
  %B.addr.142 = phi i64* [ %B.addr.2.lcssa, %for.inc85 ], [ %B.addr.046, %for.cond1.preheader ]
  %j.041 = phi i64 [ %inc86, %for.inc85 ], [ 0, %for.cond1.preheader ]
  %cmp536 = icmp sgt i64 %n, 0
  br i1 %cmp536, label %for.cond7.preheader, label %for.inc85

for.cond7.preheader:                              ; preds = %for.inc82, %for.cond4.preheader
  %B.addr.238 = phi i64* [ %B.addr.3.lcssa, %for.inc82 ], [ %B.addr.142, %for.cond4.preheader ]
  %k.037 = phi i64 [ %inc83, %for.inc82 ], [ 0, %for.cond4.preheader ]
  %cmp832 = icmp sgt i64 %n, 0
  br i1 %cmp832, label %for.cond10.preheader, label %for.inc82

for.cond10.preheader:                             ; preds = %for.inc79, %for.cond7.preheader
  %B.addr.334 = phi i64* [ %B.addr.4.lcssa, %for.inc79 ], [ %B.addr.238, %for.cond7.preheader ]
  %l.033 = phi i64 [ %inc80, %for.inc79 ], [ 0, %for.cond7.preheader ]
  %cmp1128 = icmp sgt i64 %n, 0
  br i1 %cmp1128, label %for.cond13.preheader, label %for.inc79

for.cond13.preheader:                             ; preds = %for.inc76, %for.cond10.preheader
  %B.addr.430 = phi i64* [ %B.addr.5.lcssa, %for.inc76 ], [ %B.addr.334, %for.cond10.preheader ]
  %m.029 = phi i64 [ %inc77, %for.inc76 ], [ 0, %for.cond10.preheader ]
  %cmp1424 = icmp sgt i64 %n, 0
  br i1 %cmp1424, label %for.cond16.preheader, label %for.inc76

for.cond16.preheader:                             ; preds = %for.inc73, %for.cond13.preheader
  %B.addr.526 = phi i64* [ %B.addr.6.lcssa, %for.inc73 ], [ %B.addr.430, %for.cond13.preheader ]
  %o.025 = phi i64 [ %inc74, %for.inc73 ], [ 0, %for.cond13.preheader ]
  %cmp1720 = icmp sgt i64 %n, 0
  br i1 %cmp1720, label %for.cond19.preheader, label %for.inc73

for.cond19.preheader:                             ; preds = %for.inc70, %for.cond16.preheader
  %B.addr.622 = phi i64* [ %B.addr.7.lcssa, %for.inc70 ], [ %B.addr.526, %for.cond16.preheader ]
  %p.021 = phi i64 [ %inc71, %for.inc70 ], [ 0, %for.cond16.preheader ]
  %cmp2016 = icmp sgt i64 %n, 0
  br i1 %cmp2016, label %for.cond22.preheader, label %for.inc70

for.cond22.preheader:                             ; preds = %for.inc67, %for.cond19.preheader
  %B.addr.718 = phi i64* [ %B.addr.8.lcssa, %for.inc67 ], [ %B.addr.622, %for.cond19.preheader ]
  %q.017 = phi i64 [ %inc68, %for.inc67 ], [ 0, %for.cond19.preheader ]
  %cmp2312 = icmp sgt i64 %n, 0
  br i1 %cmp2312, label %for.cond25.preheader, label %for.inc67

for.cond25.preheader:                             ; preds = %for.inc64, %for.cond22.preheader
  %B.addr.814 = phi i64* [ %B.addr.9.lcssa, %for.inc64 ], [ %B.addr.718, %for.cond22.preheader ]
  %r.013 = phi i64 [ %inc65, %for.inc64 ], [ 0, %for.cond22.preheader ]
  %cmp268 = icmp sgt i64 %n, 0
  br i1 %cmp268, label %for.cond28.preheader, label %for.inc64

for.cond28.preheader:                             ; preds = %for.inc61, %for.cond25.preheader
  %B.addr.910 = phi i64* [ %B.addr.10.lcssa, %for.inc61 ], [ %B.addr.814, %for.cond25.preheader ]
  %s.09 = phi i64 [ %inc62, %for.inc61 ], [ 0, %for.cond25.preheader ]
  %cmp294 = icmp sgt i64 %n, 0
  br i1 %cmp294, label %for.cond31.preheader, label %for.inc61

for.cond31.preheader:                             ; preds = %for.inc58, %for.cond28.preheader
  %u.06 = phi i64 [ %inc59, %for.inc58 ], [ 0, %for.cond28.preheader ]
  %B.addr.105 = phi i64* [ %B.addr.11.lcssa, %for.inc58 ], [ %B.addr.910, %for.cond28.preheader ]
  %cmp321 = icmp sgt i64 %n, 0
  br i1 %cmp321, label %for.body33, label %for.inc58

for.body33:                                       ; preds = %for.body33, %for.cond31.preheader
  %t.03 = phi i64 [ %inc, %for.body33 ], [ 0, %for.cond31.preheader ]
  %B.addr.112 = phi i64* [ %incdec.ptr, %for.body33 ], [ %B.addr.105, %for.cond31.preheader ]
  %add = add nsw i64 %r.013, %s.09
  %add34 = add nsw i64 %p.021, %q.017
  %mul = shl nsw i64 %l.033, 1
  %add3547 = or i64 %mul, 1
  %sub = add nsw i64 %k.037, -1
  %sub36 = add nsw i64 %i.045, -3
  %arrayidx43 = getelementptr inbounds [100 x [100 x [100 x [100 x [100 x [100 x [100 x i64]]]]]]]* %A, i64 %sub36, i64 %j.041, i64 2, i64 %sub, i64 %add3547, i64 %m.029, i64 %add34, i64 %add
  store i64 %i.045, i64* %arrayidx43, align 8
  %add44 = add nsw i64 %t.03, 2
  %add45 = add nsw i64 %n, 1
  %mul46 = mul nsw i64 %l.033, 3
  %sub47 = add nsw i64 %mul46, -1
  %sub48 = sub nsw i64 1, %k.037
  %add49 = add nsw i64 %i.045, 3
  %arrayidx57 = getelementptr inbounds [100 x [100 x [100 x [100 x [100 x [100 x [100 x i64]]]]]]]* %A, i64 %add49, i64 2, i64 %u.06, i64 %sub48, i64 %sub47, i64 %o.025, i64 %add45, i64 %add44
  %0 = load i64* %arrayidx57, align 8
; CHECK: da analyze - flow [-6 * * => * * * * * * * *] splitable!
; CHECK: da analyze - split level = 3, iteration = 1!
  %incdec.ptr = getelementptr inbounds i64* %B.addr.112, i64 1
  store i64 %0, i64* %B.addr.112, align 8
  %inc = add nsw i64 %t.03, 1
  %cmp32 = icmp slt i64 %inc, %n
  br i1 %cmp32, label %for.body33, label %for.inc58

for.inc58:                                        ; preds = %for.body33, %for.cond31.preheader
  %B.addr.11.lcssa = phi i64* [ %B.addr.105, %for.cond31.preheader ], [ %incdec.ptr, %for.body33 ]
  %inc59 = add nsw i64 %u.06, 1
  %cmp29 = icmp slt i64 %inc59, %n
  br i1 %cmp29, label %for.cond31.preheader, label %for.inc61

for.inc61:                                        ; preds = %for.inc58, %for.cond28.preheader
  %B.addr.10.lcssa = phi i64* [ %B.addr.910, %for.cond28.preheader ], [ %B.addr.11.lcssa, %for.inc58 ]
  %inc62 = add nsw i64 %s.09, 1
  %cmp26 = icmp slt i64 %inc62, %n
  br i1 %cmp26, label %for.cond28.preheader, label %for.inc64

for.inc64:                                        ; preds = %for.inc61, %for.cond25.preheader
  %B.addr.9.lcssa = phi i64* [ %B.addr.814, %for.cond25.preheader ], [ %B.addr.10.lcssa, %for.inc61 ]
  %inc65 = add nsw i64 %r.013, 1
  %cmp23 = icmp slt i64 %inc65, %n
  br i1 %cmp23, label %for.cond25.preheader, label %for.inc67

for.inc67:                                        ; preds = %for.inc64, %for.cond22.preheader
  %B.addr.8.lcssa = phi i64* [ %B.addr.718, %for.cond22.preheader ], [ %B.addr.9.lcssa, %for.inc64 ]
  %inc68 = add nsw i64 %q.017, 1
  %cmp20 = icmp slt i64 %inc68, %n
  br i1 %cmp20, label %for.cond22.preheader, label %for.inc70

for.inc70:                                        ; preds = %for.inc67, %for.cond19.preheader
  %B.addr.7.lcssa = phi i64* [ %B.addr.622, %for.cond19.preheader ], [ %B.addr.8.lcssa, %for.inc67 ]
  %inc71 = add nsw i64 %p.021, 1
  %cmp17 = icmp slt i64 %inc71, %n
  br i1 %cmp17, label %for.cond19.preheader, label %for.inc73

for.inc73:                                        ; preds = %for.inc70, %for.cond16.preheader
  %B.addr.6.lcssa = phi i64* [ %B.addr.526, %for.cond16.preheader ], [ %B.addr.7.lcssa, %for.inc70 ]
  %inc74 = add nsw i64 %o.025, 1
  %cmp14 = icmp slt i64 %inc74, %n
  br i1 %cmp14, label %for.cond16.preheader, label %for.inc76

for.inc76:                                        ; preds = %for.inc73, %for.cond13.preheader
  %B.addr.5.lcssa = phi i64* [ %B.addr.430, %for.cond13.preheader ], [ %B.addr.6.lcssa, %for.inc73 ]
  %inc77 = add nsw i64 %m.029, 1
  %cmp11 = icmp slt i64 %inc77, %n
  br i1 %cmp11, label %for.cond13.preheader, label %for.inc79

for.inc79:                                        ; preds = %for.inc76, %for.cond10.preheader
  %B.addr.4.lcssa = phi i64* [ %B.addr.334, %for.cond10.preheader ], [ %B.addr.5.lcssa, %for.inc76 ]
  %inc80 = add nsw i64 %l.033, 1
  %cmp8 = icmp slt i64 %inc80, %n
  br i1 %cmp8, label %for.cond10.preheader, label %for.inc82

for.inc82:                                        ; preds = %for.inc79, %for.cond7.preheader
  %B.addr.3.lcssa = phi i64* [ %B.addr.238, %for.cond7.preheader ], [ %B.addr.4.lcssa, %for.inc79 ]
  %inc83 = add nsw i64 %k.037, 1
  %cmp5 = icmp slt i64 %inc83, %n
  br i1 %cmp5, label %for.cond7.preheader, label %for.inc85

for.inc85:                                        ; preds = %for.inc82, %for.cond4.preheader
  %B.addr.2.lcssa = phi i64* [ %B.addr.142, %for.cond4.preheader ], [ %B.addr.3.lcssa, %for.inc82 ]
  %inc86 = add nsw i64 %j.041, 1
  %cmp2 = icmp slt i64 %inc86, %n
  br i1 %cmp2, label %for.cond4.preheader, label %for.inc88

for.inc88:                                        ; preds = %for.inc85, %for.cond1.preheader
  %B.addr.1.lcssa = phi i64* [ %B.addr.046, %for.cond1.preheader ], [ %B.addr.2.lcssa, %for.inc85 ]
  %inc89 = add nsw i64 %i.045, 1
  %cmp = icmp slt i64 %inc89, %n
  br i1 %cmp, label %for.cond1.preheader, label %for.end90

for.end90:                                        ; preds = %for.inc88, %entry
  ret void
}


;; cleanup around chars, shorts, ints
;;void p4(int *A, int *B, long int n)
;;  for (char i = 0; i < n; i++)
;;    A[i + 2] = ...
;;    ... = A[i];

define void @p4(i32* %A, i32* %B, i64 %n) nounwind uwtable ssp {
entry:
  %cmp1 = icmp sgt i64 %n, 0
  br i1 %cmp1, label %for.body, label %for.end

for.body:                                         ; preds = %for.body, %entry
  %i.03 = phi i8 [ %inc, %for.body ], [ 0, %entry ]
  %B.addr.02 = phi i32* [ %incdec.ptr, %for.body ], [ %B, %entry ]
  %conv2 = sext i8 %i.03 to i32
  %conv3 = sext i8 %i.03 to i64
  %add = add i64 %conv3, 2
  %arrayidx = getelementptr inbounds i32* %A, i64 %add
  store i32 %conv2, i32* %arrayidx, align 4
  %idxprom4 = sext i8 %i.03 to i64
  %arrayidx5 = getelementptr inbounds i32* %A, i64 %idxprom4
  %0 = load i32* %arrayidx5, align 4
; CHECK: da analyze - flow [*|<]!
  %incdec.ptr = getelementptr inbounds i32* %B.addr.02, i64 1
  store i32 %0, i32* %B.addr.02, align 4
  %inc = add i8 %i.03, 1
  %conv = sext i8 %inc to i64
  %cmp = icmp slt i64 %conv, %n
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}


;;void p5(int *A, int *B, long int n)
;;  for (short i = 0; i < n; i++)
;;    A[i + 2] = ...
;;    ... = A[i];

define void @p5(i32* %A, i32* %B, i64 %n) nounwind uwtable ssp {
entry:
  %cmp1 = icmp sgt i64 %n, 0
  br i1 %cmp1, label %for.body, label %for.end

for.body:                                         ; preds = %for.body, %entry
  %i.03 = phi i16 [ %inc, %for.body ], [ 0, %entry ]
  %B.addr.02 = phi i32* [ %incdec.ptr, %for.body ], [ %B, %entry ]
  %conv2 = sext i16 %i.03 to i32
  %conv3 = sext i16 %i.03 to i64
  %add = add i64 %conv3, 2
  %arrayidx = getelementptr inbounds i32* %A, i64 %add
  store i32 %conv2, i32* %arrayidx, align 4
  %idxprom4 = sext i16 %i.03 to i64
  %arrayidx5 = getelementptr inbounds i32* %A, i64 %idxprom4
  %0 = load i32* %arrayidx5, align 4
; CHECK: da analyze - flow [*|<]!
  %incdec.ptr = getelementptr inbounds i32* %B.addr.02, i64 1
  store i32 %0, i32* %B.addr.02, align 4
  %inc = add i16 %i.03, 1
  %conv = sext i16 %inc to i64
  %cmp = icmp slt i64 %conv, %n
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}


;;void p6(int *A, int *B, long int n)
;;  for (int i = 0; i < n; i++)
;;    A[i + 2] = ...
;;    ... = A[i];

define void @p6(i32* %A, i32* %B, i64 %n) nounwind uwtable ssp {
entry:
  %cmp1 = icmp sgt i64 %n, 0
  br i1 %cmp1, label %for.body, label %for.end

for.body:                                         ; preds = %for.body, %entry
  %i.03 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %B.addr.02 = phi i32* [ %incdec.ptr, %for.body ], [ %B, %entry ]
  %add = add nsw i32 %i.03, 2
  %idxprom = sext i32 %add to i64
  %arrayidx = getelementptr inbounds i32* %A, i64 %idxprom
  store i32 %i.03, i32* %arrayidx, align 4
  %idxprom2 = sext i32 %i.03 to i64
  %arrayidx3 = getelementptr inbounds i32* %A, i64 %idxprom2
  %0 = load i32* %arrayidx3, align 4
; CHECK: da analyze - consistent flow [2]!
  %incdec.ptr = getelementptr inbounds i32* %B.addr.02, i64 1
  store i32 %0, i32* %B.addr.02, align 4
  %inc = add nsw i32 %i.03, 1
  %conv = sext i32 %inc to i64
  %cmp = icmp slt i64 %conv, %n
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}


;;void p7(unsigned *A, unsigned *B,  char n)
;;  A[n] = ...
;;  ... = A[n + 1];

define void @p7(i32* %A, i32* %B, i8 signext %n) nounwind uwtable ssp {
entry:
  %idxprom = sext i8 %n to i64
  %arrayidx = getelementptr inbounds i32* %A, i64 %idxprom
  store i32 0, i32* %arrayidx, align 4
  %conv = sext i8 %n to i64
  %add = add i64 %conv, 1
  %arrayidx2 = getelementptr inbounds i32* %A, i64 %add
  %0 = load i32* %arrayidx2, align 4
; CHECK: da analyze - none!
  store i32 %0, i32* %B, align 4
  ret void
}



;;void p8(unsigned *A, unsigned *B,  short n)
;;  A[n] = ...
;;  ... = A[n + 1];

define void @p8(i32* %A, i32* %B, i16 signext %n) nounwind uwtable ssp {
entry:
  %idxprom = sext i16 %n to i64
  %arrayidx = getelementptr inbounds i32* %A, i64 %idxprom
  store i32 0, i32* %arrayidx, align 4
  %conv = sext i16 %n to i64
  %add = add i64 %conv, 1
  %arrayidx2 = getelementptr inbounds i32* %A, i64 %add
  %0 = load i32* %arrayidx2, align 4
; CHECK: da analyze - none!
  store i32 %0, i32* %B, align 4
  ret void
}


;;void p9(unsigned *A, unsigned *B,  int n)
;;  A[n] = ...
;;  ... = A[n + 1];

define void @p9(i32* %A, i32* %B, i32 %n) nounwind uwtable ssp {
entry:
  %idxprom = sext i32 %n to i64
  %arrayidx = getelementptr inbounds i32* %A, i64 %idxprom
  store i32 0, i32* %arrayidx, align 4
  %add = add nsw i32 %n, 1
  %idxprom1 = sext i32 %add to i64
  %arrayidx2 = getelementptr inbounds i32* %A, i64 %idxprom1
  %0 = load i32* %arrayidx2, align 4
; CHECK: da analyze - none!
  store i32 %0, i32* %B, align 4
  ret void
}


;;void p10(unsigned *A, unsigned *B,  unsigned n)
;;  A[n] = ...
;;  ... = A[n + 1];

define void @p10(i32* %A, i32* %B, i32 %n) nounwind uwtable ssp {
entry:
  %idxprom = zext i32 %n to i64
  %arrayidx = getelementptr inbounds i32* %A, i64 %idxprom
  store i32 0, i32* %arrayidx, align 4
  %add = add i32 %n, 1
  %idxprom1 = zext i32 %add to i64
  %arrayidx2 = getelementptr inbounds i32* %A, i64 %idxprom1
  %0 = load i32* %arrayidx2, align 4
; CHECK: da analyze - none!
  store i32 %0, i32* %B, align 4
  ret void
}
