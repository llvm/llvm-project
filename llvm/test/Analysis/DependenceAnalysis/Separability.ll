; RUN: opt < %s -disable-output "-passes=print<da>" -aa-pipeline=basic-aa 2>&1 \
; RUN: | FileCheck %s
; RUN: opt < %s -analyze -basicaa -da | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.6.0"


;;  for (long int i = 0; i < 50; i++)
;;    for (long int j = 0; j < 50; j++)
;;      for (long int k = 0; k < 50; k++)
;;        for (long int l = 0; l < 50; l++) {
;;          A[n][i][j + k] = i;
;;          *B++ = A[10][i + 10][2*j - l];

define void @sep0([100 x [100 x i32]]* %A, i32* %B, i32 %n) nounwind uwtable ssp {
entry:
  br label %for.cond1.preheader

; CHECK: da analyze - output [= * * S]!
; CHECK: da analyze - flow [* * * *|<]!
; CHECK: da analyze - confused!
; CHECK: da analyze - input [* * S *]!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

for.cond1.preheader:                              ; preds = %entry, %for.inc22
  %B.addr.08 = phi i32* [ %B, %entry ], [ %scevgep11, %for.inc22 ]
  %i.07 = phi i64 [ 0, %entry ], [ %inc23, %for.inc22 ]
  br label %for.cond4.preheader

for.cond4.preheader:                              ; preds = %for.cond1.preheader, %for.inc19
  %B.addr.16 = phi i32* [ %B.addr.08, %for.cond1.preheader ], [ %scevgep9, %for.inc19 ]
  %j.05 = phi i64 [ 0, %for.cond1.preheader ], [ %inc20, %for.inc19 ]
  br label %for.cond7.preheader

for.cond7.preheader:                              ; preds = %for.cond4.preheader, %for.inc16
  %B.addr.24 = phi i32* [ %B.addr.16, %for.cond4.preheader ], [ %scevgep, %for.inc16 ]
  %k.03 = phi i64 [ 0, %for.cond4.preheader ], [ %inc17, %for.inc16 ]
  br label %for.body9

for.body9:                                        ; preds = %for.cond7.preheader, %for.body9
  %l.02 = phi i64 [ 0, %for.cond7.preheader ], [ %inc, %for.body9 ]
  %B.addr.31 = phi i32* [ %B.addr.24, %for.cond7.preheader ], [ %incdec.ptr, %for.body9 ]
  %conv = trunc i64 %i.07 to i32
  %add = add nsw i64 %j.05, %k.03
  %idxprom = sext i32 %n to i64
  %arrayidx11 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* %A, i64 %idxprom, i64 %i.07, i64 %add
  store i32 %conv, i32* %arrayidx11, align 4
  %mul = shl nsw i64 %j.05, 1
  %sub = sub nsw i64 %mul, %l.02
  %add12 = add nsw i64 %i.07, 10
  %arrayidx15 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* %A, i64 10, i64 %add12, i64 %sub
  %0 = load i32, i32* %arrayidx15, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %B.addr.31, i64 1
  store i32 %0, i32* %B.addr.31, align 4
  %inc = add nsw i64 %l.02, 1
  %exitcond = icmp ne i64 %inc, 50
  br i1 %exitcond, label %for.body9, label %for.inc16

for.inc16:                                        ; preds = %for.body9
  %scevgep = getelementptr i32, i32* %B.addr.24, i64 50
  %inc17 = add nsw i64 %k.03, 1
  %exitcond10 = icmp ne i64 %inc17, 50
  br i1 %exitcond10, label %for.cond7.preheader, label %for.inc19

for.inc19:                                        ; preds = %for.inc16
  %scevgep9 = getelementptr i32, i32* %B.addr.16, i64 2500
  %inc20 = add nsw i64 %j.05, 1
  %exitcond12 = icmp ne i64 %inc20, 50
  br i1 %exitcond12, label %for.cond4.preheader, label %for.inc22

for.inc22:                                        ; preds = %for.inc19
  %scevgep11 = getelementptr i32, i32* %B.addr.08, i64 125000
  %inc23 = add nsw i64 %i.07, 1
  %exitcond13 = icmp ne i64 %inc23, 50
  br i1 %exitcond13, label %for.cond1.preheader, label %for.end24

for.end24:                                        ; preds = %for.inc22
  ret void
}


;;  for (long int i = 0; i < 50; i++)
;;    for (long int j = 0; j < 50; j++)
;;      for (long int k = 0; k < 50; k++)
;;        for (long int l = 0; l < 50; l++) {
;;          A[i][i][j + k] = i;
;;          *B++ = A[10][i + 10][2*j - l];

define void @sep1([100 x [100 x i32]]* %A, i32* %B, i32 %n) nounwind uwtable ssp {
entry:
  br label %for.cond1.preheader

; CHECK: da analyze - output [= * * S]!
; CHECK: da analyze - flow [* * * *|<]!
; CHECK: da analyze - confused!
; CHECK: da analyze - input [* * S *]!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

for.cond1.preheader:                              ; preds = %entry, %for.inc22
  %B.addr.08 = phi i32* [ %B, %entry ], [ %scevgep11, %for.inc22 ]
  %i.07 = phi i64 [ 0, %entry ], [ %inc23, %for.inc22 ]
  br label %for.cond4.preheader

for.cond4.preheader:                              ; preds = %for.cond1.preheader, %for.inc19
  %B.addr.16 = phi i32* [ %B.addr.08, %for.cond1.preheader ], [ %scevgep9, %for.inc19 ]
  %j.05 = phi i64 [ 0, %for.cond1.preheader ], [ %inc20, %for.inc19 ]
  br label %for.cond7.preheader

for.cond7.preheader:                              ; preds = %for.cond4.preheader, %for.inc16
  %B.addr.24 = phi i32* [ %B.addr.16, %for.cond4.preheader ], [ %scevgep, %for.inc16 ]
  %k.03 = phi i64 [ 0, %for.cond4.preheader ], [ %inc17, %for.inc16 ]
  br label %for.body9

for.body9:                                        ; preds = %for.cond7.preheader, %for.body9
  %l.02 = phi i64 [ 0, %for.cond7.preheader ], [ %inc, %for.body9 ]
  %B.addr.31 = phi i32* [ %B.addr.24, %for.cond7.preheader ], [ %incdec.ptr, %for.body9 ]
  %conv = trunc i64 %i.07 to i32
  %add = add nsw i64 %j.05, %k.03
  %arrayidx11 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* %A, i64 %i.07, i64 %i.07, i64 %add
  store i32 %conv, i32* %arrayidx11, align 4
  %mul = shl nsw i64 %j.05, 1
  %sub = sub nsw i64 %mul, %l.02
  %add12 = add nsw i64 %i.07, 10
  %arrayidx15 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* %A, i64 10, i64 %add12, i64 %sub
  %0 = load i32, i32* %arrayidx15, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %B.addr.31, i64 1
  store i32 %0, i32* %B.addr.31, align 4
  %inc = add nsw i64 %l.02, 1
  %exitcond = icmp ne i64 %inc, 50
  br i1 %exitcond, label %for.body9, label %for.inc16

for.inc16:                                        ; preds = %for.body9
  %scevgep = getelementptr i32, i32* %B.addr.24, i64 50
  %inc17 = add nsw i64 %k.03, 1
  %exitcond10 = icmp ne i64 %inc17, 50
  br i1 %exitcond10, label %for.cond7.preheader, label %for.inc19

for.inc19:                                        ; preds = %for.inc16
  %scevgep9 = getelementptr i32, i32* %B.addr.16, i64 2500
  %inc20 = add nsw i64 %j.05, 1
  %exitcond12 = icmp ne i64 %inc20, 50
  br i1 %exitcond12, label %for.cond4.preheader, label %for.inc22

for.inc22:                                        ; preds = %for.inc19
  %scevgep11 = getelementptr i32, i32* %B.addr.08, i64 125000
  %inc23 = add nsw i64 %i.07, 1
  %exitcond13 = icmp ne i64 %inc23, 50
  br i1 %exitcond13, label %for.cond1.preheader, label %for.end24

for.end24:                                        ; preds = %for.inc22
  ret void
}


;;  for (long int i = 0; i < 50; i++)
;;    for (long int j = 0; j < 50; j++)
;;      for (long int k = 0; k < 50; k++)
;;        for (long int l = 0; l < 50; l++) {
;;          A[i][i][i + k][l] = i;
;;          *B++ = A[10][i + 10][j + k][l + 10];

define void @sep2([100 x [100 x [100 x i32]]]* %A, i32* %B, i32 %n) nounwind uwtable ssp {
entry:
  br label %for.cond1.preheader

; CHECK: da analyze - output [= S = =]!
; CHECK: da analyze - flow [* * * <>]!
; CHECK: da analyze - confused!
; CHECK: da analyze - input [= * * *]!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

for.cond1.preheader:                              ; preds = %entry, %for.inc26
  %B.addr.08 = phi i32* [ %B, %entry ], [ %scevgep11, %for.inc26 ]
  %i.07 = phi i64 [ 0, %entry ], [ %inc27, %for.inc26 ]
  br label %for.cond4.preheader

for.cond4.preheader:                              ; preds = %for.cond1.preheader, %for.inc23
  %B.addr.16 = phi i32* [ %B.addr.08, %for.cond1.preheader ], [ %scevgep9, %for.inc23 ]
  %j.05 = phi i64 [ 0, %for.cond1.preheader ], [ %inc24, %for.inc23 ]
  br label %for.cond7.preheader

for.cond7.preheader:                              ; preds = %for.cond4.preheader, %for.inc20
  %B.addr.24 = phi i32* [ %B.addr.16, %for.cond4.preheader ], [ %scevgep, %for.inc20 ]
  %k.03 = phi i64 [ 0, %for.cond4.preheader ], [ %inc21, %for.inc20 ]
  br label %for.body9

for.body9:                                        ; preds = %for.cond7.preheader, %for.body9
  %l.02 = phi i64 [ 0, %for.cond7.preheader ], [ %inc, %for.body9 ]
  %B.addr.31 = phi i32* [ %B.addr.24, %for.cond7.preheader ], [ %incdec.ptr, %for.body9 ]
  %conv = trunc i64 %i.07 to i32
  %add = add nsw i64 %i.07, %k.03
  %arrayidx12 = getelementptr inbounds [100 x [100 x [100 x i32]]], [100 x [100 x [100 x i32]]]* %A, i64 %i.07, i64 %i.07, i64 %add, i64 %l.02
  store i32 %conv, i32* %arrayidx12, align 4
  %add13 = add nsw i64 %l.02, 10
  %add14 = add nsw i64 %j.05, %k.03
  %add15 = add nsw i64 %i.07, 10
  %arrayidx19 = getelementptr inbounds [100 x [100 x [100 x i32]]], [100 x [100 x [100 x i32]]]* %A, i64 10, i64 %add15, i64 %add14, i64 %add13
  %0 = load i32, i32* %arrayidx19, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %B.addr.31, i64 1
  store i32 %0, i32* %B.addr.31, align 4
  %inc = add nsw i64 %l.02, 1
  %exitcond = icmp ne i64 %inc, 50
  br i1 %exitcond, label %for.body9, label %for.inc20

for.inc20:                                        ; preds = %for.body9
  %scevgep = getelementptr i32, i32* %B.addr.24, i64 50
  %inc21 = add nsw i64 %k.03, 1
  %exitcond10 = icmp ne i64 %inc21, 50
  br i1 %exitcond10, label %for.cond7.preheader, label %for.inc23

for.inc23:                                        ; preds = %for.inc20
  %scevgep9 = getelementptr i32, i32* %B.addr.16, i64 2500
  %inc24 = add nsw i64 %j.05, 1
  %exitcond12 = icmp ne i64 %inc24, 50
  br i1 %exitcond12, label %for.cond4.preheader, label %for.inc26

for.inc26:                                        ; preds = %for.inc23
  %scevgep11 = getelementptr i32, i32* %B.addr.08, i64 125000
  %inc27 = add nsw i64 %i.07, 1
  %exitcond13 = icmp ne i64 %inc27, 50
  br i1 %exitcond13, label %for.cond1.preheader, label %for.end28

for.end28:                                        ; preds = %for.inc26
  ret void
}


;;  for (long int i = 0; i < 50; i++)
;;    for (long int j = 0; j < 50; j++)
;;      for (long int k = 0; k < 50; k++)
;;        for (long int l = 0; l < 50; l++) {
;;          A[i][i][i + k][l + k] = i;
;;          *B++ = A[10][i + 10][j + k][l + 10];

define void @sep3([100 x [100 x [100 x i32]]]* %A, i32* %B, i32 %n) nounwind uwtable ssp {
entry:
  br label %for.cond1.preheader

; CHECK: da analyze - output [= S = =]!
; CHECK: da analyze - flow [* * * *|<]!
; CHECK: da analyze - confused!
; CHECK: da analyze - input [= * * *]!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

for.cond1.preheader:                              ; preds = %entry, %for.inc27
  %B.addr.08 = phi i32* [ %B, %entry ], [ %scevgep11, %for.inc27 ]
  %i.07 = phi i64 [ 0, %entry ], [ %inc28, %for.inc27 ]
  br label %for.cond4.preheader

for.cond4.preheader:                              ; preds = %for.cond1.preheader, %for.inc24
  %B.addr.16 = phi i32* [ %B.addr.08, %for.cond1.preheader ], [ %scevgep9, %for.inc24 ]
  %j.05 = phi i64 [ 0, %for.cond1.preheader ], [ %inc25, %for.inc24 ]
  br label %for.cond7.preheader

for.cond7.preheader:                              ; preds = %for.cond4.preheader, %for.inc21
  %B.addr.24 = phi i32* [ %B.addr.16, %for.cond4.preheader ], [ %scevgep, %for.inc21 ]
  %k.03 = phi i64 [ 0, %for.cond4.preheader ], [ %inc22, %for.inc21 ]
  br label %for.body9

for.body9:                                        ; preds = %for.cond7.preheader, %for.body9
  %l.02 = phi i64 [ 0, %for.cond7.preheader ], [ %inc, %for.body9 ]
  %B.addr.31 = phi i32* [ %B.addr.24, %for.cond7.preheader ], [ %incdec.ptr, %for.body9 ]
  %conv = trunc i64 %i.07 to i32
  %add = add nsw i64 %l.02, %k.03
  %add10 = add nsw i64 %i.07, %k.03
  %arrayidx13 = getelementptr inbounds [100 x [100 x [100 x i32]]], [100 x [100 x [100 x i32]]]* %A, i64 %i.07, i64 %i.07, i64 %add10, i64 %add
  store i32 %conv, i32* %arrayidx13, align 4
  %add14 = add nsw i64 %l.02, 10
  %add15 = add nsw i64 %j.05, %k.03
  %add16 = add nsw i64 %i.07, 10
  %arrayidx20 = getelementptr inbounds [100 x [100 x [100 x i32]]], [100 x [100 x [100 x i32]]]* %A, i64 10, i64 %add16, i64 %add15, i64 %add14
  %0 = load i32, i32* %arrayidx20, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %B.addr.31, i64 1
  store i32 %0, i32* %B.addr.31, align 4
  %inc = add nsw i64 %l.02, 1
  %exitcond = icmp ne i64 %inc, 50
  br i1 %exitcond, label %for.body9, label %for.inc21

for.inc21:                                        ; preds = %for.body9
  %scevgep = getelementptr i32, i32* %B.addr.24, i64 50
  %inc22 = add nsw i64 %k.03, 1
  %exitcond10 = icmp ne i64 %inc22, 50
  br i1 %exitcond10, label %for.cond7.preheader, label %for.inc24

for.inc24:                                        ; preds = %for.inc21
  %scevgep9 = getelementptr i32, i32* %B.addr.16, i64 2500
  %inc25 = add nsw i64 %j.05, 1
  %exitcond12 = icmp ne i64 %inc25, 50
  br i1 %exitcond12, label %for.cond4.preheader, label %for.inc27

for.inc27:                                        ; preds = %for.inc24
  %scevgep11 = getelementptr i32, i32* %B.addr.08, i64 125000
  %inc28 = add nsw i64 %i.07, 1
  %exitcond13 = icmp ne i64 %inc28, 50
  br i1 %exitcond13, label %for.cond1.preheader, label %for.end29

for.end29:                                        ; preds = %for.inc27
  ret void
}
