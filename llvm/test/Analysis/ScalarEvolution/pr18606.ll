; RUN: opt -S -passes=indvars < %s | FileCheck %s

; CHECK: @main
; CHECK: %mul.lcssa5 = phi i32 [ %a.promoted4, %entry ], [ %mul.30, %for.body3 ]
; CHECK: %mul = mul nsw i32 %mul.lcssa5, %mul.lcssa5
; CHECK: %mul.30 = mul nsw i32 %mul.29, %mul.29

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a = local_unnamed_addr global i32 0, align 4
@b = local_unnamed_addr global i32 0, align 4

; Function Attrs: norecurse nounwind uwtable
define i32 @main() local_unnamed_addr {
entry:
  %a.promoted4 = load i32, ptr @a, align 4
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %entry, %for.body3
  %mul.lcssa5 = phi i32 [ %a.promoted4, %entry ], [ %mul.30, %for.body3 ]
  %i.03 = phi i32 [ 0, %entry ], [ %inc5, %for.body3 ]
  br label %for.body3

for.body3:                                        ; preds = %for.cond1.preheader
  %mul = mul nsw i32 %mul.lcssa5, %mul.lcssa5
  %mul.1 = mul nsw i32 %mul, %mul
  %mul.2 = mul nsw i32 %mul.1, %mul.1
  %mul.3 = mul nsw i32 %mul.2, %mul.2
  %mul.4 = mul nsw i32 %mul.3, %mul.3
  %mul.5 = mul nsw i32 %mul.4, %mul.4
  %mul.6 = mul nsw i32 %mul.5, %mul.5
  %mul.7 = mul nsw i32 %mul.6, %mul.6
  %mul.8 = mul nsw i32 %mul.7, %mul.7
  %mul.9 = mul nsw i32 %mul.8, %mul.8
  %mul.10 = mul nsw i32 %mul.9, %mul.9
  %mul.11 = mul nsw i32 %mul.10, %mul.10
  %mul.12 = mul nsw i32 %mul.11, %mul.11
  %mul.13 = mul nsw i32 %mul.12, %mul.12
  %mul.14 = mul nsw i32 %mul.13, %mul.13
  %mul.15 = mul nsw i32 %mul.14, %mul.14
  %mul.16 = mul nsw i32 %mul.15, %mul.15
  %mul.17 = mul nsw i32 %mul.16, %mul.16
  %mul.18 = mul nsw i32 %mul.17, %mul.17
  %mul.19 = mul nsw i32 %mul.18, %mul.18
  %mul.20 = mul nsw i32 %mul.19, %mul.19
  %mul.21 = mul nsw i32 %mul.20, %mul.20
  %mul.22 = mul nsw i32 %mul.21, %mul.21
  %mul.23 = mul nsw i32 %mul.22, %mul.22
  %mul.24 = mul nsw i32 %mul.23, %mul.23
  %mul.25 = mul nsw i32 %mul.24, %mul.24
  %mul.26 = mul nsw i32 %mul.25, %mul.25
  %mul.27 = mul nsw i32 %mul.26, %mul.26
  %mul.28 = mul nsw i32 %mul.27, %mul.27
  %mul.29 = mul nsw i32 %mul.28, %mul.28
  %mul.30 = mul nsw i32 %mul.29, %mul.29
  %inc5 = add nuw nsw i32 %i.03, 1
  %exitcond = icmp ne i32 %inc5, 10
  br i1 %exitcond, label %for.cond1.preheader, label %for.end6

for.end6:                                         ; preds = %for.body3
  %mul.lcssa.lcssa = phi i32 [ %mul.30, %for.body3 ]
  %inc.lcssa.lcssa = phi i32 [ 31, %for.body3 ]
  store i32 %mul.lcssa.lcssa, ptr @a, align 4
  store i32 %inc.lcssa.lcssa, ptr @b, align 4
  ret i32 0
}
