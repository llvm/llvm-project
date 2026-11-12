; REQUIRES: asserts
; RUN: opt < %s -disable-output -stats -passes=loop-unroll -info-output-file - | FileCheck %s --check-prefix=STATS
; STATS: 1 loop-unroll - Number of loops unrolled (completely or otherwise)
; Test that llvm.annotation intrinsic do not count against the loop body size
; and prevent unrolling.
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"

@B = common global i32 0, align 4

define void @foo(ptr noalias %A, i32 %B, i32 %C) {
entry:
  br label %for.body

; A loop that has a small loop body (except for the annotations) that should be
; unrolled with the default heuristic. Make sure the extra annotations do not
; prevent unrolling
for.body:                                         ; preds = %entry, %for.body
  %i.01 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  ; The real loop.
  %mul = mul nsw i32 %B, %C
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i.01
  store i32 %mul, ptr %arrayidx, align 4
  %inc = add nsw i32 %i.01, 1
  %exitcond = icmp ne i32 %inc, 4

  ; A bunch of annotations
  %annot.0 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.1 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.2 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.3 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.4 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.5 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.6 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.7 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.8 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.9 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.10 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.11 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.12 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.13 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.14 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.15 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.16 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.17 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.18 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.19 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.20 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.21 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.22 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.23 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.24 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.25 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.26 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.27 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.28 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.29 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.30 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.31 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.32 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.33 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.34 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.35 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.36 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.37 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.38 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.39 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.40 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.41 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.42 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.43 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.44 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.45 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.46 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.47 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.48 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.49 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.50 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.51 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.52 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.53 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.54 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.55 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.56 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.57 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.58 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.59 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.60 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.61 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.62 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.63 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.64 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.65 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.66 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.67 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.68 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.69 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.70 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.71 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.72 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.73 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.74 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.75 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.76 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.77 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.78 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.79 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.80 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.81 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.82 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.83 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.84 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.85 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.86 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.87 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.88 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.89 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.90 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.91 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.92 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.93 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.94 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.95 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.96 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.97 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.98 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  %annot.99 = tail call i32 @llvm.annotation.i32(i32 %i.01, ptr null, ptr null, i32 0)
  br i1 %exitcond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}

declare i32 @llvm.annotation.i32(i32, ptr, ptr, i32)
