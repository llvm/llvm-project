; RUN: opt -hexagon-loop-idiom < %s -mtriple=hexagon-unknown-unknown -S \
; RUN:  | FileCheck %s
; RUN: opt -p hexagon-loop-idiom < %s -mtriple=hexagon-unknown-unknown -S \
; RUN:  | FileCheck %s

target triple = "hexagon"

; CHECK: define i64 @basic_pmpy
; CHECK: llvm.hexagon.M4.pmpyw
define i64 @basic_pmpy(i32 %P, i32 %Q) #0 {
entry:
  %conv = zext i32 %Q to i64
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.07 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %R.06 = phi i64 [ 0, %entry ], [ %xor.R.06, %for.body ]
  %shl = shl i32 1, %i.07
  %and = and i32 %shl, %P
  %tobool = icmp eq i32 %and, 0
  %sh_prom = zext i32 %i.07 to i64
  %shl1 = shl i64 %conv, %sh_prom
  %xor = xor i64 %shl1, %R.06
  %xor.R.06 = select i1 %tobool, i64 %R.06, i64 %xor
  %inc = add nuw nsw i32 %i.07, 1
  %exitcond = icmp ne i32 %inc, 32
  br i1 %exitcond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  %R.1.lcssa = phi i64 [ %xor.R.06, %for.body ]
  ret i64 %R.1.lcssa
}

attributes #0 = { nounwind }

