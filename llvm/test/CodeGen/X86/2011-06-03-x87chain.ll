; RUN: llc < %s -mcpu=generic -mtriple=i686-- -mattr=+sse | FileCheck %s

define float @chainfail1(i64* nocapture %a, i64* nocapture %b, i32 %x, i32 %y, float* nocapture %f) nounwind uwtable noinline ssp {
entry:
  %tmp1 = load i64, i64* %a, align 8
; Insure x87 ops are properly chained, order preserved.
; CHECK: fildll
  %conv = sitofp i64 %tmp1 to float
; CHECK: fstps
  store float %conv, float* %f, align 4
; CHECK: idivl
  %div = sdiv i32 %x, %y
  %conv5 = sext i32 %div to i64
  store i64 %conv5, i64* %b, align 8
  ret float %conv
}

define float @chainfail2(i64* nocapture %a, i64* nocapture %b, i32 %x, i32 %y, float* nocapture %f) nounwind uwtable noinline ssp {
entry:
; CHECK: movl $0,
  store i64 0, i64* %b, align 8
  %mul = mul nsw i32 %y, %x
  %sub = add nsw i32 %mul, -1
  %idxprom = sext i32 %sub to i64
  %arrayidx = getelementptr inbounds i64, i64* %a, i64 %idxprom
  %tmp4 = load i64, i64* %arrayidx, align 8
; CHECK: fildll
  %conv = sitofp i64 %tmp4 to float
  store float %conv, float* %f, align 4
  ret float %conv
}

define void @PR17495() {
entry:
  br i1 undef, label %while.end, label %while.body

while.body:                                       ; preds = %while.body, %entry
  %x.1.copyload = load i24, i24* undef, align 1
  %conv = sitofp i24 %x.1.copyload to float
  %div = fmul float %conv, 0x3E80000000000000
  store float %div, float* undef, align 4
  br i1 false, label %while.end, label %while.body

while.end:                                        ; preds = %while.body, %entry
  ret void

; CHECK-LABEL: @PR17495
; CHECK-NOT: fildll
}
