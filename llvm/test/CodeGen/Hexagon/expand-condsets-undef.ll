; RUN: llc -O2 < %s
; REQUIRES: asserts

target datalayout = "e-m:e-p:32:32-i64:64-a:0-v32:32-n16:32"
target triple = "hexagon"

; Function Attrs: nounwind optsize ssp
define internal fastcc void @foo() nounwind {
if.else473:
  %0 = load i64, ptr undef, align 8
  %sub = sub nsw i64 undef, %0
  %conv476 = sitofp i64 %sub to double
  %mul477 = fmul double %conv476, 0x3F50624DE0000000
  br i1 undef, label %cond.true540, label %cond.end548

cond.true540:
  %1 = fptrunc double %mul477 to float
  %2 = fptosi float %1 to i32
  br label %cond.end548

cond.end548:
  %cond549 = phi i32 [ %2, %cond.true540 ], [ undef, %if.else473 ]
  call void @bar(i32 %cond549) nounwind
  unreachable
}

declare void @bar(i32) nounwind

