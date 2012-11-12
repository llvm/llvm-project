target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
; RUN: opt < %s -bb-vectorize -bb-vectorize-req-chain-depth=3 -S | FileCheck %s

; Simple 3-pair chain with loads and stores (with fpmath)
define void @test1(double* %a, double* %b, double* %c) nounwind uwtable readonly {
entry:
  %i0 = load double* %a, align 8
  %i1 = load double* %b, align 8
  %mul = fmul double %i0, %i1, !fpmath !2
  %arrayidx3 = getelementptr inbounds double* %a, i64 1
  %i3 = load double* %arrayidx3, align 8
  %arrayidx4 = getelementptr inbounds double* %b, i64 1
  %i4 = load double* %arrayidx4, align 8
  %mul5 = fmul double %i3, %i4, !fpmath !3
  store double %mul, double* %c, align 8
  %arrayidx5 = getelementptr inbounds double* %c, i64 1
  store double %mul5, double* %arrayidx5, align 8
  ret void
; CHECK: @test1
; CHECK: !fpmath
; CHECK: ret void
}

; Simple 3-pair chain with loads and stores (ints with range)
define void @test2(i64* %a, i64* %b, i64* %c) nounwind uwtable readonly {
entry:
  %i0 = load i64* %a, align 8, !range !0
  %i1 = load i64* %b, align 8
  %mul = mul i64 %i0, %i1
  %arrayidx3 = getelementptr inbounds i64* %a, i64 1
  %i3 = load i64* %arrayidx3, align 8, !range !1
  %arrayidx4 = getelementptr inbounds i64* %b, i64 1
  %i4 = load i64* %arrayidx4, align 8
  %mul5 = mul i64 %i3, %i4
  store i64 %mul, i64* %c, align 8
  %arrayidx5 = getelementptr inbounds i64* %c, i64 1
  store i64 %mul5, i64* %arrayidx5, align 8
  ret void
; CHECK: @test2
; CHECK-NOT: !range
; CHECK: ret void
}

!0 = metadata !{i64 0, i64 2}
!1 = metadata !{i64 3, i64 5}

!2 = metadata !{ float 5.0 }
!3 = metadata !{ float 2.5 }

