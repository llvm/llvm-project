target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
; RUN: opt < %s -bb-vectorize -bb-vectorize-req-chain-depth=3 -instcombine -gvn -S | FileCheck %s

; This test checks the non-trivial pairing-induced cycle avoidance. Without this cycle avoidance, the algorithm would otherwise
; want to select the pairs:
; %div77 = fdiv double %sub74, %mul76.v.r1 <->   %div125 = fdiv double %mul121, %mul76.v.r2 (div125 depends on mul117)
; %add84 = fadd double %sub83, 2.000000e+00 <->   %add127 = fadd double %mul126, 1.000000e+00 (add127 depends on div77)
; %mul95 = fmul double %sub45.v.r1, %sub36.v.r1 <->   %mul88 = fmul double %sub36.v.r1, %sub87 (mul88 depends on add84)
; %mul117 = fmul double %sub39.v.r1, %sub116 <->   %mul97 = fmul double %mul96, %sub39.v.r1 (mul97 depends on mul95)
; and so a dependency cycle would be created.

declare double @fabs(double) nounwind readnone
define void @test1(double %a, double %b, double %c, double %add80, double %mul1, double %mul2.v.r1, double %mul73, double %sub, double %sub65, double %F.0, i32 %n.0, double %Bnm3.0, double %Bnm2.0, double %Bnm1.0, double %Anm3.0, double %Anm2.0, double %Anm1.0) {
entry:
  br label %go
go:
  %conv = sitofp i32 %n.0 to double
  %add35 = fadd double %conv, %a
  %sub36 = fadd double %add35, -1.000000e+00
  %add38 = fadd double %conv, %b
  %sub39 = fadd double %add38, -1.000000e+00
  %add41 = fadd double %conv, %c
  %sub42 = fadd double %add41, -1.000000e+00
  %sub45 = fadd double %add35, -2.000000e+00
  %sub48 = fadd double %add38, -2.000000e+00
  %sub51 = fadd double %add41, -2.000000e+00
  %mul52 = shl nsw i32 %n.0, 1
  %sub53 = add nsw i32 %mul52, -1
  %conv54 = sitofp i32 %sub53 to double
  %sub56 = add nsw i32 %mul52, -3
  %conv57 = sitofp i32 %sub56 to double
  %sub59 = add nsw i32 %mul52, -5
  %conv60 = sitofp i32 %sub59 to double
  %mul61 = mul nsw i32 %n.0, %n.0
  %conv62 = sitofp i32 %mul61 to double
  %mul63 = fmul double %conv62, 3.000000e+00
  %mul67 = fmul double %sub65, %conv
  %add68 = fadd double %mul63, %mul67
  %add69 = fadd double %add68, 2.000000e+00
  %sub71 = fsub double %add69, %mul2.v.r1
  %sub74 = fsub double %sub71, %mul73
  %mul75 = fmul double %conv57, 2.000000e+00
  %mul76 = fmul double %mul75, %sub42
  %div77 = fdiv double %sub74, %mul76
  %mul82 = fmul double %add80, %conv
  %sub83 = fsub double %mul63, %mul82
  %add84 = fadd double %sub83, 2.000000e+00
  %sub86 = fsub double %add84, %mul2.v.r1
  %sub87 = fsub double -0.000000e+00, %sub86
  %mul88 = fmul double %sub36, %sub87
  %mul89 = fmul double %mul88, %sub39
  %mul90 = fmul double %conv54, 4.000000e+00
  %mul91 = fmul double %mul90, %conv57
  %mul92 = fmul double %mul91, %sub51
  %mul93 = fmul double %mul92, %sub42
  %div94 = fdiv double %mul89, %mul93
  %mul95 = fmul double %sub45, %sub36
  %mul96 = fmul double %mul95, %sub48
  %mul97 = fmul double %mul96, %sub39
  %sub99 = fsub double %conv, %a
  %sub100 = fadd double %sub99, -2.000000e+00
  %mul101 = fmul double %mul97, %sub100
  %sub103 = fsub double %conv, %b
  %sub104 = fadd double %sub103, -2.000000e+00
  %mul105 = fmul double %mul101, %sub104
  %mul106 = fmul double %conv57, 8.000000e+00
  %mul107 = fmul double %mul106, %conv57
  %mul108 = fmul double %mul107, %conv60
  %sub111 = fadd double %add41, -3.000000e+00
  %mul112 = fmul double %mul108, %sub111
  %mul113 = fmul double %mul112, %sub51
  %mul114 = fmul double %mul113, %sub42
  %div115 = fdiv double %mul105, %mul114
  %sub116 = fsub double -0.000000e+00, %sub36
  %mul117 = fmul double %sub39, %sub116
  %sub119 = fsub double %conv, %c
  %sub120 = fadd double %sub119, -1.000000e+00
  %mul121 = fmul double %mul117, %sub120
  %mul123 = fmul double %mul75, %sub51
  %mul124 = fmul double %mul123, %sub42
  %div125 = fdiv double %mul121, %mul124
  %mul126 = fmul double %div77, %sub
  %add127 = fadd double %mul126, 1.000000e+00
  %mul128 = fmul double %add127, %Anm1.0
  %mul129 = fmul double %div94, %sub
  %add130 = fadd double %div125, %mul129
  %mul131 = fmul double %add130, %sub
  %mul132 = fmul double %mul131, %Anm2.0
  %add133 = fadd double %mul128, %mul132
  %mul134 = fmul double %div115, %mul1
  %mul135 = fmul double %mul134, %Anm3.0
  %add136 = fadd double %add133, %mul135
  %mul139 = fmul double %add127, %Bnm1.0
  %mul143 = fmul double %mul131, %Bnm2.0
  %add144 = fadd double %mul139, %mul143
  %mul146 = fmul double %mul134, %Bnm3.0
  %add147 = fadd double %add144, %mul146
  %div148 = fdiv double %add136, %add147
  %sub149 = fsub double %F.0, %div148
  %div150 = fdiv double %sub149, %F.0
  %call = tail call double @fabs(double %div150) nounwind readnone
  %cmp = fcmp olt double %call, 0x3CB0000000000000
  %cmp152 = icmp sgt i32 %n.0, 20000
  %or.cond = or i1 %cmp, %cmp152
  br i1 %or.cond, label %done, label %go
done:
  ret void
; CHECK: @test1
; CHECK: go:
; CHECK: %conv.v.i0.1 = insertelement <2 x i32> undef, i32 %n.0, i32 0
; FIXME: When tree pruning is deterministic, include the entire output.
}
