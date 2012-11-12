; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=+avx | FileCheck %s

; CHECK: vcvtdq2ps %ymm
define <8 x float> @sitofp00(<8 x i32> %a) nounwind {
  %b = sitofp <8 x i32> %a to <8 x float>
  ret <8 x float> %b
}

; CHECK: vcvttps2dq %ymm
define <8 x i32> @fptosi00(<8 x float> %a) nounwind {
  %b = fptosi <8 x float> %a to <8 x i32>
  ret <8 x i32> %b
}

; CHECK: vcvtdq2pd %xmm
define <4 x double> @sitofp01(<4 x i32> %a) {
  %b = sitofp <4 x i32> %a to <4 x double>
  ret <4 x double> %b
}

; CHECK: vcvttpd2dqy %ymm
define <4 x i32> @fptosi01(<4 x double> %a) {
  %b = fptosi <4 x double> %a to <4 x i32>
  ret <4 x i32> %b
}

; CHECK: vcvtpd2psy %ymm
; CHECK-NEXT: vcvtpd2psy %ymm
; CHECK-NEXT: vinsertf128 $1
define <8 x float> @fptrunc00(<8 x double> %b) nounwind {
  %a = fptrunc <8 x double> %b to <8 x float>
  ret <8 x float> %a
}

; CHECK: vcvtps2pd %xmm
define <4 x double> @fpext00(<4 x float> %b) nounwind {
  %a = fpext <4 x float> %b to <4 x double>
  ret <4 x double> %a
}

; CHECK: vcvtsi2sdq (%
define double @funcA(i64* nocapture %e) nounwind uwtable readonly ssp {
entry:
  %tmp1 = load i64* %e, align 8
  %conv = sitofp i64 %tmp1 to double
  ret double %conv
}

; CHECK: vcvtsi2sd (%
define double @funcB(i32* nocapture %e) nounwind uwtable readonly ssp {
entry:
  %tmp1 = load i32* %e, align 4
  %conv = sitofp i32 %tmp1 to double
  ret double %conv
}

; CHECK: vcvtsi2ss (%
define float @funcC(i32* nocapture %e) nounwind uwtable readonly ssp {
entry:
  %tmp1 = load i32* %e, align 4
  %conv = sitofp i32 %tmp1 to float
  ret float %conv
}

; CHECK: vcvtsi2ssq  (%
define float @funcD(i64* nocapture %e) nounwind uwtable readonly ssp {
entry:
  %tmp1 = load i64* %e, align 8
  %conv = sitofp i64 %tmp1 to float
  ret float %conv
}

; CHECK: vcvtss2sd
define void @fpext() nounwind uwtable {
entry:
  %f = alloca float, align 4
  %d = alloca double, align 8
  %tmp = load float* %f, align 4
  %conv = fpext float %tmp to double
  store double %conv, double* %d, align 8
  ret void
}

