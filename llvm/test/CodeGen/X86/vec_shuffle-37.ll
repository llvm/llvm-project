; RUN: llc < %s -mtriple=x86_64-linux -mcpu=core2 | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-win32 -mcpu=core2 | FileCheck %s
; RUN: llc -O0 < %s -march=x86 -mcpu=core2 | FileCheck %s --check-prefix=CHECK_O0

define <4 x i32> @t00(<4 x i32>* %a0) nounwind ssp {
entry:
; CHECK: movaps  ({{%rdi|%rcx}}), %[[XMM0:xmm[0-9]+]]
; CHECK: movaps  %[[XMM0]], %[[XMM1:xmm[0-9]+]]
; CHECK-NEXT: movss   %xmm{{[0-9]+}}, %[[XMM1]]
; CHECK-NEXT: shufps  $36, %[[XMM1]], %[[XMM0]]
  %0 = load <4 x i32>* undef, align 16
  %1 = load <4 x i32>* %a0, align 16
  %2 = shufflevector <4 x i32> %1, <4 x i32> %0, <4 x i32> <i32 0, i32 1, i32 2, i32 4>
  ret <4 x i32> %2
}

define void @t01(double* %a0) nounwind ssp {
entry:
; CHECK_O0: movsd (%eax), %xmm0
; CHECK_O0: unpcklpd  %xmm0, %xmm0
  %tmp93 = load double* %a0, align 8
  %vecinit94 = insertelement <2 x double> undef, double %tmp93, i32 1
  store <2 x double> %vecinit94, <2 x double>* undef
  ret void
}

define void @t02(<8 x i32>* %source, <2 x i32>* %dest) nounwind noinline {
entry:
; CHECK: t02
; CHECK: movaps
; CHECK: shufps
; CHECK: pshufd
; CHECK: movq
; CHECK: ret
  %0 = bitcast <8 x i32>* %source to <4 x i32>*
  %arrayidx = getelementptr inbounds <4 x i32>* %0, i64 3
  %tmp2 = load <4 x i32>* %arrayidx, align 16
  %tmp3 = extractelement <4 x i32> %tmp2, i32 0
  %tmp5 = insertelement <2 x i32> <i32 undef, i32 0>, i32 %tmp3, i32 0
  %arrayidx7 = getelementptr inbounds <8 x i32>* %source, i64 1
  %1 = bitcast <8 x i32>* %arrayidx7 to <4 x i32>*
  %tmp8 = load <4 x i32>* %1, align 16
  %tmp9 = extractelement <4 x i32> %tmp8, i32 1
  %tmp11 = insertelement <2 x i32> %tmp5, i32 %tmp9, i32 1
  store <2 x i32> %tmp11, <2 x i32>* %dest, align 8
  ret void
}
