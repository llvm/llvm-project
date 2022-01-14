; RUN: llc %s -O0 -verify-machineinstrs -o - | FileCheck %s
; This file uses to trigger a machine verifier error because we
; were generating a stack adjustement with SP as second argument,
; which is unpredictable behavior for t2ADDrr.
; This file has been generated from the constpool test of the test-suite.
target datalayout = "e-m:o-p:32:32-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "thumbv7s-apple-ios"

@.str = external unnamed_addr constant [21 x i8], align 1

; CHECK-LABEL: f:
; CHECK: movw [[ADDR:(r[0-9]+|lr)]], #
; CHECK-NEXT: add [[ADDR]], sp
; CHECK-NEXT: vst1.64 {d{{[0-9]+}}, d{{[0-9]+}}}, [[[ADDR]]:128]
define <4 x float> @f(<4 x float> %x) {
entry:
  %.compoundliteral7837 = alloca <4 x float>, align 16
  %.compoundliteral7865 = alloca <4 x float>, align 16
  %.compoundliteral7991 = alloca <4 x float>, align 16
  %.compoundliteral8019 = alloca <4 x float>, align 16
  %.compoundliteral8061 = alloca <4 x float>, align 16
  %.compoundliteral8075 = alloca <4 x float>, align 16
  %.compoundliteral8089 = alloca <4 x float>, align 16
  %.compoundliteral8103 = alloca <4 x float>, align 16
  %.compoundliteral8117 = alloca <4 x float>, align 16
  %.compoundliteral8145 = alloca <4 x float>, align 16
  %.compoundliteral8243 = alloca <4 x float>, align 16
  %.compoundliteral8285 = alloca <4 x float>, align 16
  %.compoundliteral8299 = alloca <4 x float>, align 16
  %.compoundliteral8313 = alloca <4 x float>, align 16
  %.compoundliteral8327 = alloca <4 x float>, align 16
  %.compoundliteral9601 = alloca <4 x float>, align 16
  %.compoundliteral9615 = alloca <4 x float>, align 16
  %.compoundliteral9629 = alloca <4 x float>, align 16
  %.compoundliteral9657 = alloca <4 x float>, align 16
  %.compoundliteral9755 = alloca <4 x float>, align 16
  %.compoundliteral9769 = alloca <4 x float>, align 16
  %.compoundliteral9853 = alloca <4 x float>, align 16
  %.compoundliteral9867 = alloca <4 x float>, align 16
  %.compoundliteral9895 = alloca <4 x float>, align 16
  %.compoundliteral9909 = alloca <4 x float>, align 16
  %.compoundliteral9923 = alloca <4 x float>, align 16
  %.compoundliteral9937 = alloca <4 x float>, align 16
  %.compoundliteral9951 = alloca <4 x float>, align 16
  %.compoundliteral9979 = alloca <4 x float>, align 16
  %.compoundliteral10021 = alloca <4 x float>, align 16
  %.compoundliteral10049 = alloca <4 x float>, align 16
  %.compoundliteral10063 = alloca <4 x float>, align 16
  %.compoundliteral10077 = alloca <4 x float>, align 16
  %.compoundliteral10091 = alloca <4 x float>, align 16
  %.compoundliteral10119 = alloca <4 x float>, align 16
  %.compoundliteral10133 = alloca <4 x float>, align 16
  %.compoundliteral10147 = alloca <4 x float>, align 16
  %.compoundliteral10161 = alloca <4 x float>, align 16
  %.compoundliteral10203 = alloca <4 x float>, align 16
  %.compoundliteral10231 = alloca <4 x float>, align 16
  %.compoundliteral10385 = alloca <4 x float>, align 16
  %.compoundliteral10399 = alloca <4 x float>, align 16
  %.compoundliteral10413 = alloca <4 x float>, align 16
  %.compoundliteral10539 = alloca <4 x float>, align 16
  %.compoundliteral10553 = alloca <4 x float>, align 16
  %.compoundliteral10567 = alloca <4 x float>, align 16
  %.compoundliteral10581 = alloca <4 x float>, align 16
  %.compoundliteral10595 = alloca <4 x float>, align 16
  %.compoundliteral10609 = alloca <4 x float>, align 16
  %.compoundliteral10623 = alloca <4 x float>, align 16
  %.compoundliteral10637 = alloca <4 x float>, align 16
  %.compoundliteral10665 = alloca <4 x float>, align 16
  %.compoundliteral10693 = alloca <4 x float>, align 16
  %.compoundliteral10707 = alloca <4 x float>, align 16
  %.compoundliteral10721 = alloca <4 x float>, align 16
  %.compoundliteral10735 = alloca <4 x float>, align 16
  %.compoundliteral10749 = alloca <4 x float>, align 16
  %.compoundliteral10763 = alloca <4 x float>, align 16
  %.compoundliteral10945 = alloca <4 x float>, align 16
  %.compoundliteral10959 = alloca <4 x float>, align 16
  %.compoundliteral10987 = alloca <4 x float>, align 16
  %.compoundliteral11001 = alloca <4 x float>, align 16
  %.compoundliteral11015 = alloca <4 x float>, align 16
  %.compoundliteral11197 = alloca <4 x float>, align 16
  %.compoundliteral11421 = alloca <4 x float>, align 16
  %.compoundliteral11435 = alloca <4 x float>, align 16
  %.compoundliteral11463 = alloca <4 x float>, align 16
  %.compoundliteral11477 = alloca <4 x float>, align 16
  %.compoundliteral11491 = alloca <4 x float>, align 16
  %.compoundliteral11519 = alloca <4 x float>, align 16
  %.compoundliteral11533 = alloca <4 x float>, align 16
  %.compoundliteral11547 = alloca <4 x float>, align 16
  %.compoundliteral11631 = alloca <4 x float>, align 16
  %.compoundliteral11645 = alloca <4 x float>, align 16
  %.compoundliteral11659 = alloca <4 x float>, align 16
  %.compoundliteral11701 = alloca <4 x float>, align 16
  %.compoundliteral11743 = alloca <4 x float>, align 16
  %.compoundliteral11757 = alloca <4 x float>, align 16
  %.compoundliteral11771 = alloca <4 x float>, align 16
  %.compoundliteral11785 = alloca <4 x float>, align 16
  %.compoundliteral11799 = alloca <4 x float>, align 16
  %.compoundliteral11827 = alloca <4 x float>, align 16
  %.compoundliteral11841 = alloca <4 x float>, align 16
  %.compoundliteral11855 = alloca <4 x float>, align 16
  %.compoundliteral11869 = alloca <4 x float>, align 16
  %.compoundliteral11939 = alloca <4 x float>, align 16
  %.compoundliteral11953 = alloca <4 x float>, align 16
  %.compoundliteral11967 = alloca <4 x float>, align 16
  %.compoundliteral11981 = alloca <4 x float>, align 16
  %.compoundliteral11995 = alloca <4 x float>, align 16
  %.compoundliteral12023 = alloca <4 x float>, align 16
  %.compoundliteral12051 = alloca <4 x float>, align 16
  %.compoundliteral12065 = alloca <4 x float>, align 16
  %.compoundliteral12247 = alloca <4 x float>, align 16
  %.compoundliteral12261 = alloca <4 x float>, align 16
  %.compoundliteral12275 = alloca <4 x float>, align 16
  %.compoundliteral12499 = alloca <4 x float>, align 16
  %.compoundliteral12541 = alloca <4 x float>, align 16
  %.compoundliteral12555 = alloca <4 x float>, align 16
  %.compoundliteral12751 = alloca <4 x float>, align 16
  %.compoundliteral12891 = alloca <4 x float>, align 16
  %.compoundliteral12905 = alloca <4 x float>, align 16
  %.compoundliteral12919 = alloca <4 x float>, align 16
  %.compoundliteral12933 = alloca <4 x float>, align 16
  %.compoundliteral12947 = alloca <4 x float>, align 16
  %.compoundliteral12961 = alloca <4 x float>, align 16
  %.compoundliteral12975 = alloca <4 x float>, align 16
  %.compoundliteral12989 = alloca <4 x float>, align 16
  %.compoundliteral13003 = alloca <4 x float>, align 16
  %.compoundliteral13017 = alloca <4 x float>, align 16
  %.compoundliteral13031 = alloca <4 x float>, align 16
  %.compoundliteral13423 = alloca <4 x float>, align 16
  %.compoundliteral13437 = alloca <4 x float>, align 16
  %.compoundliteral13493 = alloca <4 x float>, align 16
  %.compoundliteral13535 = alloca <4 x float>, align 16
  %.compoundliteral13549 = alloca <4 x float>, align 16
  %.compoundliteral13647 = alloca <4 x float>, align 16
  %.compoundliteral13675 = alloca <4 x float>, align 16
  %.compoundliteral13689 = alloca <4 x float>, align 16
  %.compoundliteral13703 = alloca <4 x float>, align 16
  %.compoundliteral13717 = alloca <4 x float>, align 16
  %.compoundliteral13745 = alloca <4 x float>, align 16
  %.compoundliteral13759 = alloca <4 x float>, align 16
  %.compoundliteral13773 = alloca <4 x float>, align 16
  %.compoundliteral13787 = alloca <4 x float>, align 16
  %.compoundliteral13941 = alloca <4 x float>, align 16
  %.compoundliteral13969 = alloca <4 x float>, align 16
  %.compoundliteral13983 = alloca <4 x float>, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0x40746999A0000000, float 0xC0719B3340000000, float 0xC070B66660000000, float 0xC07404CCC0000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0x40701B3340000000, float 0x405B866660000000, float 0xC0763999A0000000, float 4.895000e+02>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp1 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add68 = fadd <4 x float> %tmp1, %tmp
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add68, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp2 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add76 = fadd float undef, 0x4074C999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp3 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins77 = insertelement <4 x float> %tmp3, float %add76, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins77, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp4 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext78 = extractelement <4 x float> %tmp4, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add79 = fadd float %vecext78, 0x40776E6660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp5 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins80 = insertelement <4 x float> %tmp5, float %add79, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins80, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0x40678CCCC0000000, float 0xC03E4CCCC0000000, float -4.170000e+02, float -1.220000e+02>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp6 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add82 = fadd <4 x float> undef, %tmp6
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add82, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp7 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext83 = extractelement <4 x float> %tmp7, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add84 = fadd float %vecext83, 1.300000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp8 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins85 = insertelement <4 x float> %tmp8, float %add84, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins85, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp9 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext86 = extractelement <4 x float> %tmp9, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add93 = fadd float undef, 0xC076C66660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp10 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins94 = insertelement <4 x float> %tmp10, float %add93, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0x406C2999A0000000, float 8.050000e+01, float 0xC0794999A0000000, float 0xC073E4CCC0000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp11 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp12 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add96 = fadd <4 x float> %tmp12, %tmp11
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp13 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext97 = extractelement <4 x float> %tmp13, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add98 = fadd float %vecext97, 0x4079E66660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp14 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins102 = insertelement <4 x float> undef, float undef, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins102, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp15 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add104 = fadd float undef, 0x406AB999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp16 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0xC0531999A0000000, float 0xC0737999A0000000, float 0x407CB33340000000, float 0xC06DCCCCC0000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext579 = extractelement <4 x float> undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add580 = fadd float %vecext579, 0xC07424CCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp17 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins581 = insertelement <4 x float> %tmp17, float %add580, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins581, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp18 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext582 = extractelement <4 x float> %tmp18, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add583 = fadd float %vecext582, 0x40444CCCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp19 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext590 = extractelement <4 x float> %tmp19, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add591 = fadd float %vecext590, 1.725000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins592 = insertelement <4 x float> undef, float %add591, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins592, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp20 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add594 = fadd float undef, 0xC05B466660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add605 = fadd float undef, 0x407164CCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp21 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add616 = fadd float undef, 1.885000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp22 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp23 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins620 = insertelement <4 x float> undef, float undef, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins620, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext621 = extractelement <4 x float> undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add622 = fadd float %vecext621, 0x40709B3340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins623 = insertelement <4 x float> undef, float %add622, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins623, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp24 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext624 = extractelement <4 x float> %tmp24, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add625 = fadd float %vecext624, 0xC064033340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp25 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins626 = insertelement <4 x float> %tmp25, float %add625, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins626, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0x404D0CCCC0000000, float 3.955000e+02, float 0xC0334CCCC0000000, float 0x40754E6660000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp26 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp27 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add628 = fadd <4 x float> %tmp27, %tmp26
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add628, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp28 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext629 = extractelement <4 x float> %tmp28, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add630 = fadd float %vecext629, 0x40730CCCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp29 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins631 = insertelement <4 x float> %tmp29, float %add630, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins631, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp30 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext632 = extractelement <4 x float> %tmp30, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add633 = fadd float %vecext632, 0xC0630999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp31 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins634 = insertelement <4 x float> %tmp31, float %add633, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins634, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp32 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext635 = extractelement <4 x float> %tmp32, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add636 = fadd float %vecext635, 0xC078833340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp33 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp34 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp35 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add658 = fadd float undef, 0xC04A4CCCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext663 = extractelement <4 x float> undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp36 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins665 = insertelement <4 x float> %tmp36, float undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext694 = extractelement <4 x float> undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add695 = fadd float %vecext694, 0xC03CCCCCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp37 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins696 = insertelement <4 x float> %tmp37, float %add695, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins696, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0xC069FCCCC0000000, float 0xC07C6E6660000000, float 0x4067E33340000000, float 0x4078DB3340000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp38 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext699 = extractelement <4 x float> undef, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add703 = fadd float undef, 0x4068F33340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins704 = insertelement <4 x float> undef, float %add703, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins704, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp39 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp40 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins710 = insertelement <4 x float> %tmp40, float undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins710, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0xC05D9999A0000000, float 0x405D6CCCC0000000, float 0x40765CCCC0000000, float 0xC07C64CCC0000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp41 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp42 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add712 = fadd <4 x float> %tmp42, %tmp41
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add712, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp43 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext713 = extractelement <4 x float> %tmp43, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp44 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins715 = insertelement <4 x float> %tmp44, float undef, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp45 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext716 = extractelement <4 x float> %tmp45, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add717 = fadd float %vecext716, -4.315000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp46 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins718 = insertelement <4 x float> %tmp46, float %add717, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins718, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp47 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext719 = extractelement <4 x float> %tmp47, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add723 = fadd float undef, 0xC06A6CCCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins724 = insertelement <4 x float> undef, float %add723, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add726 = fadd <4 x float> undef, undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext730 = extractelement <4 x float> undef, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add731 = fadd float %vecext730, 0xC0759CCCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp48 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins732 = insertelement <4 x float> %tmp48, float %add731, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins732, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp49 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext733 = extractelement <4 x float> %tmp49, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp50 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins738 = insertelement <4 x float> %tmp50, float undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0x406E6CCCC0000000, float 0xC07A766660000000, float 0xC0608CCCC0000000, float 0xC063333340000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp51 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add740 = fadd <4 x float> undef, %tmp51
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp52 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext741 = extractelement <4 x float> %tmp52, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add742 = fadd float %vecext741, 0xC07984CCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp53 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins743 = insertelement <4 x float> %tmp53, float %add742, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins743, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp54 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp55 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add754 = fadd <4 x float> %tmp55, undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add754, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp56 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext755 = extractelement <4 x float> %tmp56, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add756 = fadd float %vecext755, 0xC070ACCCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp57 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins757 = insertelement <4 x float> %tmp57, float %add756, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add765 = fadd float undef, 0x405BA66660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp58 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins766 = insertelement <4 x float> %tmp58, float %add765, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp59 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext769 = extractelement <4 x float> %tmp59, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add770 = fadd float %vecext769, 0x40797199A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp60 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins771 = insertelement <4 x float> %tmp60, float %add770, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins771, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp61 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add776 = fadd float undef, 0xC055F33340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins777 = insertelement <4 x float> undef, float %add776, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp62 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp63 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add782 = fadd <4 x float> %tmp63, undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add782, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp64 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext783 = extractelement <4 x float> %tmp64, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add784 = fadd float %vecext783, -3.455000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0xC07A866660000000, float 0xC05CF999A0000000, float 0xC0757199A0000000, float -3.845000e+02>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add796 = fadd <4 x float> undef, undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add796, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp65 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add801 = fadd float undef, 3.045000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp66 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins802 = insertelement <4 x float> %tmp66, float %add801, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins802, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext803 = extractelement <4 x float> undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp67 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp68 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add810 = fadd <4 x float> undef, %tmp68
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add810, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp69 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext811 = extractelement <4 x float> %tmp69, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp70 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins813 = insertelement <4 x float> %tmp70, float undef, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext817 = extractelement <4 x float> undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add818 = fadd float %vecext817, -4.830000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins822 = insertelement <4 x float> undef, float undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins822, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 2.700000e+01, float 0xC05F666660000000, float 0xC07D0199A0000000, float 0x407A6CCCC0000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp71 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp72 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add824 = fadd <4 x float> %tmp72, %tmp71
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add838 = fadd <4 x float> undef, undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add838, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp73 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext839 = extractelement <4 x float> %tmp73, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add849 = fadd float undef, 0xC07C266660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0xC07D566660000000, float 0xC06D233340000000, float 0x4068B33340000000, float 0xC07ADCCCC0000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp74 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add852 = fadd <4 x float> %tmp74, undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext856 = extractelement <4 x float> undef, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add857 = fadd float %vecext856, 0xC070666660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp75 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp76 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext859 = extractelement <4 x float> %tmp76, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add860 = fadd float %vecext859, 4.705000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp77 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins861 = insertelement <4 x float> %tmp77, float %add860, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins889 = insertelement <4 x float> undef, float undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins889, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp78 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext890 = extractelement <4 x float> %tmp78, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add891 = fadd float %vecext890, 0xC070633340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp79 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins892 = insertelement <4 x float> %tmp79, float %add891, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins892, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0x4063D33340000000, float 0xC076433340000000, float 0x407C966660000000, float 0xC07B5199A0000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp80 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp81 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add894 = fadd <4 x float> %tmp81, %tmp80
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add894, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext895 = extractelement <4 x float> undef, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add896 = fadd float %vecext895, 0xC070F33340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins897 = insertelement <4 x float> undef, float %add896, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp82 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext898 = extractelement <4 x float> %tmp82, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add899 = fadd float %vecext898, 0xC076F33340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins900 = insertelement <4 x float> undef, float %add899, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp83 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext901 = extractelement <4 x float> %tmp83, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add902 = fadd float %vecext901, 0xC054ECCCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp84 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins903 = insertelement <4 x float> %tmp84, float %add902, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins903, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext904 = extractelement <4 x float> undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add905 = fadd float %vecext904, 0x4056A66660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp85 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins906 = insertelement <4 x float> %tmp85, float %add905, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0xC07EFCCCC0000000, float 1.795000e+02, float 0x407E3E6660000000, float 0x4070633340000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp86 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp87 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add908 = fadd <4 x float> %tmp87, %tmp86
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add908, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp88 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp89 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp90 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext912 = extractelement <4 x float> %tmp90, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add913 = fadd float %vecext912, 2.575000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins914 = insertelement <4 x float> undef, float %add913, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp91 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext915 = extractelement <4 x float> %tmp91, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add916 = fadd float %vecext915, -3.115000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp92 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins917 = insertelement <4 x float> %tmp92, float %add916, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins917, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp93 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext918 = extractelement <4 x float> %tmp93, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add919 = fadd float %vecext918, 2.950000e+01
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp94 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins920 = insertelement <4 x float> %tmp94, float %add919, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins920, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp95 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins925 = insertelement <4 x float> %tmp95, float undef, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins925, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp96 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add927 = fadd float undef, 0xC0501999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp97 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins928 = insertelement <4 x float> %tmp97, float %add927, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext929 = extractelement <4 x float> undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add930 = fadd float %vecext929, 0xC07C8B3340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp98 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins931 = insertelement <4 x float> %tmp98, float %add930, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0xC047B33340000000, float 0x404ACCCCC0000000, float 0x40708E6660000000, float 0x4060F999A0000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp99 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp100 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext937 = extractelement <4 x float> %tmp100, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add941 = fadd float undef, -4.665000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins942 = insertelement <4 x float> undef, float %add941, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins942, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp101 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext943 = extractelement <4 x float> %tmp101, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add944 = fadd float %vecext943, 4.580000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp102 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins945 = insertelement <4 x float> %tmp102, float %add944, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins945, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp103 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add947 = fadd float undef, 0xC051933340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp104 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins948 = insertelement <4 x float> %tmp104, float %add947, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins948, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0x4060CCCCC0000000, float 0xC07BAB3340000000, float 0xC061233340000000, float 0xC076C199A0000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp105 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add955 = fadd float undef, 0x4077F4CCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp106 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins956 = insertelement <4 x float> %tmp106, float %add955, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins956, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext971 = extractelement <4 x float> undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add972 = fadd float %vecext971, 0x4024333340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp107 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins973 = insertelement <4 x float> %tmp107, float %add972, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins973, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp108 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext974 = extractelement <4 x float> %tmp108, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins976 = insertelement <4 x float> undef, float undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins976, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0x407E266660000000, float -1.225000e+02, float 0x407EB199A0000000, float 0x407BA199A0000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp109 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp110 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add978 = fadd <4 x float> %tmp110, %tmp109
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp111 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp112 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext982 = extractelement <4 x float> undef, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add983 = fadd float %vecext982, 0x407E1B3340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins984 = insertelement <4 x float> undef, float %add983, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins984, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp113 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext985 = extractelement <4 x float> %tmp113, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add986 = fadd float %vecext985, 0x406C8CCCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp114 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins987 = insertelement <4 x float> %tmp114, float %add986, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins987, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp115 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp116 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins995 = insertelement <4 x float> %tmp116, float undef, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins995, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp117 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add997 = fadd float undef, 0xC0798999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp118 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins998 = insertelement <4 x float> %tmp118, float %add997, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins998, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp119 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1013 = extractelement <4 x float> undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1014 = fadd float %vecext1013, 3.105000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp120 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp121 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1016 = extractelement <4 x float> %tmp121, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1017 = fadd float %vecext1016, 0x406A1999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp122 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1030 = extractelement <4 x float> undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1031 = fadd float %vecext1030, 2.010000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp123 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp124 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1085 = insertelement <4 x float> %tmp124, float undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp125 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1086 = extractelement <4 x float> %tmp125, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1087 = fadd float %vecext1086, -1.575000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp126 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1088 = insertelement <4 x float> %tmp126, float %add1087, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins1088, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp127 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1090 = fadd <4 x float> undef, %tmp127
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp128 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1094 = extractelement <4 x float> %tmp128, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1095 = fadd float %vecext1094, 0x4072C999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp129 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1096 = insertelement <4 x float> %tmp129, float %add1095, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins1096, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp130 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1097 = extractelement <4 x float> %tmp130, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1098 = fadd float %vecext1097, 0xC073E999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp131 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1099 = insertelement <4 x float> %tmp131, float %add1098, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins1099, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp132 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1100 = extractelement <4 x float> %tmp132, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1101 = fadd float %vecext1100, 2.885000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp133 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1102 = insertelement <4 x float> %tmp133, float %add1101, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins1102, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0x4059866660000000, float 0x4072466660000000, float 0xC078FE6660000000, float 0xC058ACCCC0000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp134 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1104 = fadd <4 x float> undef, %tmp134
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp135 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1105 = extractelement <4 x float> %tmp135, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1106 = fadd float %vecext1105, 0xC078A999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp136 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1107 = insertelement <4 x float> %tmp136, float %add1106, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1108 = extractelement <4 x float> undef, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp137 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1110 = insertelement <4 x float> %tmp137, float undef, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins1110, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp138 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1111 = extractelement <4 x float> %tmp138, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1112 = fadd float %vecext1111, 0x407D566660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp139 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1113 = insertelement <4 x float> %tmp139, float %add1112, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins1113, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1115 = fadd float undef, 0x4072B33340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1116 = insertelement <4 x float> undef, float %add1115, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins1116, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0xC0721999A0000000, float 0x4075633340000000, float 0x40794199A0000000, float 0x4061066660000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp140 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1118 = fadd <4 x float> %tmp140, undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add1118, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp141 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1119 = extractelement <4 x float> %tmp141, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1120 = fadd float %vecext1119, 0xC065A66660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1121 = insertelement <4 x float> undef, float %add1120, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins1121, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp142 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1122 = extractelement <4 x float> %tmp142, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1123 = fadd float %vecext1122, 0x4072533340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp143 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1124 = insertelement <4 x float> %tmp143, float %add1123, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1125 = extractelement <4 x float> undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1127 = insertelement <4 x float> undef, float undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins1127, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp144 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1128 = extractelement <4 x float> %tmp144, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1129 = fadd float %vecext1128, 0x405C866660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp145 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1130 = insertelement <4 x float> %tmp145, float %add1129, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0xC06D6CCCC0000000, float 0xC032E66660000000, float -1.005000e+02, float 0x40765B3340000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp146 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp147 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1132 = fadd <4 x float> %tmp147, %tmp146
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp148 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1133 = extractelement <4 x float> %tmp148, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1134 = fadd float %vecext1133, 0xC07EB999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp149 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1135 = insertelement <4 x float> %tmp149, float %add1134, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins1135, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp150 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1136 = extractelement <4 x float> %tmp150, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp151 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1138 = insertelement <4 x float> %tmp151, float undef, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins1138, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp152 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1140 = fadd float undef, 0x407AE999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp153 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1141 = insertelement <4 x float> %tmp153, float %add1140, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1142 = extractelement <4 x float> undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1143 = fadd float %vecext1142, 0x407A24CCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp154 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1144 = insertelement <4 x float> %tmp154, float %add1143, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins1144, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp155 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp156 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1146 = fadd <4 x float> %tmp156, %tmp155
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add1146, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp157 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1148 = fadd float undef, 4.145000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp158 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1158 = insertelement <4 x float> undef, float undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins1158, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0x40603999A0000000, float -9.150000e+01, float 0xC051E66660000000, float -4.825000e+02>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1218 = fadd float undef, 0xC078733340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1219 = insertelement <4 x float> undef, float %add1218, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0xC0655CCCC0000000, float -4.900000e+01, float -4.525000e+02, float 4.205000e+02>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp159 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1279 = extractelement <4 x float> %tmp159, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1280 = fadd float %vecext1279, 0xC062D999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp160 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1281 = insertelement <4 x float> %tmp160, float %add1280, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins1281, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp161 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1282 = extractelement <4 x float> %tmp161, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1283 = fadd float %vecext1282, 4.365000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp162 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1284 = insertelement <4 x float> %tmp162, float %add1283, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins1284, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp163 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp164 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1286 = fadd <4 x float> %tmp164, %tmp163
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add1286, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp165 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1288 = fadd float undef, 0xC0731199A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp166 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp167 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1444 = extractelement <4 x float> %tmp167, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1460 = insertelement <4 x float> undef, float undef, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins1460, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp168 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1462 = fadd float undef, -1.670000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1463 = insertelement <4 x float> undef, float %add1462, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp169 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1464 = extractelement <4 x float> %tmp169, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1465 = fadd float %vecext1464, 0xC066333340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp170 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1466 = insertelement <4 x float> %tmp170, float %add1465, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins1466, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 3.885000e+02, float 0x4054266660000000, float -9.500000e+01, float 8.500000e+01>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp171 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp172 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1468 = fadd <4 x float> %tmp172, %tmp171
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add1468, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp173 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1470 = fadd float undef, 0x4033B33340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp174 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1471 = insertelement <4 x float> %tmp174, float %add1470, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins1471, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp175 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1472 = extractelement <4 x float> %tmp175, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1473 = fadd float %vecext1472, 0xC05F666660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp176 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1474 = insertelement <4 x float> %tmp176, float %add1473, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp177 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1475 = extractelement <4 x float> %tmp177, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp178 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1477 = insertelement <4 x float> %tmp178, float undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins1477, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp179 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1478 = extractelement <4 x float> %tmp179, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1479 = fadd float %vecext1478, 0x407E2E6660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp180 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1480 = insertelement <4 x float> %tmp180, float %add1479, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins1480, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0xC061B33340000000, float 3.290000e+02, float 0xC067766660000000, float 0x407DB33340000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp181 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp182 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp183 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1483 = extractelement <4 x float> %tmp183, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1484 = fadd float %vecext1483, 0xC053D999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp184 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp185 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1486 = extractelement <4 x float> %tmp185, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1502 = insertelement <4 x float> undef, float undef, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins1502, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1503 = extractelement <4 x float> undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1504 = fadd float %vecext1503, -2.475000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp186 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1505 = insertelement <4 x float> %tmp186, float %add1504, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins1505, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp187 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1506 = extractelement <4 x float> %tmp187, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1507 = fadd float %vecext1506, 0x40715199A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp188 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1508 = insertelement <4 x float> %tmp188, float %add1507, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins1508, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0x40762B3340000000, float 0xC074566660000000, float 0xC07C74CCC0000000, float 0xC053F999A0000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp189 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp190 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1510 = fadd <4 x float> %tmp190, %tmp189
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add1510, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp191 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp192 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1654 = extractelement <4 x float> %tmp192, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1655 = fadd float %vecext1654, 0xC07D8CCCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp193 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1656 = insertelement <4 x float> %tmp193, float %add1655, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins1656, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1658 = fadd float undef, 0x40709999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp194 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1660 = extractelement <4 x float> undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1661 = fadd float %vecext1660, 0xC06F166660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp195 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1662 = insertelement <4 x float> %tmp195, float %add1661, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins1662, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0xC075266660000000, float 0xC072C4CCC0000000, float 0x407C4E6660000000, float -4.485000e+02>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1676 = insertelement <4 x float> undef, float undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp196 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1692 = fadd <4 x float> %tmp196, undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add1692, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp197 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1693 = extractelement <4 x float> %tmp197, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1694 = fadd float %vecext1693, 0x407A1999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp198 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1695 = insertelement <4 x float> %tmp198, float %add1694, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins1695, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp199 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1696 = extractelement <4 x float> %tmp199, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1697 = fadd float %vecext1696, 2.850000e+01
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp200 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1698 = insertelement <4 x float> %tmp200, float %add1697, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins1698, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp201 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1699 = extractelement <4 x float> %tmp201, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp202 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1701 = insertelement <4 x float> %tmp202, float undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins1701, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp203 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1704 = insertelement <4 x float> undef, float undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0xC075933340000000, float 0xC0489999A0000000, float 0xC078AB3340000000, float 0x406DFCCCC0000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp204 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp205 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp206 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1709 = insertelement <4 x float> %tmp206, float undef, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins1709, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp207 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1713 = extractelement <4 x float> undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1714 = fadd float %vecext1713, 0xC0703199A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1723 = insertelement <4 x float> undef, float undef, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp208 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1730 = extractelement <4 x float> undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1731 = fadd float %vecext1730, 4.130000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp209 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1732 = insertelement <4 x float> %tmp209, float %add1731, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins1732, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0x40551999A0000000, float 0xC0708999A0000000, float 0xC054F33340000000, float 0xC07C5999A0000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp210 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1734 = fadd <4 x float> undef, %tmp210
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp211 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1736 = fadd float undef, 0x407C3999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp212 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1737 = insertelement <4 x float> %tmp212, float %add1736, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp213 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1738 = extractelement <4 x float> %tmp213, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1739 = fadd float %vecext1738, 0xC0711E6660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp214 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1740 = insertelement <4 x float> %tmp214, float %add1739, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins1740, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp215 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1741 = extractelement <4 x float> %tmp215, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1742 = fadd float %vecext1741, -2.545000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp216 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1743 = insertelement <4 x float> %tmp216, float %add1742, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins1743, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1744 = extractelement <4 x float> undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp217 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1746 = insertelement <4 x float> %tmp217, float undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0xC076466660000000, float 0x4060BCCCC0000000, float 0x405EF999A0000000, float 0x4074766660000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp218 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1748 = fadd <4 x float> undef, %tmp218
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add1748, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp219 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1750 = fadd float undef, 0x407C6B3340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1751 = insertelement <4 x float> undef, float %add1750, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp220 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1752 = extractelement <4 x float> %tmp220, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1753 = fadd float %vecext1752, 0x40730CCCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp221 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1754 = insertelement <4 x float> %tmp221, float %add1753, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp222 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1755 = extractelement <4 x float> %tmp222, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1756 = fadd float %vecext1755, 0xC059F33340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp223 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1759 = fadd float undef, 0x40678999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp224 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1760 = insertelement <4 x float> %tmp224, float %add1759, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins1760, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0x405E333340000000, float 0x40571999A0000000, float 0xC02E333340000000, float 0x4053A66660000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp225 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1762 = fadd <4 x float> undef, %tmp225
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add1762, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp226 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1763 = extractelement <4 x float> %tmp226, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1764 = fadd float %vecext1763, 0xC0299999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp227 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1765 = insertelement <4 x float> %tmp227, float %add1764, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins1765, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp228 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1766 = extractelement <4 x float> %tmp228, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1767 = fadd float %vecext1766, 0x407DDE6660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp229 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1768 = insertelement <4 x float> %tmp229, float %add1767, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins1768, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1769 = extractelement <4 x float> undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1770 = fadd float %vecext1769, 0x407A1B3340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp230 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1771 = insertelement <4 x float> %tmp230, float %add1770, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins1771, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp231 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp232 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp233 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp234 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1779 = insertelement <4 x float> %tmp234, float undef, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins1779, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp235 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp236 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1783 = extractelement <4 x float> %tmp236, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1784 = fadd float %vecext1783, 0x405E933340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1785 = insertelement <4 x float> undef, float %add1784, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins1785, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0xC07074CCC0000000, float 0xC04D666660000000, float 3.235000e+02, float 0xC0724199A0000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp237 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1790 = fadd <4 x float> undef, %tmp237
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp238 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1791 = extractelement <4 x float> %tmp238, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1792 = fadd float %vecext1791, 0x4077DE6660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp239 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1793 = insertelement <4 x float> %tmp239, float %add1792, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins1793, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp240 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1795 = fadd float undef, 0x4055266660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp241 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1796 = insertelement <4 x float> %tmp241, float %add1795, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1799 = insertelement <4 x float> undef, float undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1800 = extractelement <4 x float> undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp242 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float -6.600000e+01, float 0xC07B2199A0000000, float 0x4011333340000000, float 0xC0635CCCC0000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp243 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp244 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp245 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp246 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1865 = fadd float undef, -2.235000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp247 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1866 = insertelement <4 x float> %tmp247, float %add1865, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp248 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp249 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1872 = insertelement <4 x float> %tmp249, float undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0x406B8999A0000000, float 0xC0696CCCC0000000, float 0xC07A34CCC0000000, float 0x407654CCC0000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp250 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1874 = fadd <4 x float> %tmp250, undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add1874, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1875 = extractelement <4 x float> undef, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp251 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1894 = insertelement <4 x float> %tmp251, float undef, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp252 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1895 = extractelement <4 x float> %tmp252, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1900 = insertelement <4 x float> undef, float undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins1900, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1905 = insertelement <4 x float> undef, float undef, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins1905, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp253 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1906 = extractelement <4 x float> %tmp253, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1907 = fadd float %vecext1906, 0xC07E5E6660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1908 = insertelement <4 x float> undef, float %add1907, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins1908, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1909 = extractelement <4 x float> undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp254 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1912 = extractelement <4 x float> %tmp254, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1913 = fadd float %vecext1912, 0xC063ECCCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp255 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp256 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1916 = fadd <4 x float> %tmp256, undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add1916, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1923 = extractelement <4 x float> undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp257 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1927 = fadd float undef, 0x40761999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp258 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1928 = insertelement <4 x float> %tmp258, float %add1927, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins1928, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 7.100000e+01, float 0xC0634999A0000000, float 0x407B0B3340000000, float 0xC07DE999A0000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp259 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp260 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1930 = fadd <4 x float> %tmp260, %tmp259
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp261 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp262 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1933 = insertelement <4 x float> %tmp262, float undef, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins1933, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp263 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1934 = extractelement <4 x float> %tmp263, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1935 = fadd float %vecext1934, 0xC07D3199A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp264 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1936 = insertelement <4 x float> %tmp264, float %add1935, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1940 = extractelement <4 x float> undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1942 = insertelement <4 x float> undef, float undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float -8.200000e+01, float 0xC04C733340000000, float 0xC077ACCCC0000000, float 0x4074566660000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp265 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp266 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp267 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1945 = extractelement <4 x float> %tmp267, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1946 = fadd float %vecext1945, 0xC074866660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1953 = insertelement <4 x float> undef, float undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins1953, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp268 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp269 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp270 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1958 = fadd <4 x float> %tmp270, %tmp269
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp271 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1959 = extractelement <4 x float> %tmp271, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1960 = fadd float %vecext1959, 0x4065ACCCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1962 = extractelement <4 x float> undef, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1963 = fadd float %vecext1962, 0xC07134CCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp272 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1964 = insertelement <4 x float> %tmp272, float %add1963, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins1964, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1965 = extractelement <4 x float> undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp273 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1967 = insertelement <4 x float> %tmp273, float undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins1967, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp274 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1968 = extractelement <4 x float> %tmp274, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1969 = fadd float %vecext1968, 7.100000e+01
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp275 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1970 = insertelement <4 x float> %tmp275, float %add1969, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins1970, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0x402E9999A0000000, float 0x407344CCC0000000, float -4.165000e+02, float 0x4078FCCCC0000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp276 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp277 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp278 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1975 = insertelement <4 x float> %tmp278, float undef, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins1975, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp279 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1976 = extractelement <4 x float> %tmp279, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1978 = insertelement <4 x float> undef, float undef, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins1978, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1979 = extractelement <4 x float> undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1981 = insertelement <4 x float> undef, float undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins1981, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1984 = insertelement <4 x float> undef, float undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins1984, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0xC06A766660000000, float 0xC07CE4CCC0000000, float -1.055000e+02, float 0x40786E6660000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1990 = extractelement <4 x float> undef, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext1996 = extractelement <4 x float> undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add1997 = fadd float %vecext1996, -1.400000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp280 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins1998 = insertelement <4 x float> %tmp280, float %add1997, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins1998, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0xC0794E6660000000, float 0xC073CCCCC0000000, float 0x407994CCC0000000, float 6.500000e+01>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext2004 = extractelement <4 x float> undef, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add2005 = fadd float %vecext2004, -1.970000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp281 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins2006 = insertelement <4 x float> %tmp281, float %add2005, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins2006, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp282 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext2007 = extractelement <4 x float> %tmp282, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp283 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins2009 = insertelement <4 x float> %tmp283, float undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp284 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext2010 = extractelement <4 x float> %tmp284, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add2011 = fadd float %vecext2010, 0xC074533340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp285 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins2012 = insertelement <4 x float> %tmp285, float %add2011, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins2012, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0xC04E733340000000, float 0xC074566660000000, float 0x4079F66660000000, float 0xC0705B3340000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp286 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp287 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp288 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext2015 = extractelement <4 x float> %tmp288, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add2016 = fadd float %vecext2015, 0xC060633340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp289 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins2017 = insertelement <4 x float> %tmp289, float %add2016, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add2022 = fadd float undef, 8.350000e+01
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp290 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins2023 = insertelement <4 x float> %tmp290, float %add2022, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp291 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext2024 = extractelement <4 x float> %tmp291, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp292 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add2028 = fadd <4 x float> %tmp292, undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add2028, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext2029 = extractelement <4 x float> undef, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add2030 = fadd float %vecext2029, -9.450000e+01
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp293 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp294 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add2036 = fadd float undef, 0x407DE66660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp295 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp296 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp297 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp298 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp299 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins2045 = insertelement <4 x float> %tmp299, float undef, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins2045, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp300 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext2046 = extractelement <4 x float> %tmp300, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add2047 = fadd float %vecext2046, 0xC065433340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext2052 = extractelement <4 x float> undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp301 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins2054 = insertelement <4 x float> %tmp301, float undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins2054, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0x4024666660000000, float 0x4079366660000000, float 0x40721B3340000000, float 0x406E533340000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp302 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add2056 = fadd <4 x float> undef, %tmp302
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add2056, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp303 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp304 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins2062 = insertelement <4 x float> %tmp304, float undef, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins2062, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp305 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp306 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext2066 = extractelement <4 x float> %tmp306, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add2067 = fadd float %vecext2066, 0x40690999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins2068 = insertelement <4 x float> undef, float %add2067, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins2068, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0xC07EFCCCC0000000, float -3.420000e+02, float 0xC07BC999A0000000, float 0x40751999A0000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp307 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp308 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add2070 = fadd <4 x float> %tmp308, %tmp307
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add2070, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp309 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext2071 = extractelement <4 x float> %tmp309, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add2072 = fadd float %vecext2071, 0x4057733340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp310 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins2073 = insertelement <4 x float> %tmp310, float %add2072, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins2073, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp311 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext2074 = extractelement <4 x float> %tmp311, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp312 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins2076 = insertelement <4 x float> %tmp312, float undef, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp313 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext2077 = extractelement <4 x float> %tmp313, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add2078 = fadd float %vecext2077, 0x4061F999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp314 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins2079 = insertelement <4 x float> %tmp314, float %add2078, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins2079, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp315 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext2080 = extractelement <4 x float> %tmp315, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add2081 = fadd float %vecext2080, 0x407A1B3340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp316 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins2082 = insertelement <4 x float> %tmp316, float %add2081, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins2082, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0x40764E6660000000, float 0x40501999A0000000, float 0xC079A4CCC0000000, float 0x4050533340000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp317 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp318 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp319 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext2085 = extractelement <4 x float> %tmp319, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add2086 = fadd float %vecext2085, 0x406E666660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins2087 = insertelement <4 x float> undef, float %add2086, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins2087, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext2480 = extractelement <4 x float> undef, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add2481 = fadd float %vecext2480, 0x4039666660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext2483 = extractelement <4 x float> undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add2484 = fadd float %vecext2483, 0xC06A3999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp320 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins2485 = insertelement <4 x float> %tmp320, float %add2484, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins2485, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp321 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add2487 = fadd float undef, 2.030000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp322 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0x4073DE6660000000, float 0x4067CCCCC0000000, float 0xC03F1999A0000000, float 4.350000e+01>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext2491 = extractelement <4 x float> undef, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp323 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp324 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext2494 = extractelement <4 x float> %tmp324, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add2495 = fadd float %vecext2494, 0xC0743CCCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp325 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins2499 = insertelement <4 x float> undef, float undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins2499, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext2500 = extractelement <4 x float> undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add2501 = fadd float %vecext2500, 0x40796E6660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp326 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp327 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext2508 = extractelement <4 x float> undef, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add2518 = fadd <4 x float> undef, undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp328 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext2519 = extractelement <4 x float> %tmp328, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add2520 = fadd float %vecext2519, 0xC0399999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp329 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add2534 = fadd float undef, 0x4072C66660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext2536 = extractelement <4 x float> undef, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add2537 = fadd float %vecext2536, 0x407D066660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp330 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins2538 = insertelement <4 x float> %tmp330, float %add2537, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins2538, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext2539 = extractelement <4 x float> undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add2540 = fadd float %vecext2539, 0x406F9999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins2580 = insertelement <4 x float> undef, float undef, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins2580, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp331 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext2581 = extractelement <4 x float> %tmp331, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add2582 = fadd float %vecext2581, 0x406BE66660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins2583 = insertelement <4 x float> undef, float %add2582, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins2583, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext2584 = extractelement <4 x float> undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add2585 = fadd float %vecext2584, 3.585000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp332 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0x40773199A0000000, float 0x407D7999A0000000, float 0xC0717199A0000000, float 0xC07E9CCCC0000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add2590 = fadd float undef, 0x407B1999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp333 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp334 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add2672 = fadd <4 x float> undef, undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add2672, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp335 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext2676 = extractelement <4 x float> undef, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add2677 = fadd float %vecext2676, 0x406D6999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp336 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins2678 = insertelement <4 x float> %tmp336, float %add2677, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins2678, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp337 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext2679 = extractelement <4 x float> %tmp337, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins2681 = insertelement <4 x float> undef, float undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins2681, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp338 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext2682 = extractelement <4 x float> %tmp338, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins2684 = insertelement <4 x float> undef, float undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp339 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp340 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp341 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add2688 = fadd float undef, 0x4063266660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins2692 = insertelement <4 x float> undef, float undef, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins2692, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp342 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext2696 = extractelement <4 x float> undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add2697 = fadd float %vecext2696, 4.140000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp343 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins2698 = insertelement <4 x float> %tmp343, float %add2697, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins2698, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0x40547999A0000000, float 0xC060633340000000, float 0x4075766660000000, float 0x4072D33340000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp344 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp345 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add2700 = fadd <4 x float> %tmp345, %tmp344
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add2700, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp346 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp347 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp348 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext2704 = extractelement <4 x float> %tmp348, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add2705 = fadd float %vecext2704, 4.700000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp349 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext3121 = extractelement <4 x float> undef, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add3125 = fadd float undef, 0xC06F266660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins3126 = insertelement <4 x float> undef, float %add3125, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins3126, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp350 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext3127 = extractelement <4 x float> %tmp350, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add3128 = fadd float %vecext3127, 0x40638999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp351 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins3129 = insertelement <4 x float> %tmp351, float %add3128, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins3129, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp352 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add3131 = fadd float undef, 3.215000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp353 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp354 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add3134 = fadd <4 x float> %tmp354, %tmp353
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add3134, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp355 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add3136 = fadd float undef, 0x4074333340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins3140 = insertelement <4 x float> undef, float undef, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins3140, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp356 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext3141 = extractelement <4 x float> %tmp356, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add3142 = fadd float %vecext3141, 2.425000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp357 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins3143 = insertelement <4 x float> %tmp357, float %add3142, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins3143, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp358 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext3144 = extractelement <4 x float> %tmp358, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add3145 = fadd float %vecext3144, -3.760000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp359 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins3146 = insertelement <4 x float> %tmp359, float %add3145, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins3146, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp360 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins3272 = insertelement <4 x float> undef, float undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins3272, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0x407B4999A0000000, float 0x40695CCCC0000000, float 0xC05C0CCCC0000000, float 0x407EB33340000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp361 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp362 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add3274 = fadd <4 x float> %tmp362, %tmp361
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add3274, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp363 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext3275 = extractelement <4 x float> %tmp363, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add3276 = fadd float %vecext3275, 0x4058066660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp364 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins3277 = insertelement <4 x float> %tmp364, float %add3276, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins3277, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp365 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext3278 = extractelement <4 x float> %tmp365, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add3279 = fadd float %vecext3278, 0xC053666660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins3280 = insertelement <4 x float> undef, float %add3279, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins3280, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp366 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext3281 = extractelement <4 x float> %tmp366, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add3282 = fadd float %vecext3281, 0xC0650CCCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp367 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins3283 = insertelement <4 x float> %tmp367, float %add3282, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins3283, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp368 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext3284 = extractelement <4 x float> %tmp368, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add3285 = fadd float %vecext3284, 0x4062533340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins3286 = insertelement <4 x float> undef, float %add3285, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp369 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp370 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext3289 = extractelement <4 x float> %tmp370, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add3290 = fadd float %vecext3289, 0xC07E133340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp371 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins3291 = insertelement <4 x float> %tmp371, float %add3290, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins3291, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext3292 = extractelement <4 x float> undef, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp372 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp373 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins3328 = insertelement <4 x float> %tmp373, float undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add3330 = fadd <4 x float> undef, undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add3330, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext3331 = extractelement <4 x float> undef, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add3332 = fadd float %vecext3331, 0x4061633340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp374 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins3333 = insertelement <4 x float> %tmp374, float %add3332, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins3333, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext3334 = extractelement <4 x float> undef, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add3335 = fadd float %vecext3334, 0x401B333340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins3336 = insertelement <4 x float> undef, float %add3335, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp375 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext3337 = extractelement <4 x float> %tmp375, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add3338 = fadd float %vecext3337, 0x403C4CCCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp376 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins3339 = insertelement <4 x float> %tmp376, float %add3338, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins3339, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp377 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext3340 = extractelement <4 x float> %tmp377, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp378 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins3342 = insertelement <4 x float> %tmp378, float undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp379 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add3344 = fadd <4 x float> %tmp379, undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add3344, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp380 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext3345 = extractelement <4 x float> %tmp380, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add3346 = fadd float %vecext3345, 0x407E7E6660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp381 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins3347 = insertelement <4 x float> %tmp381, float %add3346, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext3348 = extractelement <4 x float> undef, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add3349 = fadd float %vecext3348, 0xC05F666660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp382 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins3350 = insertelement <4 x float> %tmp382, float %add3349, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins3350, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add3352 = fadd float undef, 0xC06ACCCCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp383 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins3423 = insertelement <4 x float> undef, float undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins3423, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext3424 = extractelement <4 x float> undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add3425 = fadd float %vecext3424, 0xC05DB33340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp384 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins3426 = insertelement <4 x float> %tmp384, float %add3425, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins3426, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 2.795000e+02, float -4.065000e+02, float 0xC05CD999A0000000, float 1.825000e+02>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp385 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp386 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add3428 = fadd <4 x float> %tmp386, %tmp385
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp387 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext3429 = extractelement <4 x float> %tmp387, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add3430 = fadd float %vecext3429, 0x40695CCCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp388 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins3431 = insertelement <4 x float> %tmp388, float %add3430, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins3431, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp389 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext3432 = extractelement <4 x float> %tmp389, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add3433 = fadd float %vecext3432, 0x4052A66660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp390 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins3434 = insertelement <4 x float> %tmp390, float %add3433, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins3434, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext3435 = extractelement <4 x float> undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp391 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins3437 = insertelement <4 x float> %tmp391, float undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins3437, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp392 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext3438 = extractelement <4 x float> %tmp392, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add3439 = fadd float %vecext3438, 0xC071D999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0xC0798199A0000000, float -3.385000e+02, float 0xC050066660000000, float 0xC075E999A0000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp393 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp394 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add3442 = fadd <4 x float> %tmp394, %tmp393
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add3442, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext3443 = extractelement <4 x float> undef, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add3444 = fadd float %vecext3443, 0xC07CF999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp395 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext3446 = extractelement <4 x float> %tmp395, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add3447 = fadd float %vecext3446, 0xC06E4999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp396 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins3448 = insertelement <4 x float> %tmp396, float %add3447, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins3448, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp397 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext3449 = extractelement <4 x float> %tmp397, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add3450 = fadd float %vecext3449, 0x40779B3340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp398 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins3451 = insertelement <4 x float> %tmp398, float %add3450, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins3451, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add3453 = fadd float undef, 0xC07ADCCCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp399 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins3454 = insertelement <4 x float> %tmp399, float %add3453, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins3454, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp400 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext3457 = extractelement <4 x float> %tmp400, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add3458 = fadd float %vecext3457, -4.440000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins3459 = insertelement <4 x float> undef, float %add3458, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins3459, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp401 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext3460 = extractelement <4 x float> %tmp401, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp402 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins3462 = insertelement <4 x float> %tmp402, float undef, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins3462, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp403 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add3464 = fadd float undef, 0xC057B999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp404 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins3465 = insertelement <4 x float> %tmp404, float %add3464, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins3465, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp405 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext3466 = extractelement <4 x float> %tmp405, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add3467 = fadd float %vecext3466, 0xC07A9CCCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp406 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0x405C3999A0000000, float 0xC07C6B3340000000, float 0x407ACB3340000000, float 0xC06E0999A0000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp407 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp408 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext3477 = extractelement <4 x float> %tmp408, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins3479 = insertelement <4 x float> undef, float undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins3479, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext3480 = extractelement <4 x float> undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add3481 = fadd float %vecext3480, 0xC053F33340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp409 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins3482 = insertelement <4 x float> %tmp409, float %add3481, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins3482, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 3.565000e+02, float 0xC0464CCCC0000000, float 0x4037666660000000, float 0xC0788CCCC0000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp410 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add3484 = fadd <4 x float> %tmp410, undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add3484, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp411 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add3486 = fadd float undef, -1.415000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins3487 = insertelement <4 x float> undef, float %add3486, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins3487, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp412 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext3488 = extractelement <4 x float> %tmp412, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add3489 = fadd float %vecext3488, 0x405A1999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp413 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins3490 = insertelement <4 x float> %tmp413, float %add3489, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins3490, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add3492 = fadd float undef, 0x4078066660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp414 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins3493 = insertelement <4 x float> %tmp414, float %add3492, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins3493, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp415 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add3495 = fadd float undef, 0xC0798999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp416 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins3496 = insertelement <4 x float> %tmp416, float %add3495, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins3496, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp417 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp418 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add3498 = fadd <4 x float> %tmp418, %tmp417
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add3498, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext3499 = extractelement <4 x float> undef, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add3500 = fadd float %vecext3499, -1.605000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins3501 = insertelement <4 x float> undef, float %add3500, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp419 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext3502 = extractelement <4 x float> %tmp419, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add3503 = fadd float %vecext3502, 0x4058C66660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp420 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add3506 = fadd float undef, 0xC074DB3340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp421 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins3507 = insertelement <4 x float> %tmp421, float %add3506, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins3507, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add3509 = fadd float undef, 0xC066033340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp422 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0x404B333340000000, float 4.680000e+02, float 0x40577999A0000000, float 0xC07D9999A0000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp423 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext3513 = extractelement <4 x float> undef, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add3514 = fadd float %vecext3513, 2.300000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp424 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp425 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext3516 = extractelement <4 x float> %tmp425, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins5414 = insertelement <4 x float> undef, float undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins5414, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp426 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp427 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add5416 = fadd <4 x float> %tmp427, %tmp426
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add5416, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp428 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add5418 = fadd float undef, 0xC07ED999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp429 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins5419 = insertelement <4 x float> %tmp429, float %add5418, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins5624 = insertelement <4 x float> undef, float undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins5624, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0xC07B4999A0000000, float 0x4078B33340000000, float 0xC07674CCC0000000, float 0xC07C533340000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add5626 = fadd <4 x float> undef, undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add5626, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext5627 = extractelement <4 x float> undef, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp430 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins5629 = insertelement <4 x float> %tmp430, float undef, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins5629, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp431 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext5630 = extractelement <4 x float> %tmp431, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add5631 = fadd float %vecext5630, 0x405EECCCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins5632 = insertelement <4 x float> undef, float %add5631, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins5632, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp432 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins5688 = insertelement <4 x float> %tmp432, float undef, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins5688, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp433 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext5689 = extractelement <4 x float> %tmp433, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp434 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins5691 = insertelement <4 x float> %tmp434, float undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins5691, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext5692 = extractelement <4 x float> undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float -4.350000e+02, float 0xC0775CCCC0000000, float 0xC0714999A0000000, float 0xC0661999A0000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp435 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add5696 = fadd <4 x float> undef, %tmp435
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add5696, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add5701 = fadd float undef, 0x4077D4CCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp436 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins5702 = insertelement <4 x float> %tmp436, float %add5701, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins5702, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp437 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp438 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins5705 = insertelement <4 x float> %tmp438, float undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins5705, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp439 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext5706 = extractelement <4 x float> %tmp439, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add5707 = fadd float %vecext5706, 0xC0780B3340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp440 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins5708 = insertelement <4 x float> %tmp440, float %add5707, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins5708, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0x405D666660000000, float 0xC069333340000000, float 0x407B6B3340000000, float 0xC06EB33340000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp441 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp442 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add5710 = fadd <4 x float> %tmp442, %tmp441
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add5710, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp443 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext5711 = extractelement <4 x float> %tmp443, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add5712 = fadd float %vecext5711, 1.850000e+01
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp444 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins5713 = insertelement <4 x float> %tmp444, float %add5712, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins5713, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp445 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp446 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins5716 = insertelement <4 x float> %tmp446, float undef, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp447 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add5724 = fadd <4 x float> %tmp447, undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add5724, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp448 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext5748 = extractelement <4 x float> undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp449 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins5750 = insertelement <4 x float> %tmp449, float undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0x40692999A0000000, float 0xC07C4CCCC0000000, float 0x407D1E6660000000, float 0x407B4199A0000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp450 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add5752 = fadd <4 x float> undef, %tmp450
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add5754 = fadd float undef, 0xC064033340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp451 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins5755 = insertelement <4 x float> %tmp451, float %add5754, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins5755, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp452 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext5756 = extractelement <4 x float> %tmp452, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add5757 = fadd float %vecext5756, 0x40787B3340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp453 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins5758 = insertelement <4 x float> %tmp453, float %add5757, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins5758, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp454 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext5759 = extractelement <4 x float> %tmp454, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp455 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins5761 = insertelement <4 x float> %tmp455, float undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins5761, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp456 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext5762 = extractelement <4 x float> %tmp456, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add5763 = fadd float %vecext5762, 0x40703E6660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp457 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins5764 = insertelement <4 x float> %tmp457, float %add5763, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins5764, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0x407A6B3340000000, float 0x40470CCCC0000000, float 0xC076F4CCC0000000, float 0x40791999A0000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add5766 = fadd <4 x float> undef, undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add5766, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp458 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext5767 = extractelement <4 x float> %tmp458, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add5768 = fadd float %vecext5767, 0x4065533340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp459 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins5769 = insertelement <4 x float> %tmp459, float %add5768, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins5769, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add5771 = fadd float undef, 8.000000e+00
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp460 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins5772 = insertelement <4 x float> %tmp460, float %add5771, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp461 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add5796 = fadd float undef, 0x4058ECCCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins5797 = insertelement <4 x float> undef, float %add5796, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins5797, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp462 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext5798 = extractelement <4 x float> %tmp462, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp463 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins5800 = insertelement <4 x float> %tmp463, float undef, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp464 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext5801 = extractelement <4 x float> %tmp464, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add5802 = fadd float %vecext5801, 0xC072A199A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp465 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins5803 = insertelement <4 x float> %tmp465, float %add5802, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins5803, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp466 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext5804 = extractelement <4 x float> %tmp466, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add5805 = fadd float %vecext5804, 0x40785999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp467 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins5806 = insertelement <4 x float> %tmp467, float %add5805, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins5806, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp468 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp469 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext5809 = extractelement <4 x float> %tmp469, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add5810 = fadd float %vecext5809, 0x407B7B3340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp470 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp471 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext5818 = extractelement <4 x float> undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add5819 = fadd float %vecext5818, 0x4071733340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp472 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins5820 = insertelement <4 x float> %tmp472, float %add5819, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins5820, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0x40514CCCC0000000, float 0x406A7999A0000000, float 0xC078766660000000, float 0xC0522CCCC0000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp473 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp474 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add5822 = fadd <4 x float> %tmp474, %tmp473
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add5822, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp475 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext5823 = extractelement <4 x float> %tmp475, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp476 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins5825 = insertelement <4 x float> %tmp476, float undef, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp477 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext5826 = extractelement <4 x float> %tmp477, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add5827 = fadd float %vecext5826, 0x407F14CCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp478 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins5828 = insertelement <4 x float> %tmp478, float %add5827, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins5828, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp479 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext5829 = extractelement <4 x float> %tmp479, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add5830 = fadd float %vecext5829, 3.350000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp480 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins5831 = insertelement <4 x float> %tmp480, float %add5830, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float -3.370000e+02, float 0xC072DE6660000000, float -2.670000e+02, float 0x4062333340000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp481 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext5837 = extractelement <4 x float> %tmp481, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins5839 = insertelement <4 x float> undef, float undef, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins5839, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp482 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext5840 = extractelement <4 x float> %tmp482, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp483 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins5842 = insertelement <4 x float> %tmp483, float undef, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins5842, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp484 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp485 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins5845 = insertelement <4 x float> %tmp485, float undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins5845, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0xC06EC999A0000000, float 0x406D5999A0000000, float 0x4056F33340000000, float 0xC07E14CCC0000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add5850 = fadd <4 x float> undef, undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add5850, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp486 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add5852 = fadd float undef, 2.985000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp487 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins5853 = insertelement <4 x float> %tmp487, float %add5852, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins5853, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp488 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext5854 = extractelement <4 x float> %tmp488, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add5855 = fadd float %vecext5854, 0xC053F999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp489 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins5856 = insertelement <4 x float> %tmp489, float %add5855, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins5856, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp490 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add5858 = fadd float undef, 0x4071666660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp491 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins5859 = insertelement <4 x float> %tmp491, float %add5858, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins5859, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp492 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext5860 = extractelement <4 x float> %tmp492, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp493 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext5899 = extractelement <4 x float> %tmp493, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add5900 = fadd float %vecext5899, -2.700000e+01
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp494 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins5901 = insertelement <4 x float> %tmp494, float %add5900, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins5901, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add5914 = fadd float undef, 0x40786E6660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins5918 = insertelement <4 x float> undef, float undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins5918, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0x406F266660000000, float 7.900000e+01, float -4.695000e+02, float -4.880000e+02>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add5920 = fadd <4 x float> undef, undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add5920, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add5934 = fadd <4 x float> undef, undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext5935 = extractelement <4 x float> undef, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add5936 = fadd float %vecext5935, 0xC056B999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp495 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp496 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext5994 = extractelement <4 x float> undef, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add5995 = fadd float %vecext5994, 0x4051666660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins5996 = insertelement <4 x float> undef, float %add5995, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins5996, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp497 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext5997 = extractelement <4 x float> %tmp497, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp498 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6000 = extractelement <4 x float> %tmp498, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6001 = fadd float %vecext6000, -7.600000e+01
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp499 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6002 = insertelement <4 x float> %tmp499, float %add6001, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6002, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0xC07EA199A0000000, float 0x407DC33340000000, float 0xC0753199A0000000, float -3.895000e+02>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp500 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6004 = fadd <4 x float> undef, %tmp500
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add6004, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp501 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6005 = extractelement <4 x float> %tmp501, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp502 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6007 = insertelement <4 x float> %tmp502, float undef, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp503 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6008 = extractelement <4 x float> %tmp503, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp504 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6024 = insertelement <4 x float> %tmp504, float undef, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6024, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp505 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6025 = extractelement <4 x float> %tmp505, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6026 = fadd float %vecext6025, 3.700000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp506 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6027 = insertelement <4 x float> %tmp506, float %add6026, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6027, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6028 = extractelement <4 x float> undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6029 = fadd float %vecext6028, 0x4071666660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp507 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6030 = insertelement <4 x float> %tmp507, float %add6029, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6030, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0xC0527999A0000000, float 0xC06AD999A0000000, float 0x3FF6666660000000, float 0xC03F666660000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp508 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp509 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp510 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6033 = extractelement <4 x float> %tmp510, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp511 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6036 = extractelement <4 x float> undef, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6037 = fadd float %vecext6036, 0xC075CB3340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6038 = insertelement <4 x float> undef, float %add6037, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6038, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp512 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6040 = fadd float undef, 0x4071ECCCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp513 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6041 = insertelement <4 x float> %tmp513, float %add6040, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6041, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp514 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6042 = extractelement <4 x float> %tmp514, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6043 = fadd float %vecext6042, 0xC07DD33340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp515 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6044 = insertelement <4 x float> %tmp515, float %add6043, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6044, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0xC065FCCCC0000000, float 0x40767CCCC0000000, float 0x4079D4CCC0000000, float 0xC07314CCC0000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp516 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp517 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6046 = fadd <4 x float> %tmp517, %tmp516
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add6046, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6047 = extractelement <4 x float> undef, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp518 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6049 = insertelement <4 x float> %tmp518, float undef, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6049, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp519 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6050 = extractelement <4 x float> %tmp519, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6051 = fadd float %vecext6050, 0x407E4E6660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6055 = insertelement <4 x float> undef, float undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6056 = extractelement <4 x float> undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp520 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6061 = extractelement <4 x float> undef, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp521 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp522 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6067 = extractelement <4 x float> %tmp522, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6068 = fadd float %vecext6067, 0x40768E6660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6070 = extractelement <4 x float> undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6071 = fadd float %vecext6070, 0xC07C6CCCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6072 = insertelement <4 x float> undef, float %add6071, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6072, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0x40546CCCC0000000, float 0x4067D66660000000, float 0xC060E33340000000, float 0x4061533340000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp523 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp524 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6074 = fadd <4 x float> %tmp524, %tmp523
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add6074, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp525 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6075 = extractelement <4 x float> %tmp525, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6076 = fadd float %vecext6075, 0x405D733340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp526 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6077 = insertelement <4 x float> %tmp526, float %add6076, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6077, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp527 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6079 = fadd float undef, 0xC07E9B3340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp528 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp529 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6082 = fadd float undef, 0x407DCE6660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6083 = insertelement <4 x float> undef, float %add6082, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6083, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp530 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6084 = extractelement <4 x float> %tmp530, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6085 = fadd float %vecext6084, 0xC061A33340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6086 = insertelement <4 x float> undef, float %add6085, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6086, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0x4055C66660000000, float 0x40735199A0000000, float 0xC0713199A0000000, float 0x40729B3340000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp531 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp532 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6088 = fadd <4 x float> %tmp532, %tmp531
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add6088, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp533 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6089 = extractelement <4 x float> %tmp533, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6107 = fadd float undef, 0xC06A166660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp534 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6108 = insertelement <4 x float> %tmp534, float %add6107, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6108, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp535 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6109 = extractelement <4 x float> %tmp535, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6110 = fadd float %vecext6109, 0x4070FB3340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp536 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp537 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6112 = extractelement <4 x float> %tmp537, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6113 = fadd float %vecext6112, 0xC04AF33340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp538 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp539 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6117 = extractelement <4 x float> %tmp539, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6118 = fadd float %vecext6117, 0x407AB33340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp540 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6119 = insertelement <4 x float> %tmp540, float %add6118, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6119, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp541 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6120 = extractelement <4 x float> %tmp541, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6121 = fadd float %vecext6120, 0x405AE66660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp542 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6122 = insertelement <4 x float> %tmp542, float %add6121, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6122, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6123 = extractelement <4 x float> undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6124 = fadd float %vecext6123, -4.385000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp543 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6126 = extractelement <4 x float> undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp544 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6128 = insertelement <4 x float> %tmp544, float undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6128, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float -2.980000e+02, float 0xC06F0CCCC0000000, float 0xC054A66660000000, float 0xC040CCCCC0000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp545 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp546 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6130 = fadd <4 x float> %tmp546, %tmp545
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp547 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6131 = extractelement <4 x float> %tmp547, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6132 = fadd float %vecext6131, 0x407BDE6660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6133 = insertelement <4 x float> undef, float %add6132, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6133, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6134 = extractelement <4 x float> undef, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6135 = fadd float %vecext6134, 0xC06B7999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp548 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6136 = insertelement <4 x float> %tmp548, float %add6135, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6137 = extractelement <4 x float> undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6138 = fadd float %vecext6137, 0x40752199A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp549 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6172 = fadd <4 x float> undef, %tmp549
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp550 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp551 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6178 = insertelement <4 x float> %tmp551, float undef, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6178, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp552 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6179 = extractelement <4 x float> %tmp552, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6180 = fadd float %vecext6179, -3.905000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp553 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6181 = insertelement <4 x float> %tmp553, float %add6180, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp554 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6182 = extractelement <4 x float> %tmp554, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6183 = fadd float %vecext6182, 1.515000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp555 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6184 = insertelement <4 x float> %tmp555, float %add6183, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6184, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp556 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6189 = insertelement <4 x float> undef, float undef, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6189, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp557 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6190 = extractelement <4 x float> %tmp557, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6191 = fadd float %vecext6190, 0xC07BD33340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp558 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6192 = insertelement <4 x float> %tmp558, float %add6191, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6192, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp559 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp560 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6196 = extractelement <4 x float> undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6197 = fadd float %vecext6196, -4.070000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp561 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6198 = insertelement <4 x float> %tmp561, float %add6197, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0x407904CCC0000000, float 0x406A833340000000, float 4.895000e+02, float 0x40648999A0000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp562 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp563 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6200 = fadd <4 x float> %tmp563, %tmp562
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add6200, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp564 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6201 = extractelement <4 x float> %tmp564, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp565 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6203 = insertelement <4 x float> %tmp565, float undef, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp566 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6204 = extractelement <4 x float> %tmp566, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6205 = fadd float %vecext6204, 1.740000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp567 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6206 = insertelement <4 x float> %tmp567, float %add6205, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp568 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6209 = insertelement <4 x float> %tmp568, float undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6209, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp569 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6210 = extractelement <4 x float> %tmp569, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp570 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6219 = fadd float undef, 0xC0596CCCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp571 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6220 = insertelement <4 x float> %tmp571, float %add6219, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6224 = extractelement <4 x float> undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6225 = fadd float %vecext6224, 0xC074533340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp572 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6228 = fadd <4 x float> undef, undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add6228, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6229 = extractelement <4 x float> undef, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6230 = fadd float %vecext6229, 1.695000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp573 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6231 = insertelement <4 x float> %tmp573, float %add6230, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6231, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp574 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6232 = extractelement <4 x float> %tmp574, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6233 = fadd float %vecext6232, 0x4079C33340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp575 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6234 = insertelement <4 x float> %tmp575, float %add6233, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6234, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6235 = extractelement <4 x float> undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6236 = fadd float %vecext6235, 0xC07D8199A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6237 = insertelement <4 x float> undef, float %add6236, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6237, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp576 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6245 = insertelement <4 x float> undef, float undef, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6245, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp577 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6246 = extractelement <4 x float> %tmp577, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6247 = fadd float %vecext6246, 0x40631999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp578 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6251 = insertelement <4 x float> undef, float undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp579 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6253 = fadd float undef, 0xC0692999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6254 = insertelement <4 x float> undef, float %add6253, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6254, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 4.600000e+02, float 0xC0777B3340000000, float 0x40351999A0000000, float 0xC06E433340000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp580 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp581 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6256 = fadd <4 x float> %tmp581, %tmp580
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add6256, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp582 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6257 = extractelement <4 x float> %tmp582, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6258 = fadd float %vecext6257, 4.670000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp583 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6259 = insertelement <4 x float> %tmp583, float %add6258, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6259, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp584 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6260 = extractelement <4 x float> %tmp584, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6261 = fadd float %vecext6260, 0xC05F733340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp585 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6262 = insertelement <4 x float> %tmp585, float %add6261, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6262, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp586 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6263 = extractelement <4 x float> %tmp586, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp587 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6265 = insertelement <4 x float> %tmp587, float undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6265, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp588 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6266 = extractelement <4 x float> %tmp588, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6267 = fadd float %vecext6266, 0x407174CCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp589 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6268 = insertelement <4 x float> %tmp589, float %add6267, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6268, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float -3.130000e+02, float 0xC079733340000000, float -4.660000e+02, float 0xC064E66660000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp590 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp591 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6270 = fadd <4 x float> %tmp591, %tmp590
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add6270, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp592 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6271 = extractelement <4 x float> %tmp592, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6272 = fadd float %vecext6271, 1.765000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp593 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6273 = insertelement <4 x float> %tmp593, float %add6272, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6273, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp594 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6274 = extractelement <4 x float> %tmp594, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6275 = fadd float %vecext6274, 0x402C666660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp595 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6276 = insertelement <4 x float> %tmp595, float %add6275, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6276, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp596 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6277 = extractelement <4 x float> %tmp596, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6278 = fadd float %vecext6277, -8.450000e+01
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp597 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6279 = insertelement <4 x float> %tmp597, float %add6278, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6279, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp598 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6280 = extractelement <4 x float> %tmp598, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6281 = fadd float %vecext6280, 0xC07A133340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6282 = insertelement <4 x float> undef, float %add6281, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6282, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0x4067ECCCC0000000, float 0xC040CCCCC0000000, float 0xC0762E6660000000, float -4.750000e+02>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6284 = fadd <4 x float> undef, undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6285 = extractelement <4 x float> undef, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6289 = fadd float undef, 0xC0738999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp599 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6293 = insertelement <4 x float> %tmp599, float undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6293, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp600 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6294 = extractelement <4 x float> %tmp600, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6295 = fadd float %vecext6294, 0xC01CCCCCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6296 = insertelement <4 x float> undef, float %add6295, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6296, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0x40704199A0000000, float 0x40753CCCC0000000, float 0xC07E2199A0000000, float 0xC068833340000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp601 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6298 = fadd <4 x float> undef, %tmp601
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add6298, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp602 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6299 = extractelement <4 x float> %tmp602, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6300 = fadd float %vecext6299, 0x4074B33340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp603 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6301 = insertelement <4 x float> %tmp603, float %add6300, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6301, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp604 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6302 = extractelement <4 x float> %tmp604, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6303 = fadd float %vecext6302, 0xC05B333340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp605 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6304 = insertelement <4 x float> %tmp605, float %add6303, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6304, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp606 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6305 = extractelement <4 x float> %tmp606, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6306 = fadd float %vecext6305, 0x4077E999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6307 = insertelement <4 x float> undef, float %add6306, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6307, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp607 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6308 = extractelement <4 x float> %tmp607, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6309 = fadd float %vecext6308, 0x40707E6660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp608 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6310 = insertelement <4 x float> %tmp608, float %add6309, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6310, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0x407A233340000000, float 0x406DA33340000000, float 3.725000e+02, float 0x40761199A0000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp609 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp610 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6312 = fadd <4 x float> %tmp610, %tmp609
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add6312, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp611 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6313 = extractelement <4 x float> %tmp611, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6314 = fadd float %vecext6313, 0xC07CF33340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp612 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6315 = insertelement <4 x float> %tmp612, float %add6314, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp613 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6655 = extractelement <4 x float> %tmp613, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6656 = fadd float %vecext6655, 2.185000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp614 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6657 = insertelement <4 x float> %tmp614, float %add6656, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6657, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6660 = insertelement <4 x float> undef, float undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6660, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0xC064E33340000000, float 0xC064833340000000, float 0xC0673CCCC0000000, float 0xC074266660000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp615 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6663 = extractelement <4 x float> undef, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6664 = fadd float %vecext6663, 0xC05B7999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp616 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6665 = insertelement <4 x float> %tmp616, float %add6664, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp617 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6669 = extractelement <4 x float> %tmp617, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp618 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0xC07CC4CCC0000000, float 0x404EE66660000000, float 0xC0754CCCC0000000, float 0xC0744B3340000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp619 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6676 = fadd <4 x float> %tmp619, undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add6676, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp620 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6677 = extractelement <4 x float> %tmp620, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6678 = fadd float %vecext6677, 0x4077F4CCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp621 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6679 = insertelement <4 x float> %tmp621, float %add6678, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6680 = extractelement <4 x float> undef, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6681 = fadd float %vecext6680, 0x4061766660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp622 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp623 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6683 = extractelement <4 x float> %tmp623, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6684 = fadd float %vecext6683, 0x40718999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp624 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6685 = insertelement <4 x float> %tmp624, float %add6684, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6685, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp625 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6686 = extractelement <4 x float> %tmp625, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6687 = fadd float %vecext6686, 0x4076D66660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp626 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6688 = insertelement <4 x float> %tmp626, float %add6687, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6688, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 7.500000e+00, float 0x4077E33340000000, float 0xC0596CCCC0000000, float 0xC07D4E6660000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp627 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6690 = fadd <4 x float> undef, %tmp627
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add6690, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp628 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6691 = extractelement <4 x float> %tmp628, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6692 = fadd float %vecext6691, 3.250000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp629 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6693 = insertelement <4 x float> %tmp629, float %add6692, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6693, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp630 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6694 = extractelement <4 x float> %tmp630, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6695 = fadd float %vecext6694, 0x407DF999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp631 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6696 = insertelement <4 x float> %tmp631, float %add6695, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6696, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp632 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6697 = extractelement <4 x float> %tmp632, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6698 = fadd float %vecext6697, 0xC075FE6660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp633 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6699 = insertelement <4 x float> %tmp633, float %add6698, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6699, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp634 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6700 = extractelement <4 x float> %tmp634, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6701 = fadd float %vecext6700, 0xC07BCE6660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp635 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6702 = insertelement <4 x float> %tmp635, float %add6701, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6702, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0x40772CCCC0000000, float 0xC0625CCCC0000000, float 6.200000e+01, float 0xC06ADCCCC0000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp636 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp637 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6707 = insertelement <4 x float> undef, float undef, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6707, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp638 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6708 = extractelement <4 x float> %tmp638, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp639 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp640 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6714 = extractelement <4 x float> %tmp640, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6715 = fadd float %vecext6714, 0xC0537999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp641 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6716 = insertelement <4 x float> %tmp641, float %add6715, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6719 = extractelement <4 x float> undef, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6720 = fadd float %vecext6719, 2.870000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp642 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6721 = insertelement <4 x float> %tmp642, float %add6720, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp643 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6722 = extractelement <4 x float> %tmp643, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6723 = fadd float %vecext6722, 0xC07704CCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp644 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6724 = insertelement <4 x float> %tmp644, float %add6723, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp645 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6726 = fadd float undef, 0x4059B999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp646 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6727 = insertelement <4 x float> %tmp646, float %add6726, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6727, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6728 = extractelement <4 x float> undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6729 = fadd float %vecext6728, 0xC073466660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0xC0309999A0000000, float -2.715000e+02, float 1.620000e+02, float 0x40674CCCC0000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp647 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp648 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6732 = fadd <4 x float> %tmp648, %tmp647
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add6732, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp649 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6733 = extractelement <4 x float> %tmp649, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6734 = fadd float %vecext6733, 0x4040733340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp650 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6735 = insertelement <4 x float> %tmp650, float %add6734, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6735, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp651 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6736 = extractelement <4 x float> %tmp651, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6737 = fadd float %vecext6736, 0xC07B74CCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp652 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6738 = insertelement <4 x float> %tmp652, float %add6737, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6738, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp653 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6739 = extractelement <4 x float> %tmp653, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6740 = fadd float %vecext6739, 0x40699CCCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp654 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6741 = insertelement <4 x float> %tmp654, float %add6740, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6741, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp655 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6742 = extractelement <4 x float> %tmp655, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6743 = fadd float %vecext6742, 0x4078533340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp656 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6744 = insertelement <4 x float> %tmp656, float %add6743, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6744, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp657 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp658 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6746 = fadd <4 x float> %tmp658, %tmp657
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add6746, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp659 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6749 = insertelement <4 x float> undef, float undef, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6749, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp660 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6751 = fadd float undef, 0x4075DE6660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6752 = insertelement <4 x float> undef, float %add6751, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6752, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp661 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6753 = extractelement <4 x float> %tmp661, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6754 = fadd float %vecext6753, 0xC008CCCCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6755 = insertelement <4 x float> undef, float %add6754, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6755, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp662 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6756 = extractelement <4 x float> %tmp662, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6757 = fadd float %vecext6756, 0x406CA999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp663 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6758 = insertelement <4 x float> %tmp663, float %add6757, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6758, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0x403D1999A0000000, float 0xC05F533340000000, float 3.945000e+02, float 3.950000e+01>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp664 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6760 = fadd <4 x float> undef, %tmp664
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add6760, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp665 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6761 = extractelement <4 x float> %tmp665, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6762 = fadd float %vecext6761, 2.860000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6763 = insertelement <4 x float> undef, float %add6762, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp666 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0xC079BE6660000000, float 4.930000e+02, float 0x406CC33340000000, float 0xC062E999A0000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp667 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6774 = fadd <4 x float> undef, %tmp667
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp668 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6775 = extractelement <4 x float> %tmp668, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6776 = fadd float %vecext6775, 0x407B8199A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp669 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6777 = insertelement <4 x float> %tmp669, float %add6776, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6777, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp670 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6778 = extractelement <4 x float> %tmp670, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6779 = fadd float %vecext6778, 0x401C666660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp671 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6784 = extractelement <4 x float> %tmp671, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6875 = insertelement <4 x float> undef, float undef, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6875, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp672 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6876 = extractelement <4 x float> %tmp672, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6877 = fadd float %vecext6876, 0x4073A66660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6878 = insertelement <4 x float> undef, float %add6877, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6878, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6888 = fadd float undef, 0x4057CCCCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp673 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6889 = insertelement <4 x float> %tmp673, float %add6888, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6889, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp674 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6890 = extractelement <4 x float> %tmp674, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6891 = fadd float %vecext6890, -4.430000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp675 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6892 = insertelement <4 x float> %tmp675, float %add6891, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6892, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp676 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6893 = extractelement <4 x float> %tmp676, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6894 = fadd float %vecext6893, -3.280000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp677 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6895 = insertelement <4 x float> %tmp677, float %add6894, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6895, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp678 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp679 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp680 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6900 = fadd <4 x float> %tmp680, %tmp679
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add6900, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp681 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6901 = extractelement <4 x float> %tmp681, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6902 = fadd float %vecext6901, 0x4079DCCCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp682 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6903 = insertelement <4 x float> %tmp682, float %add6902, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6903, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6905 = fadd float undef, 0x4031B33340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp683 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6906 = insertelement <4 x float> %tmp683, float %add6905, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp684 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6912 = insertelement <4 x float> %tmp684, float undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 3.315000e+02, float 0xC066C999A0000000, float 0xC061F33340000000, float 0x4071166660000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp685 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp686 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6914 = fadd <4 x float> %tmp686, %tmp685
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add6914, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6915 = extractelement <4 x float> undef, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6920 = insertelement <4 x float> undef, float undef, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6920, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6921 = extractelement <4 x float> undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6922 = fadd float %vecext6921, 0xC064066660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp687 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6926 = insertelement <4 x float> %tmp687, float undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6926, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0xC03C4CCCC0000000, float 0xC07E5199A0000000, float -8.250000e+01, float 0xC043B33340000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp688 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp689 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6928 = fadd <4 x float> %tmp689, %tmp688
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add6928, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6930 = fadd float undef, -4.590000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6931 = insertelement <4 x float> undef, float %add6930, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6931, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp690 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6932 = extractelement <4 x float> %tmp690, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6933 = fadd float %vecext6932, 0xC063F999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp691 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp692 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6935 = extractelement <4 x float> %tmp692, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6936 = fadd float %vecext6935, -3.335000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp693 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6937 = insertelement <4 x float> %tmp693, float %add6936, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp694 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6938 = extractelement <4 x float> %tmp694, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6939 = fadd float %vecext6938, 0x405F3999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6942 = fadd <4 x float> undef, undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6943 = extractelement <4 x float> undef, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6944 = fadd float %vecext6943, 0x40530CCCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp695 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6950 = fadd float undef, 0xC078F33340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp696 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6951 = insertelement <4 x float> %tmp696, float %add6950, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6951, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp697 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6952 = extractelement <4 x float> %tmp697, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6953 = fadd float %vecext6952, 0xC06E5999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp698 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6954 = insertelement <4 x float> %tmp698, float %add6953, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6954, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp699 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp700 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6956 = fadd <4 x float> %tmp700, %tmp699
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add6956, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp701 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6957 = extractelement <4 x float> %tmp701, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6958 = fadd float %vecext6957, 0xC077633340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp702 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6959 = insertelement <4 x float> %tmp702, float %add6958, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6959, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp703 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6963 = extractelement <4 x float> %tmp703, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6964 = fadd float %vecext6963, 0x4068666660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp704 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6965 = insertelement <4 x float> %tmp704, float %add6964, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6965, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6975 = fadd float undef, 0x406AF33340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp705 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6976 = insertelement <4 x float> %tmp705, float %add6975, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6976, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp706 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp707 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6984 = fadd <4 x float> %tmp707, undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add6984, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp708 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6985 = extractelement <4 x float> %tmp708, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6986 = fadd float %vecext6985, 0xC05E266660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp709 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6987 = insertelement <4 x float> %tmp709, float %add6986, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6987, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp710 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6988 = extractelement <4 x float> %tmp710, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6989 = fadd float %vecext6988, 0x40706E6660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp711 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins6996 = insertelement <4 x float> %tmp711, float undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins6996, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0x4077A4CCC0000000, float 0xC0757199A0000000, float 0xC072F4CCC0000000, float 0xC071DCCCC0000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp712 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp713 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add6998 = fadd <4 x float> %tmp713, %tmp712
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add6998, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp714 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext6999 = extractelement <4 x float> %tmp714, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add7000 = fadd float %vecext6999, 0x4076233340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp715 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins7001 = insertelement <4 x float> %tmp715, float %add7000, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins7001, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp716 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext7002 = extractelement <4 x float> %tmp716, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add7003 = fadd float %vecext7002, 0x403BCCCCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp717 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins7004 = insertelement <4 x float> %tmp717, float %add7003, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp718 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add7140 = fadd float undef, 0x403D333340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins7141 = insertelement <4 x float> undef, float %add7140, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext7142 = extractelement <4 x float> undef, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add7143 = fadd float %vecext7142, 0xC058F999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins7144 = insertelement <4 x float> undef, float %add7143, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp719 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext7148 = extractelement <4 x float> %tmp719, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add7149 = fadd float %vecext7148, 0x4075333340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp720 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins7150 = insertelement <4 x float> %tmp720, float %add7149, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins7150, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 1.700000e+02, float 0xC077B4CCC0000000, float 0x40625999A0000000, float 0x406C166660000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp721 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add7152 = fadd <4 x float> %tmp721, undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add7152, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext7156 = extractelement <4 x float> undef, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add7157 = fadd float %vecext7156, 0xC05F533340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp722 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins7158 = insertelement <4 x float> %tmp722, float %add7157, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins7158, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp723 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext7159 = extractelement <4 x float> %tmp723, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add7160 = fadd float %vecext7159, 0x407A5999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp724 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins7161 = insertelement <4 x float> %tmp724, float %add7160, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins7161, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add7168 = fadd float undef, 0xC072F199A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp725 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext7170 = extractelement <4 x float> undef, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add7171 = fadd float %vecext7170, 0x406AACCCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins7172 = insertelement <4 x float> undef, float %add7171, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins7172, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext7173 = extractelement <4 x float> undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp726 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext7419 = extractelement <4 x float> %tmp726, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add7420 = fadd float %vecext7419, 0x404EA66660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins7421 = insertelement <4 x float> undef, float %add7420, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins7421, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp727 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext7422 = extractelement <4 x float> %tmp727, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add7423 = fadd float %vecext7422, 4.800000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp728 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins7424 = insertelement <4 x float> %tmp728, float %add7423, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins7424, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp729 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext7425 = extractelement <4 x float> %tmp729, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add7426 = fadd float %vecext7425, 0xC072C999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp730 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins7427 = insertelement <4 x float> %tmp730, float %add7426, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins7427, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext7428 = extractelement <4 x float> undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp731 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext7568 = extractelement <4 x float> %tmp731, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add7569 = fadd float %vecext7568, 1.090000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp732 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins7570 = insertelement <4 x float> %tmp732, float %add7569, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins7570, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0x40745199A0000000, float 0xC0411999A0000000, float -5.650000e+01, float -4.005000e+02>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp733 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp734 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add7572 = fadd <4 x float> %tmp734, %tmp733
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add7572, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext7573 = extractelement <4 x float> undef, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add7574 = fadd float %vecext7573, -3.920000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp735 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins7575 = insertelement <4 x float> %tmp735, float %add7574, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins7575, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp736 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add7577 = fadd float undef, 0xC051666660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp737 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp738 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext7579 = extractelement <4 x float> %tmp738, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add7580 = fadd float %vecext7579, 0x407E9199A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins7581 = insertelement <4 x float> undef, float %add7580, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins7581, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp739 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext7582 = extractelement <4 x float> %tmp739, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add7583 = fadd float %vecext7582, 2.760000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp740 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins7584 = insertelement <4 x float> %tmp740, float %add7583, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0xC057533340000000, float 0x4060A33340000000, float 0x40791E6660000000, float 2.455000e+02>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp741 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp742 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add7586 = fadd <4 x float> %tmp742, %tmp741
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add7586, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp743 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext7587 = extractelement <4 x float> %tmp743, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add7588 = fadd float %vecext7587, 6.100000e+01
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp744 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp745 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext7590 = extractelement <4 x float> %tmp745, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add7591 = fadd float %vecext7590, -3.935000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp746 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins7592 = insertelement <4 x float> %tmp746, float %add7591, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins7592, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp747 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext7593 = extractelement <4 x float> %tmp747, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins7595 = insertelement <4 x float> undef, float undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins7595, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp748 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext7596 = extractelement <4 x float> %tmp748, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add7597 = fadd float %vecext7596, 0x407E666660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0x406A766660000000, float 0xBFC99999A0000000, float 0xC0751B3340000000, float -4.075000e+02>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp749 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add7616 = fadd float undef, 0xC04DE66660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp750 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins7617 = insertelement <4 x float> %tmp750, float %add7616, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins7617, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp751 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext7618 = extractelement <4 x float> %tmp751, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add7619 = fadd float %vecext7618, 6.050000e+01
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp752 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins7620 = insertelement <4 x float> %tmp752, float %add7619, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins7620, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp753 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add7622 = fadd float undef, 0xC054B999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp754 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins7626 = insertelement <4 x float> undef, float undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins7626, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp755 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp756 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add7628 = fadd <4 x float> %tmp756, %tmp755
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add7628, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp757 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext7629 = extractelement <4 x float> %tmp757, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add7630 = fadd float %vecext7629, 0xC05E2CCCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp758 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins7631 = insertelement <4 x float> %tmp758, float %add7630, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add7639 = fadd float undef, 0x407C5999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp759 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins7640 = insertelement <4 x float> %tmp759, float %add7639, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0x406AA66660000000, float 0x4067C66660000000, float 0xC054866660000000, float -2.400000e+01>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp760 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add7642 = fadd <4 x float> %tmp760, undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp761 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add7644 = fadd float undef, 0xC0758999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp762 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext7646 = extractelement <4 x float> %tmp762, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add7647 = fadd float %vecext7646, 0xC07A3B3340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp763 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins7648 = insertelement <4 x float> %tmp763, float %add7647, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins7648, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp764 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext7649 = extractelement <4 x float> %tmp764, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add7650 = fadd float %vecext7649, 0x40760CCCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp765 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins7651 = insertelement <4 x float> %tmp765, float %add7650, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins7651, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp766 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext7652 = extractelement <4 x float> %tmp766, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add7653 = fadd float %vecext7652, 0x40620CCCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp767 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins7654 = insertelement <4 x float> %tmp767, float %add7653, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins7654, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp768 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp769 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add7656 = fadd <4 x float> %tmp769, %tmp768
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add7656, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp770 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext7657 = extractelement <4 x float> %tmp770, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add7658 = fadd float %vecext7657, 0xC06EF999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp771 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins7659 = insertelement <4 x float> %tmp771, float %add7658, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins7659, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp772 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext7660 = extractelement <4 x float> %tmp772, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add7661 = fadd float %vecext7660, 0x404B9999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp773 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins7662 = insertelement <4 x float> %tmp773, float %add7661, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins7662, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp774 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext7663 = extractelement <4 x float> %tmp774, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add7664 = fadd float %vecext7663, 0x4074B66660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp775 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins7665 = insertelement <4 x float> %tmp775, float %add7664, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins7665, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp776 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext7666 = extractelement <4 x float> %tmp776, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add7667 = fadd float %vecext7666, 0x4074166660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins7668 = insertelement <4 x float> undef, float %add7667, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins7668, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp777 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp778 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add7670 = fadd <4 x float> %tmp778, %tmp777
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp779 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext7671 = extractelement <4 x float> %tmp779, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add7672 = fadd float %vecext7671, 0x406F166660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add7726 = fadd <4 x float> undef, undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp780 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext7727 = extractelement <4 x float> %tmp780, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp781 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp782 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add7731 = fadd float undef, 1.900000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp783 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins7732 = insertelement <4 x float> %tmp783, float %add7731, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins7732, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp784 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins7735 = insertelement <4 x float> %tmp784, float undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins7735, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp785 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext7736 = extractelement <4 x float> %tmp785, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add7737 = fadd float %vecext7736, 0xC06AF66660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins7850 = insertelement <4 x float> undef, float undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins7850, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0x4062A33340000000, float 2.290000e+02, float 0x40509999A0000000, float 0xC078BE6660000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp786 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp787 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add7852 = fadd <4 x float> %tmp787, %tmp786
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add7852, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp788 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9396 = extractelement <4 x float> undef, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9397 = fadd float %vecext9396, 0xC074533340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp789 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9398 = insertelement <4 x float> %tmp789, float %add9397, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9398, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9399 = extractelement <4 x float> undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp790 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9401 = insertelement <4 x float> %tmp790, float undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp791 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9402 = extractelement <4 x float> %tmp791, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9403 = fadd float %vecext9402, 0xC03E4CCCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp792 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9404 = insertelement <4 x float> %tmp792, float %add9403, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9404, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp793 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp794 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9407 = extractelement <4 x float> %tmp794, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9408 = fadd float %vecext9407, 0x407B2999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp795 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9410 = extractelement <4 x float> %tmp795, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9411 = fadd float %vecext9410, 0x40726E6660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp796 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp797 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9413 = extractelement <4 x float> %tmp797, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9414 = fadd float %vecext9413, 0xC057ECCCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp798 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9415 = insertelement <4 x float> %tmp798, float %add9414, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9415, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp799 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9416 = extractelement <4 x float> %tmp799, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9417 = fadd float %vecext9416, 0x406B0CCCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp800 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9418 = insertelement <4 x float> %tmp800, float %add9417, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9418, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 3.555000e+02, float 0xC062E33340000000, float 0x4065C66660000000, float -3.645000e+02>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp801 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp802 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9420 = fadd <4 x float> %tmp802, %tmp801
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add9420, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp803 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9421 = extractelement <4 x float> %tmp803, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp804 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9423 = insertelement <4 x float> %tmp804, float undef, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9423, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp805 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9424 = extractelement <4 x float> %tmp805, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9425 = fadd float %vecext9424, 0x4079C199A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp806 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9426 = insertelement <4 x float> %tmp806, float %add9425, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9426, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp807 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9428 = fadd float undef, 0xC065466660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp808 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9429 = insertelement <4 x float> %tmp808, float %add9428, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9429, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp809 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9430 = extractelement <4 x float> %tmp809, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9431 = fadd float %vecext9430, 0xC0742CCCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp810 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9432 = insertelement <4 x float> %tmp810, float %add9431, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0xC07C7E6660000000, float 1.205000e+02, float 0x4050D999A0000000, float 0xC06B233340000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp811 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp812 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9434 = fadd <4 x float> %tmp812, %tmp811
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9436 = fadd float undef, -3.185000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp813 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9437 = insertelement <4 x float> %tmp813, float %add9436, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp814 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp815 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9441 = extractelement <4 x float> %tmp815, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9442 = fadd float %vecext9441, 0xC079CE6660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp816 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9443 = insertelement <4 x float> %tmp816, float %add9442, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9443, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp817 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9444 = extractelement <4 x float> %tmp817, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9445 = fadd float %vecext9444, 0xC06F533340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp818 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9446 = insertelement <4 x float> %tmp818, float %add9445, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9446, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp819 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp820 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9448 = fadd <4 x float> %tmp820, %tmp819
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add9448, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9450 = fadd float undef, 0xC0718199A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp821 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9451 = insertelement <4 x float> %tmp821, float %add9450, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9451, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp822 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp823 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9454 = insertelement <4 x float> %tmp823, float undef, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9454, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp824 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9455 = extractelement <4 x float> %tmp824, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9456 = fadd float %vecext9455, -3.380000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp825 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9457 = insertelement <4 x float> %tmp825, float %add9456, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9457, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9458 = extractelement <4 x float> undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp826 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9460 = insertelement <4 x float> %tmp826, float undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9460, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0x407B5E6660000000, float 0x40648999A0000000, float 0xC06B966660000000, float 0x40341999A0000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp827 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9462 = fadd <4 x float> %tmp827, undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add9462, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp828 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9463 = extractelement <4 x float> %tmp828, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp829 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9465 = insertelement <4 x float> %tmp829, float undef, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9467 = fadd float undef, 0x405D666660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp830 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9468 = insertelement <4 x float> %tmp830, float %add9467, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9468, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp831 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9470 = fadd float undef, 0x4077033340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp832 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9472 = extractelement <4 x float> undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9473 = fadd float %vecext9472, 0x402DCCCCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp833 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9474 = insertelement <4 x float> %tmp833, float %add9473, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9474, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0x404F733340000000, float 0x407AB4CCC0000000, float 0x40605999A0000000, float 0xC03E4CCCC0000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp834 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp835 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9476 = fadd <4 x float> %tmp835, %tmp834
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add9476, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp836 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9477 = extractelement <4 x float> %tmp836, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9478 = fadd float %vecext9477, 0xC07F266660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp837 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9479 = insertelement <4 x float> %tmp837, float %add9478, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9479, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp838 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9481 = fadd float undef, 0x407BE33340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp839 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9482 = insertelement <4 x float> %tmp839, float %add9481, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9482, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9483 = extractelement <4 x float> undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9484 = fadd float %vecext9483, 0xC073E999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp840 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9485 = insertelement <4 x float> %tmp840, float %add9484, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9485, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp841 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9486 = extractelement <4 x float> %tmp841, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9487 = fadd float %vecext9486, 0x4076E66660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp842 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0xC076B999A0000000, float 0xC0706CCCC0000000, float 0x407904CCC0000000, float 0x407EE199A0000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp843 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp844 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9491 = extractelement <4 x float> %tmp844, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9492 = fadd float %vecext9491, 0x407C166660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9495 = fadd float undef, 0x407DBB3340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp845 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9496 = insertelement <4 x float> %tmp845, float %add9495, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9496, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp846 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9497 = extractelement <4 x float> %tmp846, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9498 = fadd float %vecext9497, 0x4042CCCCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp847 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9499 = insertelement <4 x float> %tmp847, float %add9498, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9499, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp848 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9501 = fadd float undef, 0x407D5CCCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp849 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9502 = insertelement <4 x float> %tmp849, float %add9501, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9502, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp850 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9504 = fadd <4 x float> %tmp850, undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add9504, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp851 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9506 = fadd float undef, 0x4076EE6660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp852 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9507 = insertelement <4 x float> %tmp852, float %add9506, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9507, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp853 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9509 = fadd float undef, 0xC0535999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp854 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp855 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9511 = extractelement <4 x float> %tmp855, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9512 = fadd float %vecext9511, 0xC076766660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp856 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9513 = insertelement <4 x float> %tmp856, float %add9512, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9513, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp857 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9514 = extractelement <4 x float> %tmp857, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp858 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9516 = insertelement <4 x float> %tmp858, float undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9516, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0x407254CCC0000000, float 0x407844CCC0000000, float 0xC04D9999A0000000, float 0xC0550CCCC0000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp859 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp860 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9518 = fadd <4 x float> %tmp860, %tmp859
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp861 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp862 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9521 = insertelement <4 x float> %tmp862, float undef, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9521, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp863 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9522 = extractelement <4 x float> %tmp863, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9523 = fadd float %vecext9522, 0x4029333340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp864 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9524 = insertelement <4 x float> %tmp864, float %add9523, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9524, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp865 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9526 = fadd float undef, 0x4072833340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp866 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9527 = insertelement <4 x float> %tmp866, float %add9526, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9527, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp867 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9530 = insertelement <4 x float> undef, float undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9530, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0x4072F4CCC0000000, float 0x4065CCCCC0000000, float 0x4051D33340000000, float 0x40680CCCC0000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp868 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp869 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9532 = fadd <4 x float> %tmp869, %tmp868
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9533 = extractelement <4 x float> undef, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp870 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9535 = insertelement <4 x float> %tmp870, float undef, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9535, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp871 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9536 = extractelement <4 x float> %tmp871, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9537 = fadd float %vecext9536, 0xC079F199A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp872 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9538 = insertelement <4 x float> %tmp872, float %add9537, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9538, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp873 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9542 = extractelement <4 x float> %tmp873, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9543 = fadd float %vecext9542, 0x4050D999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9576 = fadd float undef, 0x40219999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9577 = insertelement <4 x float> undef, float %add9576, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9577, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp874 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9580 = insertelement <4 x float> undef, float undef, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9580, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp875 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9581 = extractelement <4 x float> %tmp875, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9582 = fadd float %vecext9581, 0xC07EF33340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp876 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9583 = insertelement <4 x float> %tmp876, float %add9582, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9583, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp877 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9673 = extractelement <4 x float> undef, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9674 = fadd float %vecext9673, 0xC04CF33340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp878 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9675 = insertelement <4 x float> %tmp878, float %add9674, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9675, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9676 = extractelement <4 x float> undef, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9677 = fadd float %vecext9676, 1.455000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp879 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9678 = insertelement <4 x float> %tmp879, float %add9677, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp880 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9679 = extractelement <4 x float> %tmp880, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9680 = fadd float %vecext9679, 0x4073A33340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp881 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9681 = insertelement <4 x float> %tmp881, float %add9680, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9681, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp882 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9682 = extractelement <4 x float> %tmp882, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp883 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9686 = fadd <4 x float> %tmp883, undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add9686, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp884 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9687 = extractelement <4 x float> %tmp884, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9688 = fadd float %vecext9687, 0xC046666660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp885 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9689 = insertelement <4 x float> %tmp885, float %add9688, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9690 = extractelement <4 x float> undef, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9691 = fadd float %vecext9690, 0x4034CCCCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp886 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9692 = insertelement <4 x float> %tmp886, float %add9691, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp887 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9693 = extractelement <4 x float> %tmp887, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9694 = fadd float %vecext9693, -3.710000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp888 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9695 = insertelement <4 x float> %tmp888, float %add9694, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9695, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp889 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9697 = fadd float undef, 0x4058D33340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp890 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9698 = insertelement <4 x float> %tmp890, float %add9697, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9698, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0x4062CCCCC0000000, float 0x407AD999A0000000, float 0x40582CCCC0000000, float 0xC0712B3340000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp891 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9700 = fadd <4 x float> %tmp891, undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp892 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9701 = extractelement <4 x float> %tmp892, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9702 = fadd float %vecext9701, 0x406DC33340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp893 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9703 = insertelement <4 x float> %tmp893, float %add9702, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9703, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp894 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9704 = extractelement <4 x float> %tmp894, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9705 = fadd float %vecext9704, 0xC073B33340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp895 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9706 = insertelement <4 x float> %tmp895, float %add9705, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9706, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9707 = extractelement <4 x float> undef, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9708 = fadd float %vecext9707, 0xC0729999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp896 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9709 = insertelement <4 x float> %tmp896, float %add9708, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9709, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp897 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9710 = extractelement <4 x float> %tmp897, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9712 = insertelement <4 x float> undef, float undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9712, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0x4069F33340000000, float 0xC048266660000000, float 0x40638CCCC0000000, float 0xC07EC199A0000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp898 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9714 = fadd <4 x float> undef, %tmp898
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add9714, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp899 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9715 = extractelement <4 x float> %tmp899, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp900 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9717 = insertelement <4 x float> %tmp900, float undef, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9717, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp901 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9718 = extractelement <4 x float> %tmp901, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9719 = fadd float %vecext9718, 0x406BC66660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp902 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9720 = insertelement <4 x float> %tmp902, float %add9719, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9720, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp903 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9721 = extractelement <4 x float> %tmp903, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9722 = fadd float %vecext9721, -3.860000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp904 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9723 = insertelement <4 x float> %tmp904, float %add9722, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9723, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp905 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9724 = extractelement <4 x float> %tmp905, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9725 = fadd float %vecext9724, 0x407CF199A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp906 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9726 = insertelement <4 x float> %tmp906, float %add9725, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9726, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float -4.575000e+02, float 0x40713E6660000000, float 0x407D133340000000, float -1.425000e+02>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp907 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9728 = fadd <4 x float> %tmp907, undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add9728, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp908 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9729 = extractelement <4 x float> %tmp908, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9730 = fadd float %vecext9729, 0x4079FB3340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp909 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9731 = insertelement <4 x float> %tmp909, float %add9730, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9731, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp910 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9733 = fadd float undef, 0xC050F33340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp911 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9734 = insertelement <4 x float> %tmp911, float %add9733, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9734, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp912 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9735 = extractelement <4 x float> %tmp912, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9736 = fadd float %vecext9735, 0x40582CCCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp913 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9737 = insertelement <4 x float> %tmp913, float %add9736, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9737, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp914 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9738 = extractelement <4 x float> %tmp914, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9740 = insertelement <4 x float> undef, float undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9740, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 2.150000e+02, float 0x405A2CCCC0000000, float 2.310000e+02, float 0x404E1999A0000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp915 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp916 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp917 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9743 = extractelement <4 x float> %tmp917, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9744 = fadd float %vecext9743, -2.510000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9745 = insertelement <4 x float> undef, float %add9744, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9745, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp918 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9746 = extractelement <4 x float> %tmp918, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9747 = fadd float %vecext9746, 4.685000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp919 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9748 = insertelement <4 x float> %tmp919, float %add9747, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9748, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp920 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9749 = extractelement <4 x float> %tmp920, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9750 = fadd float %vecext9749, 1.600000e+01
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp921 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9751 = insertelement <4 x float> %tmp921, float %add9750, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9751, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp922 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9752 = extractelement <4 x float> %tmp922, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9753 = fadd float %vecext9752, -2.600000e+01
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp923 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9754 = insertelement <4 x float> %tmp923, float %add9753, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9754, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 2.590000e+02, float 0x407B7199A0000000, float 0xC07ED199A0000000, float 0xC064FCCCC0000000>, <4 x float>* %.compoundliteral9755
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp924 = load <4 x float>, <4 x float>* %.compoundliteral9755
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp925 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9756 = fadd <4 x float> %tmp925, %tmp924
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp926 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9757 = extractelement <4 x float> %tmp926, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9758 = fadd float %vecext9757, -1.810000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp927 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9759 = insertelement <4 x float> %tmp927, float %add9758, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9759, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp928 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9760 = extractelement <4 x float> %tmp928, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9761 = fadd float %vecext9760, 0xC07C3E6660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp929 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9762 = insertelement <4 x float> %tmp929, float %add9761, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9762, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp930 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9764 = fadd float undef, 0xC060E66660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp931 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9765 = insertelement <4 x float> %tmp931, float %add9764, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9765, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp932 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9766 = extractelement <4 x float> %tmp932, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9767 = fadd float %vecext9766, 0xC0753E6660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp933 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9768 = insertelement <4 x float> %tmp933, float %add9767, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9768, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0x4032CCCCC0000000, float -9.600000e+01, float -5.000000e+02, float 0x4078EE6660000000>, <4 x float>* %.compoundliteral9769
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp934 = load <4 x float>, <4 x float>* %.compoundliteral9769
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp935 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9770 = fadd <4 x float> %tmp935, %tmp934
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add9770, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp936 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9771 = extractelement <4 x float> %tmp936, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9772 = fadd float %vecext9771, 0xC0733E6660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp937 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9773 = insertelement <4 x float> %tmp937, float %add9772, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9773, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp938 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9774 = extractelement <4 x float> %tmp938, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add9775 = fadd float %vecext9774, 1.715000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp939 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9776 = insertelement <4 x float> %tmp939, float %add9775, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins9776, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext9816 = extractelement <4 x float> undef, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp940 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins9818 = insertelement <4 x float> %tmp940, float undef, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp941 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add10388 = fadd float undef, 4.755000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp942 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins10389 = insertelement <4 x float> %tmp942, float %add10388, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins10389, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp943 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext10390 = extractelement <4 x float> %tmp943, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add10391 = fadd float %vecext10390, 0xC05AECCCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp944 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins10392 = insertelement <4 x float> %tmp944, float %add10391, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins10392, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp945 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp946 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add10405 = fadd float undef, -5.650000e+01
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp947 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins10406 = insertelement <4 x float> %tmp947, float %add10405, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins10406, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp948 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext10407 = extractelement <4 x float> %tmp948, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add10408 = fadd float %vecext10407, 0xC06A633340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp949 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins10409 = insertelement <4 x float> %tmp949, float %add10408, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins10409, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp950 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext10410 = extractelement <4 x float> %tmp950, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add10411 = fadd float %vecext10410, 0xC078D66660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp951 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float -2.340000e+02, float -4.720000e+02, float 4.350000e+02, float 0xC059A66660000000>, <4 x float>* %.compoundliteral10413
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp952 = load <4 x float>, <4 x float>* %.compoundliteral10413
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp953 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add10414 = fadd <4 x float> %tmp953, %tmp952
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add10414, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp954 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext10415 = extractelement <4 x float> %tmp954, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add10416 = fadd float %vecext10415, 3.450000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp955 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins10417 = insertelement <4 x float> %tmp955, float %add10416, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins10417, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp956 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext10418 = extractelement <4 x float> %tmp956, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add10419 = fadd float %vecext10418, -6.000000e+00
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp957 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins10420 = insertelement <4 x float> %tmp957, float %add10419, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins10420, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add10422 = fadd float undef, 0xC0662CCCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext10424 = extractelement <4 x float> undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> undef, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0x402B333340000000, float 0x40735E6660000000, float 0xC0567999A0000000, float 2.050000e+02>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp958 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp959 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add10428 = fadd <4 x float> %tmp959, %tmp958
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add10428, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp960 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext10429 = extractelement <4 x float> %tmp960, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add10430 = fadd float %vecext10429, 0xC075166660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp961 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add10436 = fadd float undef, 0xC06AF33340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp962 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins10437 = insertelement <4 x float> %tmp962, float %add10436, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins10437, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext10438 = extractelement <4 x float> undef, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add10439 = fadd float %vecext10438, 0x405C7999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp963 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins10440 = insertelement <4 x float> %tmp963, float %add10439, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins10440, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0xC065E999A0000000, float 0x4067D33340000000, float 0xC070133340000000, float 0x406B666660000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp964 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp965 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext10443 = extractelement <4 x float> %tmp965, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add10444 = fadd float %vecext10443, 0xC06CA999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp966 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins10445 = insertelement <4 x float> %tmp966, float %add10444, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins10445, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp967 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext10446 = extractelement <4 x float> %tmp967, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add10447 = fadd float %vecext10446, 0x4064B999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp968 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins10448 = insertelement <4 x float> %tmp968, float %add10447, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins10448, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp969 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext10449 = extractelement <4 x float> %tmp969, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add10450 = fadd float %vecext10449, 0x407B3CCCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp970 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins10451 = insertelement <4 x float> %tmp970, float %add10450, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins10451, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp971 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext10452 = extractelement <4 x float> %tmp971, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add10453 = fadd float %vecext10452, -2.225000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins10454 = insertelement <4 x float> undef, float %add10453, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0x406AFCCCC0000000, float 0xC07604CCC0000000, float 6.900000e+01, float 0xC060A66660000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp972 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp973 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add10456 = fadd <4 x float> %tmp973, %tmp972
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %add10456, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp974 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext10457 = extractelement <4 x float> %tmp974, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add10458 = fadd float %vecext10457, 2.375000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins10459 = insertelement <4 x float> undef, float %add10458, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins10459, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp975 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext10460 = extractelement <4 x float> %tmp975, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add10461 = fadd float %vecext10460, 0xC06B3999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp976 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins10462 = insertelement <4 x float> %tmp976, float %add10461, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp977 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext10463 = extractelement <4 x float> %tmp977, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add10464 = fadd float %vecext10463, 0x40655999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp978 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins10465 = insertelement <4 x float> %tmp978, float %add10464, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins10465, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp979 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext10466 = extractelement <4 x float> %tmp979, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add10467 = fadd float %vecext10466, 0xC07B6999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp980 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins10468 = insertelement <4 x float> %tmp980, float %add10467, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins10468, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0x4078833340000000, float 0x40786CCCC0000000, float 0xC0468CCCC0000000, float 0xC0793199A0000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp981 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add10470 = fadd <4 x float> %tmp981, undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp982 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext10471 = extractelement <4 x float> %tmp982, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add10472 = fadd float %vecext10471, 0x40710CCCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp983 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins10473 = insertelement <4 x float> %tmp983, float %add10472, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins10473, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp984 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext10474 = extractelement <4 x float> %tmp984, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add10475 = fadd float %vecext10474, 0x40709B3340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp985 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins10476 = insertelement <4 x float> %tmp985, float %add10475, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins10476, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add10489 = fadd float undef, 0x4074666660000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp986 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins10490 = insertelement <4 x float> %tmp986, float %add10489, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins10490, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp987 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp988 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext10508 = extractelement <4 x float> %tmp988, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add10509 = fadd float %vecext10508, 0xC027333340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp989 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins10510 = insertelement <4 x float> %tmp989, float %add10509, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins10510, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0x40656999A0000000, float 0xC073766660000000, float 1.685000e+02, float 0x40765199A0000000>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp990 = load <4 x float>, <4 x float>* undef
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add10512 = fadd <4 x float> undef, %tmp990
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp991 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext10513 = extractelement <4 x float> %tmp991, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add10514 = fadd float %vecext10513, 0x405BB999A0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp992 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins10515 = insertelement <4 x float> %tmp992, float %add10514, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins10515, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp993 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add10562 = fadd float undef, 2.035000e+02
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp994 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins10563 = insertelement <4 x float> %tmp994, float %add10562, i32 2
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins10563, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp995 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext10564 = extractelement <4 x float> %tmp995, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add10565 = fadd float %vecext10564, 0x407AE4CCC0000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp996 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins10566 = insertelement <4 x float> %tmp996, float %add10565, i32 3
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins10566, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> <float 0xC068B999A0000000, float 0xC050E66660000000, float 0xC0725999A0000000, float 0xC054D33340000000>, <4 x float>* %.compoundliteral10567
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp997 = load <4 x float>, <4 x float>* %.compoundliteral10567
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp998 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add10568 = fadd <4 x float> %tmp998, %tmp997
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp999 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext10569 = extractelement <4 x float> %tmp999, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add10570 = fadd float %vecext10569, 0x4074C33340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp1000 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins10571 = insertelement <4 x float> %tmp1000, float %add10570, i32 0
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins10571, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp1001 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecext10572 = extractelement <4 x float> %tmp1001, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %add10573 = fadd float %vecext10572, 0x407DF33340000000
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %tmp1002 = load <4 x float>, <4 x float>* undef, align 16
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  %vecins10574 = insertelement <4 x float> %tmp1002, float %add10573, i32 1
  tail call void asm sideeffect "", "~{q0}{q1}{q2}{q3}{q4}{q5}{q6}{q7}{q8}{q9}{q10}{q11}{q12}{q13}{q14}{q15}"()
  store <4 x float> %vecins10574, <4 x float>* undef, align 16
  %tmp1003 = load <4 x float>, <4 x float>* undef, align 16
  %vecext10575 = extractelement <4 x float> %tmp1003, i32 2
  %tmp1004 = load <4 x float>, <4 x float>* undef, align 16
  %vecins10577 = insertelement <4 x float> %tmp1004, float undef, i32 2
  store <4 x float> %vecins10577, <4 x float>* undef, align 16
  %tmp1005 = load <4 x float>, <4 x float>* undef, align 16
  %vecext10578 = extractelement <4 x float> %tmp1005, i32 3
  %add10579 = fadd float %vecext10578, 0x4076566660000000
  %tmp1006 = load <4 x float>, <4 x float>* undef, align 16
  %vecins10580 = insertelement <4 x float> %tmp1006, float %add10579, i32 3
  store <4 x float> %vecins10580, <4 x float>* undef, align 16
  store <4 x float> <float 0x407CAB3340000000, float 1.685000e+02, float 0xC07B866660000000, float 0xC061ACCCC0000000>, <4 x float>* %.compoundliteral10581
  %tmp1007 = load <4 x float>, <4 x float>* %.compoundliteral10581
  store <4 x float> undef, <4 x float>* undef, align 16
  %tmp1008 = load <4 x float>, <4 x float>* undef, align 16
  %vecext10583 = extractelement <4 x float> %tmp1008, i32 0
  %add10584 = fadd float %vecext10583, 0xC060533340000000
  %tmp1009 = load <4 x float>, <4 x float>* undef, align 16
  %vecins10585 = insertelement <4 x float> %tmp1009, float %add10584, i32 0
  store <4 x float> %vecins10585, <4 x float>* undef, align 16
  %tmp1010 = load <4 x float>, <4 x float>* undef, align 16
  %vecext10586 = extractelement <4 x float> %tmp1010, i32 1
  %add10587 = fadd float %vecext10586, 0xC0694CCCC0000000
  %tmp1011 = load <4 x float>, <4 x float>* undef, align 16
  %vecins10588 = insertelement <4 x float> %tmp1011, float %add10587, i32 1
  store <4 x float> %vecins10588, <4 x float>* undef, align 16
  %tmp1012 = load <4 x float>, <4 x float>* undef, align 16
  %vecext10589 = extractelement <4 x float> %tmp1012, i32 2
  %add10590 = fadd float %vecext10589, 0xC0541999A0000000
  %tmp1013 = load <4 x float>, <4 x float>* undef, align 16
  %vecins10591 = insertelement <4 x float> %tmp1013, float %add10590, i32 2
  store <4 x float> %vecins10591, <4 x float>* undef, align 16
  %tmp1014 = load <4 x float>, <4 x float>* undef, align 16
  %vecext10592 = extractelement <4 x float> %tmp1014, i32 3
  %add10593 = fadd float %vecext10592, 0xC06C566660000000
  %tmp1015 = load <4 x float>, <4 x float>* undef, align 16
  %vecins10594 = insertelement <4 x float> %tmp1015, float %add10593, i32 3
  store <4 x float> %vecins10594, <4 x float>* undef, align 16
  store <4 x float> <float 0x407A3199A0000000, float 0xC0659999A0000000, float 0x407E0999A0000000, float 0xC0334CCCC0000000>, <4 x float>* %.compoundliteral10595
  %tmp1016 = load <4 x float>, <4 x float>* %.compoundliteral10595
  %tmp1017 = load <4 x float>, <4 x float>* undef, align 16
  %add10596 = fadd <4 x float> %tmp1017, %tmp1016
  store <4 x float> %add10596, <4 x float>* undef, align 16
  %tmp1018 = load <4 x float>, <4 x float>* undef, align 16
  %vecext10597 = extractelement <4 x float> %tmp1018, i32 0
  %add10598 = fadd float %vecext10597, 0x40640999A0000000
  %tmp1019 = load <4 x float>, <4 x float>* undef, align 16
  %vecins10599 = insertelement <4 x float> %tmp1019, float %add10598, i32 0
  store <4 x float> %vecins10599, <4 x float>* undef, align 16
  %tmp1020 = load <4 x float>, <4 x float>* undef, align 16
  %vecext10600 = extractelement <4 x float> %tmp1020, i32 1
  %add10601 = fadd float %vecext10600, 0xC073966660000000
  %tmp1021 = load <4 x float>, <4 x float>* undef, align 16
  %vecins10602 = insertelement <4 x float> %tmp1021, float %add10601, i32 1
  %tmp1022 = load <4 x float>, <4 x float>* undef, align 16
  %vecext10603 = extractelement <4 x float> %tmp1022, i32 2
  %add10604 = fadd float %vecext10603, 1.780000e+02
  %tmp1023 = load <4 x float>, <4 x float>* undef, align 16
  %vecins10605 = insertelement <4 x float> %tmp1023, float %add10604, i32 2
  store <4 x float> %vecins10605, <4 x float>* undef, align 16
  %tmp1024 = load <4 x float>, <4 x float>* undef, align 16
  %add10607 = fadd float undef, 0x4070A33340000000
  %tmp1025 = load <4 x float>, <4 x float>* undef, align 16
  store <4 x float> <float 0x407C5999A0000000, float 0x4046733340000000, float 0xC06E6CCCC0000000, float 0xC063C33340000000>, <4 x float>* %.compoundliteral10609
  %tmp1026 = load <4 x float>, <4 x float>* %.compoundliteral10609
  %tmp1027 = load <4 x float>, <4 x float>* undef, align 16
  %tmp1028 = load <4 x float>, <4 x float>* undef, align 16
  %vecext10611 = extractelement <4 x float> %tmp1028, i32 0
  %add10612 = fadd float %vecext10611, 0x40757199A0000000
  %vecins10613 = insertelement <4 x float> undef, float %add10612, i32 0
  store <4 x float> %vecins10613, <4 x float>* undef, align 16
  %tmp1029 = load <4 x float>, <4 x float>* undef, align 16
  %vecext10614 = extractelement <4 x float> %tmp1029, i32 1
  %add10615 = fadd float %vecext10614, 0x40740CCCC0000000
  %tmp1030 = load <4 x float>, <4 x float>* undef, align 16
  %vecins10616 = insertelement <4 x float> %tmp1030, float %add10615, i32 1
  store <4 x float> %vecins10616, <4 x float>* undef, align 16
  %tmp1031 = load <4 x float>, <4 x float>* undef, align 16
  %vecext10617 = extractelement <4 x float> %tmp1031, i32 2
  %add10618 = fadd float %vecext10617, 0xC012CCCCC0000000
  %tmp1032 = load <4 x float>, <4 x float>* undef, align 16
  %vecins10619 = insertelement <4 x float> %tmp1032, float %add10618, i32 2
  store <4 x float> %vecins10619, <4 x float>* undef, align 16
  %tmp1033 = load <4 x float>, <4 x float>* undef, align 16
  %vecext10620 = extractelement <4 x float> %tmp1033, i32 3
  %add10621 = fadd float %vecext10620, 0x406E566660000000
  %tmp1034 = load <4 x float>, <4 x float>* undef, align 16
  store <4 x float> <float 0x407B2199A0000000, float 0xC07D9CCCC0000000, float -4.350000e+01, float 0xC07D3B3340000000>, <4 x float>* %.compoundliteral10623
  %tmp1035 = load <4 x float>, <4 x float>* %.compoundliteral10623
  %add10624 = fadd <4 x float> undef, %tmp1035
  %tmp1036 = load <4 x float>, <4 x float>* undef, align 16
  %vecext10625 = extractelement <4 x float> %tmp1036, i32 0
  %tmp1037 = load <4 x float>, <4 x float>* undef, align 16
  %vecins10627 = insertelement <4 x float> %tmp1037, float undef, i32 0
  store <4 x float> %vecins10627, <4 x float>* undef, align 16
  %tmp1038 = load <4 x float>, <4 x float>* undef, align 16
  %vecext10628 = extractelement <4 x float> %tmp1038, i32 1
  %add10629 = fadd float %vecext10628, 0x407E3CCCC0000000
  %tmp1039 = load <4 x float>, <4 x float>* undef, align 16
  %vecins10630 = insertelement <4 x float> %tmp1039, float %add10629, i32 1
  store <4 x float> %vecins10630, <4 x float>* undef, align 16
  %tmp1040 = load <4 x float>, <4 x float>* undef, align 16
  %vecext10631 = extractelement <4 x float> %tmp1040, i32 2
  %tmp1041 = load <4 x float>, <4 x float>* undef, align 16
  %tmp1042 = load <4 x float>, <4 x float>* undef, align 16
  %vecext10634 = extractelement <4 x float> %tmp1042, i32 3
  %add10635 = fadd float %vecext10634, 0xC067533340000000
  %tmp1043 = load <4 x float>, <4 x float>* undef, align 16
  %vecins10636 = insertelement <4 x float> %tmp1043, float %add10635, i32 3
  store <4 x float> %vecins10636, <4 x float>* undef, align 16
  store <4 x float> <float 1.950000e+02, float 0x407E8E6660000000, float 0x407D7CCCC0000000, float 0x407E166660000000>, <4 x float>* %.compoundliteral10637
  %tmp1044 = load <4 x float>, <4 x float>* undef, align 16
  %add10638 = fadd <4 x float> %tmp1044, undef
  %tmp1045 = load <4 x float>, <4 x float>* undef, align 16
  %vecext10639 = extractelement <4 x float> %tmp1045, i32 0
  %add10640 = fadd float %vecext10639, 0x406CA33340000000
  %tmp1046 = load <4 x float>, <4 x float>* undef, align 16
  %vecins10641 = insertelement <4 x float> %tmp1046, float %add10640, i32 0
  store <4 x float> %vecins10641, <4 x float>* undef, align 16
  %tmp1047 = load <4 x float>, <4 x float>* undef, align 16
  %vecext10642 = extractelement <4 x float> %tmp1047, i32 1
  %add10643 = fadd float %vecext10642, 0xC07C8999A0000000
  %tmp1048 = load <4 x float>, <4 x float>* undef, align 16
  %vecins10644 = insertelement <4 x float> %tmp1048, float %add10643, i32 1
  store <4 x float> %vecins10644, <4 x float>* undef, align 16
  %tmp1049 = load <4 x float>, <4 x float>* undef, align 16
  %vecext10645 = extractelement <4 x float> %tmp1049, i32 2
  %tmp1050 = load <4 x float>, <4 x float>* undef, align 16
  %tmp1051 = load <4 x float>, <4 x float>* undef, align 16
  %vecins10748 = insertelement <4 x float> undef, float undef, i32 3
  %tmp1052 = load <4 x float>, <4 x float>* %.compoundliteral10749
  %add10750 = fadd <4 x float> undef, %tmp1052
  store <4 x float> %add10750, <4 x float>* undef, align 16
  %tmp1053 = load <4 x float>, <4 x float>* undef, align 16
  %vecext10751 = extractelement <4 x float> %tmp1053, i32 0
  %add10752 = fadd float %vecext10751, 0x4071B33340000000
  %tmp1054 = load <4 x float>, <4 x float>* undef, align 16
  %vecins10753 = insertelement <4 x float> %tmp1054, float %add10752, i32 0
  store <4 x float> %vecins10753, <4 x float>* undef, align 16
  %tmp1055 = load <4 x float>, <4 x float>* undef, align 16
  %vecext10754 = extractelement <4 x float> %tmp1055, i32 1
  %add10755 = fadd float %vecext10754, 0xC076A66660000000
  %tmp1056 = load <4 x float>, <4 x float>* undef, align 16
  %vecins10756 = insertelement <4 x float> %tmp1056, float %add10755, i32 1
  store <4 x float> %vecins10756, <4 x float>* undef, align 16
  %tmp1057 = load <4 x float>, <4 x float>* undef, align 16
  %vecext10757 = extractelement <4 x float> %tmp1057, i32 2
  %add10758 = fadd float %vecext10757, 3.800000e+01
  %tmp1058 = load <4 x float>, <4 x float>* undef, align 16
  %vecins10759 = insertelement <4 x float> %tmp1058, float %add10758, i32 2
  store <4 x float> %vecins10759, <4 x float>* undef, align 16
  %tmp1059 = load <4 x float>, <4 x float>* undef, align 16
  %vecext10760 = extractelement <4 x float> %tmp1059, i32 3
  store <4 x float> undef, <4 x float>* undef, align 16
  store <4 x float> <float 0xC075BB3340000000, float 0x4074D4CCC0000000, float 0xC07A466660000000, float 0xC0691CCCC0000000>, <4 x float>* %.compoundliteral10763
  %tmp1060 = load <4 x float>, <4 x float>* %.compoundliteral10763
  %tmp1061 = load <4 x float>, <4 x float>* undef, align 16
  %tmp1062 = load <4 x float>, <4 x float>* undef, align 16
  %add10985 = fadd float undef, 0x405E933340000000
  %tmp1063 = load <4 x float>, <4 x float>* undef, align 16
  %vecins10986 = insertelement <4 x float> %tmp1063, float %add10985, i32 3
  store <4 x float> %vecins10986, <4 x float>* undef, align 16
  store <4 x float> <float 0xC0721E6660000000, float -4.180000e+02, float 0x406F366660000000, float 0xC055F999A0000000>, <4 x float>* %.compoundliteral10987
  %tmp1064 = load <4 x float>, <4 x float>* %.compoundliteral10987
  %tmp1065 = load <4 x float>, <4 x float>* undef, align 16
  %vecins10994 = insertelement <4 x float> %tmp1065, float undef, i32 1
  %tmp1066 = load <4 x float>, <4 x float>* undef, align 16
  %vecext10995 = extractelement <4 x float> %tmp1066, i32 2
  %add10996 = fadd float %vecext10995, 0x406F9999A0000000
  %tmp1067 = load <4 x float>, <4 x float>* undef, align 16
  %vecins10997 = insertelement <4 x float> %tmp1067, float %add10996, i32 2
  store <4 x float> %vecins10997, <4 x float>* undef, align 16
  %tmp1068 = load <4 x float>, <4 x float>* undef, align 16
  %vecext10998 = extractelement <4 x float> %tmp1068, i32 3
  %add10999 = fadd float %vecext10998, -2.765000e+02
  %tmp1069 = load <4 x float>, <4 x float>* undef, align 16
  %vecins11000 = insertelement <4 x float> %tmp1069, float %add10999, i32 3
  store <4 x float> %vecins11000, <4 x float>* undef, align 16
  store <4 x float> <float 0x4078F999A0000000, float 0xC06D166660000000, float 0x40501999A0000000, float 0x406FC999A0000000>, <4 x float>* %.compoundliteral11001
  %tmp1070 = load <4 x float>, <4 x float>* undef, align 16
  %add11002 = fadd <4 x float> %tmp1070, undef
  %vecext11003 = extractelement <4 x float> undef, i32 0
  %vecext11009 = extractelement <4 x float> undef, i32 2
  %tmp1071 = load <4 x float>, <4 x float>* undef, align 16
  %vecins11033 = insertelement <4 x float> %tmp1071, float undef, i32 0
  store <4 x float> %vecins11033, <4 x float>* undef, align 16
  %tmp1072 = load <4 x float>, <4 x float>* undef, align 16
  %vecext11034 = extractelement <4 x float> %tmp1072, i32 1
  %add11035 = fadd float %vecext11034, 0x4056D33340000000
  %tmp1073 = load <4 x float>, <4 x float>* undef, align 16
  %vecins11036 = insertelement <4 x float> %tmp1073, float %add11035, i32 1
  store <4 x float> %vecins11036, <4 x float>* undef, align 16
  %tmp1074 = load <4 x float>, <4 x float>* undef, align 16
  %vecext11037 = extractelement <4 x float> %tmp1074, i32 2
  %add11038 = fadd float %vecext11037, 0xC06EA33340000000
  %tmp1075 = load <4 x float>, <4 x float>* undef, align 16
  store <4 x float> undef, <4 x float>* undef, align 16
  %tmp1076 = load <4 x float>, <4 x float>* undef, align 16
  %vecext11040 = extractelement <4 x float> %tmp1076, i32 3
  %add11041 = fadd float %vecext11040, 0x40746CCCC0000000
  %tmp1077 = load <4 x float>, <4 x float>* undef, align 16
  %vecins11042 = insertelement <4 x float> %tmp1077, float %add11041, i32 3
  store <4 x float> <float 0x405DD999A0000000, float -3.775000e+02, float -1.265000e+02, float 0xC065C66660000000>, <4 x float>* undef
  %tmp1078 = load <4 x float>, <4 x float>* undef, align 16
  %add11044 = fadd <4 x float> %tmp1078, undef
  store <4 x float> %add11044, <4 x float>* undef, align 16
  %tmp1079 = load <4 x float>, <4 x float>* undef, align 16
  %vecext11045 = extractelement <4 x float> %tmp1079, i32 0
  %add11046 = fadd float %vecext11045, 0xC076E66660000000
  %tmp1080 = load <4 x float>, <4 x float>* undef, align 16
  %vecins11047 = insertelement <4 x float> %tmp1080, float %add11046, i32 0
  %tmp1081 = load <4 x float>, <4 x float>* undef, align 16
  %vecext11048 = extractelement <4 x float> %tmp1081, i32 1
  %add11049 = fadd float %vecext11048, 4.100000e+02
  %vecins11064 = insertelement <4 x float> undef, float undef, i32 1
  %add11074 = fadd float undef, 0xC06FF999A0000000
  %tmp1082 = load <4 x float>, <4 x float>* undef, align 16
  %vecins11075 = insertelement <4 x float> %tmp1082, float %add11074, i32 0
  store <4 x float> %vecins11075, <4 x float>* undef, align 16
  %add11077 = fadd float undef, 0xC075D33340000000
  %tmp1083 = load <4 x float>, <4 x float>* undef, align 16
  %tmp1084 = load <4 x float>, <4 x float>* undef, align 16
  store <4 x float> undef, <4 x float>* undef, align 16
  %tmp1085 = load <4 x float>, <4 x float>* undef, align 16
  %vecext11093 = extractelement <4 x float> %tmp1085, i32 2
  %add11094 = fadd float %vecext11093, 0xC07CD66660000000
  %tmp1086 = load <4 x float>, <4 x float>* undef, align 16
  %vecins11095 = insertelement <4 x float> %tmp1086, float %add11094, i32 2
  store <4 x float> %vecins11095, <4 x float>* undef, align 16
  store <4 x float> undef, <4 x float>* undef, align 16
  store <4 x float> <float 0x4061F66660000000, float 0xC076DB3340000000, float 0xC055A66660000000, float 2.415000e+02>, <4 x float>* undef
  %tmp1087 = load <4 x float>, <4 x float>* undef
  store <4 x float> undef, <4 x float>* undef, align 16
  %tmp1088 = load <4 x float>, <4 x float>* undef, align 16
  %vecext11513 = extractelement <4 x float> %tmp1088, i32 2
  %add11514 = fadd float %vecext11513, 0xC07C7199A0000000
  %vecins11515 = insertelement <4 x float> undef, float %add11514, i32 2
  store <4 x float> %vecins11515, <4 x float>* undef, align 16
  %add11520 = fadd <4 x float> undef, undef
  store <4 x float> %add11520, <4 x float>* undef, align 16
  %vecext11521 = extractelement <4 x float> undef, i32 0
  %add11522 = fadd float %vecext11521, 0x4041733340000000
  %tmp1089 = load <4 x float>, <4 x float>* undef, align 16
  store <4 x float> undef, <4 x float>* undef, align 16
  %tmp1090 = load <4 x float>, <4 x float>* undef
  %tmp1091 = load <4 x float>, <4 x float>* undef, align 16
  %add11562 = fadd <4 x float> %tmp1091, %tmp1090
  %tmp1092 = load <4 x float>, <4 x float>* undef, align 16
  %add11564 = fadd float undef, 0xC0411999A0000000
  %tmp1093 = load <4 x float>, <4 x float>* undef, align 16
  %vecins11565 = insertelement <4 x float> %tmp1093, float %add11564, i32 0
  store <4 x float> undef, <4 x float>* undef, align 16
  %vecext11586 = extractelement <4 x float> undef, i32 3
  %add11587 = fadd float %vecext11586, 3.760000e+02
  %tmp1094 = load <4 x float>, <4 x float>* undef, align 16
  store <4 x float> undef, <4 x float>* undef, align 16
  store <4 x float> <float 0xC06ED999A0000000, float 1.380000e+02, float 0xC073AB3340000000, float 0x4078A66660000000>, <4 x float>* undef
  %tmp1095 = load <4 x float>, <4 x float>* undef
  %tmp1096 = load <4 x float>, <4 x float>* undef, align 16
  %tmp1097 = load <4 x float>, <4 x float>* undef, align 16
  %tmp1098 = load <4 x float>, <4 x float>* undef, align 16
  %vecins11593 = insertelement <4 x float> %tmp1098, float undef, i32 0
  %vecext11594 = extractelement <4 x float> undef, i32 1
  %tmp1099 = load <4 x float>, <4 x float>* undef, align 16
  %vecins11596 = insertelement <4 x float> %tmp1099, float undef, i32 1
  store <4 x float> %vecins11596, <4 x float>* undef, align 16
  %tmp1100 = load <4 x float>, <4 x float>* undef, align 16
  %vecext11597 = extractelement <4 x float> %tmp1100, i32 2
  %add11598 = fadd float %vecext11597, 0x40430CCCC0000000
  %tmp1101 = load <4 x float>, <4 x float>* undef, align 16
  %vecins11599 = insertelement <4 x float> %tmp1101, float %add11598, i32 2
  %tmp1102 = load <4 x float>, <4 x float>* undef, align 16
  %vecext11600 = extractelement <4 x float> %tmp1102, i32 3
  %tmp1103 = load <4 x float>, <4 x float>* undef, align 16
  %vecins11602 = insertelement <4 x float> %tmp1103, float undef, i32 3
  store <4 x float> %vecins11602, <4 x float>* undef, align 16
  %tmp1104 = load <4 x float>, <4 x float>* undef
  %tmp1105 = load <4 x float>, <4 x float>* undef, align 16
  %add11604 = fadd <4 x float> %tmp1105, %tmp1104
  %tmp1106 = load <4 x float>, <4 x float>* undef, align 16
  %vecext11605 = extractelement <4 x float> %tmp1106, i32 0
  %tmp1107 = load <4 x float>, <4 x float>* undef, align 16
  %vecins11607 = insertelement <4 x float> %tmp1107, float undef, i32 0
  %vecins11621 = insertelement <4 x float> undef, float undef, i32 0
  %vecins11630 = insertelement <4 x float> undef, float undef, i32 3
  store <4 x float> %vecins11630, <4 x float>* undef, align 16
  store <4 x float> <float -1.190000e+02, float 0x402F666660000000, float 0xC07BD33340000000, float -1.595000e+02>, <4 x float>* %.compoundliteral11631
  %tmp1108 = load <4 x float>, <4 x float>* %.compoundliteral11631
  %tmp1109 = load <4 x float>, <4 x float>* undef, align 16
  store <4 x float> undef, <4 x float>* undef, align 16
  %add11634 = fadd float undef, -1.075000e+02
  %vecext11647 = extractelement <4 x float> undef, i32 0
  %add11648 = fadd float %vecext11647, 0x40775999A0000000
  %tmp1110 = load <4 x float>, <4 x float>* undef, align 16
  %vecext11650 = extractelement <4 x float> undef, i32 1
  %tmp1111 = load <4 x float>, <4 x float>* undef, align 16
  %vecins11784 = insertelement <4 x float> %tmp1111, float undef, i32 3
  store <4 x float> %vecins11784, <4 x float>* undef, align 16
  store <4 x float> <float 1.605000e+02, float 0x4068366660000000, float 2.820000e+02, float 0x407CF66660000000>, <4 x float>* %.compoundliteral11785
  %tmp1112 = load <4 x float>, <4 x float>* %.compoundliteral11785
  %add11786 = fadd <4 x float> undef, %tmp1112
  store <4 x float> %add11786, <4 x float>* undef, align 16
  %tmp1113 = load <4 x float>, <4 x float>* undef, align 16
  %vecext11787 = extractelement <4 x float> %tmp1113, i32 0
  %vecext11807 = extractelement <4 x float> undef, i32 2
  %add11808 = fadd float %vecext11807, 4.535000e+02
  %tmp1114 = load <4 x float>, <4 x float>* undef, align 16
  %vecext11810 = extractelement <4 x float> undef, i32 3
  %add11811 = fadd float %vecext11810, 0x4068F66660000000
  %tmp1115 = load <4 x float>, <4 x float>* undef, align 16
  %vecins11812 = insertelement <4 x float> %tmp1115, float %add11811, i32 3
  store <4 x float> %vecins11812, <4 x float>* undef, align 16
  %tmp1116 = load <4 x float>, <4 x float>* undef
  %tmp1117 = load <4 x float>, <4 x float>* undef, align 16
  %vecext11958 = extractelement <4 x float> undef, i32 1
  store <4 x float> undef, <4 x float>* undef, align 16
  %vecext11961 = extractelement <4 x float> undef, i32 2
  %add11962 = fadd float %vecext11961, -3.680000e+02
  %tmp1118 = load <4 x float>, <4 x float>* undef, align 16
  store <4 x float> undef, <4 x float>* undef, align 16
  %add11965 = fadd float undef, 0x4061133340000000
  store <4 x float> undef, <4 x float>* undef, align 16
  %tmp1119 = load <4 x float>, <4 x float>* undef, align 16
  %vecext11975 = extractelement <4 x float> %tmp1119, i32 2
  %tmp1120 = load <4 x float>, <4 x float>* undef, align 16
  %vecins11977 = insertelement <4 x float> %tmp1120, float undef, i32 2
  store <4 x float> %vecins11977, <4 x float>* undef, align 16
  %vecext11978 = extractelement <4 x float> undef, i32 3
  %add11979 = fadd float %vecext11978, 0xC0688999A0000000
  %tmp1121 = load <4 x float>, <4 x float>* undef, align 16
  %vecins11980 = insertelement <4 x float> %tmp1121, float %add11979, i32 3
  store <4 x float> %vecins11980, <4 x float>* undef, align 16
  %add11982 = fadd <4 x float> undef, undef
  store <4 x float> %add11982, <4 x float>* undef, align 16
  %tmp1122 = load <4 x float>, <4 x float>* undef, align 16
  %vecext11983 = extractelement <4 x float> %tmp1122, i32 0
  %add11984 = fadd float %vecext11983, 0xC075966660000000
  %tmp1123 = load <4 x float>, <4 x float>* undef, align 16
  %vecins12005 = insertelement <4 x float> undef, float undef, i32 2
  store <4 x float> %vecins12005, <4 x float>* undef, align 16
  %tmp1124 = load <4 x float>, <4 x float>* undef, align 16
  %add12007 = fadd float undef, 0xC07124CCC0000000
  %vecins12008 = insertelement <4 x float> undef, float %add12007, i32 3
  store <4 x float> %vecins12008, <4 x float>* undef, align 16
  %tmp1125 = load <4 x float>, <4 x float>* undef, align 16
  store <4 x float> undef, <4 x float>* undef, align 16
  %tmp1126 = load <4 x float>, <4 x float>* undef, align 16
  %add12012 = fadd float undef, 0xC0750CCCC0000000
  %tmp1127 = load <4 x float>, <4 x float>* undef, align 16
  %vecins12013 = insertelement <4 x float> %tmp1127, float %add12012, i32 0
  store <4 x float> %vecins12013, <4 x float>* undef, align 16
  %tmp1128 = load <4 x float>, <4 x float>* undef, align 16
  %add12015 = fadd float undef, 0x4079CE6660000000
  %tmp1129 = load <4 x float>, <4 x float>* undef, align 16
  %vecins12016 = insertelement <4 x float> %tmp1129, float %add12015, i32 1
  store <4 x float> %vecins12016, <4 x float>* undef, align 16
  %add12018 = fadd float undef, 3.555000e+02
  %tmp1130 = load <4 x float>, <4 x float>* undef, align 16
  %vecins12019 = insertelement <4 x float> %tmp1130, float %add12018, i32 2
  %tmp1131 = load <4 x float>, <4 x float>* undef, align 16
  %vecext12020 = extractelement <4 x float> %tmp1131, i32 3
  store <4 x float> undef, <4 x float>* undef, align 16
  %vecext12028 = extractelement <4 x float> undef, i32 1
  store <4 x float> undef, <4 x float>* undef, align 16
  store <4 x float> <float 0x40791999A0000000, float 0x407C7CCCC0000000, float 0x4070F33340000000, float 0xC056ECCCC0000000>, <4 x float>* undef
  %tmp1132 = load <4 x float>, <4 x float>* undef, align 16
  %add12038 = fadd <4 x float> %tmp1132, undef
  %tmp1133 = load <4 x float>, <4 x float>* undef, align 16
  %vecext12042 = extractelement <4 x float> %tmp1133, i32 1
  %add12043 = fadd float %vecext12042, 0x402F9999A0000000
  %tmp1134 = load <4 x float>, <4 x float>* undef, align 16
  %vecins12044 = insertelement <4 x float> %tmp1134, float %add12043, i32 1
  store <4 x float> %vecins12044, <4 x float>* undef, align 16
  %vecext12045 = extractelement <4 x float> undef, i32 2
  %add12046 = fadd float %vecext12045, 0xC07EF33340000000
  %tmp1135 = load <4 x float>, <4 x float>* undef, align 16
  %vecins12047 = insertelement <4 x float> %tmp1135, float %add12046, i32 2
  store <4 x float> %vecins12047, <4 x float>* undef, align 16
  store <4 x float> undef, <4 x float>* undef, align 16
  %tmp1136 = load <4 x float>, <4 x float>* undef, align 16
  %vecext12112 = extractelement <4 x float> %tmp1136, i32 1
  %tmp1137 = load <4 x float>, <4 x float>* undef, align 16
  store <4 x float> undef, <4 x float>* undef, align 16
  %add12116 = fadd float undef, 0xC074F4CCC0000000
  %tmp1138 = load <4 x float>, <4 x float>* undef, align 16
  %vecins12117 = insertelement <4 x float> %tmp1138, float %add12116, i32 2
  store <4 x float> %vecins12117, <4 x float>* undef, align 16
  %tmp1139 = load <4 x float>, <4 x float>* undef, align 16
  %vecext12118 = extractelement <4 x float> %tmp1139, i32 3
  %add12119 = fadd float %vecext12118, 0xC0638CCCC0000000
  %tmp1140 = load <4 x float>, <4 x float>* undef, align 16
  %vecins12120 = insertelement <4 x float> %tmp1140, float %add12119, i32 3
  %add12152 = fadd float undef, 0x4039333340000000
  %tmp1141 = load <4 x float>, <4 x float>* undef, align 16
  %vecins12153 = insertelement <4 x float> %tmp1141, float %add12152, i32 0
  %vecext12154 = extractelement <4 x float> undef, i32 1
  %add12155 = fadd float %vecext12154, 0xC07BBB3340000000
  %tmp1142 = load <4 x float>, <4 x float>* undef, align 16
  %vecins12156 = insertelement <4 x float> %tmp1142, float %add12155, i32 1
  %tmp1143 = load <4 x float>, <4 x float>* undef, align 16
  %vecext12157 = extractelement <4 x float> %tmp1143, i32 2
  %add12158 = fadd float %vecext12157, 0xC0428CCCC0000000
  %tmp1144 = load <4 x float>, <4 x float>* undef, align 16
  %vecins12159 = insertelement <4 x float> %tmp1144, float %add12158, i32 2
  %tmp1145 = load <4 x float>, <4 x float>* undef, align 16
  %vecext12160 = extractelement <4 x float> %tmp1145, i32 3
  %add12161 = fadd float %vecext12160, 0x407B1999A0000000
  %tmp1146 = load <4 x float>, <4 x float>* undef, align 16
  %vecins12162 = insertelement <4 x float> %tmp1146, float %add12161, i32 3
  store <4 x float> %vecins12162, <4 x float>* undef, align 16
  %tmp1147 = load <4 x float>, <4 x float>* undef
  %tmp1148 = load <4 x float>, <4 x float>* undef, align 16
  %tmp1149 = load <4 x float>, <4 x float>* undef, align 16
  %vecext12182 = extractelement <4 x float> %tmp1149, i32 1
  %tmp1150 = load <4 x float>, <4 x float>* undef, align 16
  store <4 x float> undef, <4 x float>* undef, align 16
  store <4 x float> <float 0x4061833340000000, float 0x405CA66660000000, float -1.275000e+02, float 0x405BC66660000000>, <4 x float>* undef
  %add12208 = fadd float undef, 0x407854CCC0000000
  %tmp1151 = load <4 x float>, <4 x float>* undef, align 16
  store <4 x float> undef, <4 x float>* undef, align 16
  %tmp1152 = load <4 x float>, <4 x float>* undef, align 16
  %tmp1153 = load <4 x float>, <4 x float>* undef, align 16
  %vecins12218 = insertelement <4 x float> undef, float undef, i32 3
  store <4 x float> %vecins12218, <4 x float>* undef, align 16
  store <4 x float> <float 0x407C3CCCC0000000, float 0xC057C66660000000, float 2.605000e+02, float 0xC07974CCC0000000>, <4 x float>* undef
  %tmp1154 = load <4 x float>, <4 x float>* undef
  %tmp1155 = load <4 x float>, <4 x float>* undef, align 16
  %add12220 = fadd <4 x float> %tmp1155, %tmp1154
  %tmp1156 = load <4 x float>, <4 x float>* undef, align 16
  %tmp1157 = load <4 x float>, <4 x float>* undef, align 16
  %vecins12223 = insertelement <4 x float> %tmp1157, float undef, i32 0
  store <4 x float> %vecins12223, <4 x float>* undef, align 16
  %tmp1158 = load <4 x float>, <4 x float>* undef, align 16
  %add12242 = fadd float undef, 0x4067E33340000000
  %tmp1159 = load <4 x float>, <4 x float>* undef, align 16
  %vecins12243 = insertelement <4 x float> %tmp1159, float %add12242, i32 2
  store <4 x float> %vecins12243, <4 x float>* undef, align 16
  %tmp1160 = load <4 x float>, <4 x float>* undef, align 16
  %vecext12244 = extractelement <4 x float> %tmp1160, i32 3
  %add12245 = fadd float %vecext12244, 0x4071AE6660000000
  %tmp1161 = load <4 x float>, <4 x float>* undef, align 16
  %vecins12246 = insertelement <4 x float> %tmp1161, float %add12245, i32 3
  store <4 x float> %vecins12246, <4 x float>* undef, align 16
  store <4 x float> <float -4.880000e+02, float 0xC079966660000000, float -8.450000e+01, float 0xC0464CCCC0000000>, <4 x float>* %.compoundliteral12247
  %tmp1162 = load <4 x float>, <4 x float>* %.compoundliteral12247
  %tmp1163 = load <4 x float>, <4 x float>* undef, align 16
  %add12248 = fadd <4 x float> %tmp1163, %tmp1162
  store <4 x float> %add12248, <4 x float>* undef, align 16
  %tmp1164 = load <4 x float>, <4 x float>* undef, align 16
  %vecext12249 = extractelement <4 x float> %tmp1164, i32 0
  %add12250 = fadd float %vecext12249, 1.075000e+02
  %tmp1165 = load <4 x float>, <4 x float>* undef, align 16
  store <4 x float> undef, <4 x float>* undef, align 16
  %tmp1166 = load <4 x float>, <4 x float>* undef, align 16
  %vecext12252 = extractelement <4 x float> %tmp1166, i32 1
  %add12253 = fadd float %vecext12252, 0xC0662CCCC0000000
  %tmp1167 = load <4 x float>, <4 x float>* undef, align 16
  %vecins12254 = insertelement <4 x float> %tmp1167, float %add12253, i32 1
  store <4 x float> %vecins12254, <4 x float>* undef, align 16
  %tmp1168 = load <4 x float>, <4 x float>* undef, align 16
  %vecext12255 = extractelement <4 x float> %tmp1168, i32 2
  %add12256 = fadd float %vecext12255, 0x40554CCCC0000000
  store <4 x float> undef, <4 x float>* undef, align 16
  %add13141 = fadd float undef, 0x40768999A0000000
  %tmp1169 = load <4 x float>, <4 x float>* undef, align 16
  %vecins13142 = insertelement <4 x float> %tmp1169, float %add13141, i32 3
  store <4 x float> %vecins13142, <4 x float>* undef, align 16
  %tmp1170 = load <4 x float>, <4 x float>* undef
  %add13144 = fadd <4 x float> undef, %tmp1170
  store <4 x float> %add13144, <4 x float>* undef, align 16
  %tmp1171 = load <4 x float>, <4 x float>* undef, align 16
  %vecext13145 = extractelement <4 x float> %tmp1171, i32 0
  %add13146 = fadd float %vecext13145, 3.975000e+02
  %tmp1172 = load <4 x float>, <4 x float>* undef, align 16
  %vecext13378 = extractelement <4 x float> %tmp1172, i32 3
  %add13379 = fadd float %vecext13378, 0xC053B33340000000
  %tmp1173 = load <4 x float>, <4 x float>* undef, align 16
  %vecins13380 = insertelement <4 x float> %tmp1173, float %add13379, i32 3
  store <4 x float> %vecins13380, <4 x float>* undef, align 16
  %tmp1174 = load <4 x float>, <4 x float>* undef, align 16
  %vecins13408 = insertelement <4 x float> %tmp1174, float undef, i32 3
  store <4 x float> %vecins13408, <4 x float>* undef, align 16
  store <4 x float> <float 0xC0455999A0000000, float 0xC07D366660000000, float 4.240000e+02, float -1.670000e+02>, <4 x float>* undef
  %tmp1175 = load <4 x float>, <4 x float>* undef
  %tmp1176 = load <4 x float>, <4 x float>* undef, align 16
  %add13410 = fadd <4 x float> %tmp1176, %tmp1175
  store <4 x float> %add13410, <4 x float>* undef, align 16
  %tmp1177 = load <4 x float>, <4 x float>* undef, align 16
  %add13412 = fadd float undef, 0xC0708999A0000000
  %tmp1178 = load <4 x float>, <4 x float>* undef, align 16
  %vecins13413 = insertelement <4 x float> %tmp1178, float %add13412, i32 0
  store <4 x float> undef, <4 x float>* undef, align 16
  %vecext13428 = extractelement <4 x float> undef, i32 1
  %add13429 = fadd float %vecext13428, 0xC063BCCCC0000000
  %tmp1179 = load <4 x float>, <4 x float>* undef, align 16
  %vecins13430 = insertelement <4 x float> %tmp1179, float %add13429, i32 1
  store <4 x float> %vecins13430, <4 x float>* undef, align 16
  %tmp1180 = load <4 x float>, <4 x float>* undef, align 16
  %vecext13431 = extractelement <4 x float> %tmp1180, i32 2
  %vecins13433 = insertelement <4 x float> undef, float undef, i32 2
  store <4 x float> undef, <4 x float>* undef, align 16
  %add13449 = fadd float undef, 4.590000e+02
  %tmp1181 = load <4 x float>, <4 x float>* undef, align 16
  %vecins13450 = insertelement <4 x float> %tmp1181, float %add13449, i32 3
  store <4 x float> %vecins13450, <4 x float>* undef, align 16
  store <4 x float> <float 0xC073A66660000000, float 0xC041B33340000000, float 0x4066233340000000, float 0x4071C33340000000>, <4 x float>* undef
  %tmp1182 = load <4 x float>, <4 x float>* undef
  %tmp1183 = load <4 x float>, <4 x float>* undef, align 16
  %add13452 = fadd <4 x float> %tmp1183, %tmp1182
  store <4 x float> %add13452, <4 x float>* undef, align 16
  %tmp1184 = load <4 x float>, <4 x float>* undef, align 16
  %vecext13453 = extractelement <4 x float> %tmp1184, i32 0
  %add13454 = fadd float %vecext13453, 0xC072866660000000
  %tmp1185 = load <4 x float>, <4 x float>* undef, align 16
  %vecins13455 = insertelement <4 x float> %tmp1185, float %add13454, i32 0
  %add13471 = fadd float undef, 0xC0556CCCC0000000
  %tmp1186 = load <4 x float>, <4 x float>* undef, align 16
  %vecins13472 = insertelement <4 x float> %tmp1186, float %add13471, i32 1
  store <4 x float> %vecins13472, <4 x float>* undef, align 16
  %tmp1187 = load <4 x float>, <4 x float>* undef, align 16
  %vecext13473 = extractelement <4 x float> %tmp1187, i32 2
  %add13474 = fadd float %vecext13473, 0xC0786999A0000000
  %tmp1188 = load <4 x float>, <4 x float>* undef, align 16
  %vecins13475 = insertelement <4 x float> %tmp1188, float %add13474, i32 2
  store <4 x float> %vecins13475, <4 x float>* undef, align 16
  %add13477 = fadd float undef, 0xC07C3E6660000000
  %tmp1189 = load <4 x float>, <4 x float>* undef, align 16
  %vecins13478 = insertelement <4 x float> %tmp1189, float %add13477, i32 3
  store <4 x float> %vecins13478, <4 x float>* undef, align 16
  store <4 x float> <float -4.740000e+02, float 0x4023CCCCC0000000, float 0xC05C266660000000, float 0x407B7199A0000000>, <4 x float>* undef
  %tmp1190 = load <4 x float>, <4 x float>* undef, align 16
  %add13480 = fadd <4 x float> %tmp1190, undef
  store <4 x float> %add13480, <4 x float>* undef, align 16
  %tmp1191 = load <4 x float>, <4 x float>* undef, align 16
  %vecext13481 = extractelement <4 x float> %tmp1191, i32 0
  %add13482 = fadd float %vecext13481, 0xC07BA4CCC0000000
  %tmp1192 = load <4 x float>, <4 x float>* undef, align 16
  %vecins13483 = insertelement <4 x float> %tmp1192, float %add13482, i32 0
  store <4 x float> %vecins13483, <4 x float>* undef, align 16
  %tmp1193 = load <4 x float>, <4 x float>* undef, align 16
  %add13485 = fadd float undef, 0x406B1999A0000000
  %tmp1194 = load <4 x float>, <4 x float>* undef, align 16
  %vecins13486 = insertelement <4 x float> %tmp1194, float %add13485, i32 1
  store <4 x float> %vecins13486, <4 x float>* undef, align 16
  %tmp1195 = load <4 x float>, <4 x float>* undef, align 16
  %vecext13487 = extractelement <4 x float> %tmp1195, i32 2
  %add13488 = fadd float %vecext13487, 0x40647999A0000000
  %tmp1196 = load <4 x float>, <4 x float>* undef, align 16
  %vecins13489 = insertelement <4 x float> %tmp1196, float %add13488, i32 2
  store <4 x float> %vecins13489, <4 x float>* undef, align 16
  %tmp1197 = load <4 x float>, <4 x float>* undef, align 16
  %vecext13490 = extractelement <4 x float> %tmp1197, i32 3
  %tmp1198 = load <4 x float>, <4 x float>* undef, align 16
  %vecins13492 = insertelement <4 x float> %tmp1198, float undef, i32 3
  store <4 x float> %vecins13492, <4 x float>* undef, align 16
  %tmp1199 = load <4 x float>, <4 x float>* %.compoundliteral13493
  %tmp1200 = load <4 x float>, <4 x float>* undef, align 16
  store <4 x float> undef, <4 x float>* undef, align 16
  %vecins13548 = insertelement <4 x float> undef, float undef, i32 3
  store <4 x float> <float 4.540000e+02, float 3.760000e+02, float 0x406EA33340000000, float 0x405AACCCC0000000>, <4 x float>* %.compoundliteral13549
  %tmp1201 = load <4 x float>, <4 x float>* undef, align 16
  %add13552 = fadd float undef, 3.230000e+02
  %tmp1202 = load <4 x float>, <4 x float>* undef, align 16
  %vecins13553 = insertelement <4 x float> %tmp1202, float %add13552, i32 0
  %tmp1203 = load <4 x float>, <4 x float>* undef, align 16
  %vecext13554 = extractelement <4 x float> %tmp1203, i32 1
  %tmp1204 = load <4 x float>, <4 x float>* undef, align 16
  %vecins13556 = insertelement <4 x float> %tmp1204, float undef, i32 1
  store <4 x float> %vecins13556, <4 x float>* undef, align 16
  %tmp1205 = load <4 x float>, <4 x float>* undef, align 16
  %add13558 = fadd float undef, 2.625000e+02
  %tmp1206 = load <4 x float>, <4 x float>* undef, align 16
  %vecins13559 = insertelement <4 x float> %tmp1206, float %add13558, i32 2
  store <4 x float> %vecins13559, <4 x float>* undef, align 16
  %add13575 = fadd float undef, -4.725000e+02
  %tmp1207 = load <4 x float>, <4 x float>* undef, align 16
  %vecins13576 = insertelement <4 x float> %tmp1207, float %add13575, i32 3
  store <4 x float> %vecins13576, <4 x float>* undef, align 16
  store <4 x float> <float 0x40334CCCC0000000, float 0xC0785CCCC0000000, float 0xC078D66660000000, float 3.745000e+02>, <4 x float>* undef
  %tmp1208 = load <4 x float>, <4 x float>* undef
  %tmp1209 = load <4 x float>, <4 x float>* undef, align 16
  %add13578 = fadd <4 x float> %tmp1209, %tmp1208
  store <4 x float> %add13578, <4 x float>* undef, align 16
  %tmp1210 = load <4 x float>, <4 x float>* undef, align 16
  %tmp1211 = load <4 x float>, <4 x float>* undef, align 16
  %add13592 = fadd <4 x float> %tmp1211, undef
  store <4 x float> %add13592, <4 x float>* undef, align 16
  %tmp1212 = load <4 x float>, <4 x float>* undef, align 16
  %vecext13593 = extractelement <4 x float> %tmp1212, i32 0
  %add13594 = fadd float %vecext13593, 0xC0708B3340000000
  %tmp1213 = load <4 x float>, <4 x float>* undef, align 16
  store <4 x float> undef, <4 x float>* undef, align 16
  %tmp1214 = load <4 x float>, <4 x float>* undef, align 16
  %vecext13596 = extractelement <4 x float> %tmp1214, i32 1
  %add13597 = fadd float %vecext13596, 0x40660999A0000000
  %vecins13604 = insertelement <4 x float> undef, float undef, i32 3
  store <4 x float> %vecins13604, <4 x float>* undef, align 16
  store <4 x float> <float 0x407B4999A0000000, float 0xC067F66660000000, float 0xC068F999A0000000, float 0xC079233340000000>, <4 x float>* undef
  %tmp1215 = load <4 x float>, <4 x float>* undef, align 16
  %add13606 = fadd <4 x float> %tmp1215, undef
  %tmp1216 = load <4 x float>, <4 x float>* undef, align 16
  %vecext13607 = extractelement <4 x float> %tmp1216, i32 0
  %vecins13609 = insertelement <4 x float> undef, float undef, i32 0
  %tmp1217 = load <4 x float>, <4 x float>* undef, align 16
  store <4 x float> undef, <4 x float>* undef, align 16
  %tmp1218 = load <4 x float>, <4 x float>* undef, align 16
  %add13622 = fadd float undef, -3.390000e+02
  %vecins13623 = insertelement <4 x float> undef, float %add13622, i32 0
  store <4 x float> %vecins13623, <4 x float>* undef, align 16
  %tmp1219 = load <4 x float>, <4 x float>* undef, align 16
  %vecext13624 = extractelement <4 x float> %tmp1219, i32 1
  %add13625 = fadd float %vecext13624, 0x405C3999A0000000
  %vecext13627 = extractelement <4 x float> undef, i32 2
  %add13628 = fadd float %vecext13627, 0xC067033340000000
  %tmp1220 = load <4 x float>, <4 x float>* undef, align 16
  %tmp1221 = load <4 x float>, <4 x float>* undef, align 16
  %vecext13630 = extractelement <4 x float> %tmp1221, i32 3
  %add13631 = fadd float %vecext13630, 0xC060333340000000
  %tmp1222 = load <4 x float>, <4 x float>* undef, align 16
  %vecins13632 = insertelement <4 x float> %tmp1222, float %add13631, i32 3
  store <4 x float> %vecins13632, <4 x float>* undef, align 16
  store <4 x float> <float 0x4078D66660000000, float 0x4048B33340000000, float 0x4051466660000000, float -2.965000e+02>, <4 x float>* undef
  %tmp1223 = load <4 x float>, <4 x float>* undef
  %tmp1224 = load <4 x float>, <4 x float>* undef, align 16
  %add13634 = fadd <4 x float> %tmp1224, %tmp1223
  store <4 x float> %add13634, <4 x float>* undef, align 16
  %vecext13635 = extractelement <4 x float> undef, i32 0
  %add13636 = fadd float %vecext13635, 0x406A5999A0000000
  %tmp1225 = load <4 x float>, <4 x float>* undef, align 16
  %vecins13637 = insertelement <4 x float> %tmp1225, float %add13636, i32 0
  store <4 x float> %vecins13637, <4 x float>* undef, align 16
  %tmp1226 = load <4 x float>, <4 x float>* undef, align 16
  %tmp1227 = load <4 x float>, <4 x float>* undef, align 16
  %vecins13643 = insertelement <4 x float> %tmp1227, float undef, i32 2
  store <4 x float> undef, <4 x float>* undef, align 16
  %tmp1228 = load <4 x float>, <4 x float>* undef, align 16
  %add13785 = fadd float undef, 0x4068866660000000
  %tmp1229 = load <4 x float>, <4 x float>* undef, align 16
  %vecins13786 = insertelement <4 x float> %tmp1229, float %add13785, i32 3
  store <4 x float> %vecins13786, <4 x float>* undef, align 16
  store <4 x float> <float 0x407704CCC0000000, float 0x4047B33340000000, float 0x40797B3340000000, float 0xC0652CCCC0000000>, <4 x float>* %.compoundliteral13787
  %tmp1230 = load <4 x float>, <4 x float>* undef, align 16
  %add13788 = fadd <4 x float> %tmp1230, undef
  %tmp1231 = load <4 x float>, <4 x float>* undef
  %tmp1232 = load <4 x float>, <4 x float>* undef, align 16
  %add13802 = fadd <4 x float> %tmp1232, %tmp1231
  store <4 x float> %add13802, <4 x float>* undef, align 16
  %tmp1233 = load <4 x float>, <4 x float>* undef, align 16
  %vecext13803 = extractelement <4 x float> %tmp1233, i32 0
  %add13804 = fadd float %vecext13803, -2.900000e+01
  %tmp1234 = load <4 x float>, <4 x float>* undef, align 16
  %vecins13805 = insertelement <4 x float> %tmp1234, float %add13804, i32 0
  store <4 x float> %vecins13805, <4 x float>* undef, align 16
  %tmp1235 = load <4 x float>, <4 x float>* undef, align 16
  %add13807 = fadd float undef, 6.400000e+01
  %tmp1236 = load <4 x float>, <4 x float>* undef, align 16
  %tmp1237 = load <4 x float>, <4 x float>* undef, align 16
  %vecext13809 = extractelement <4 x float> %tmp1237, i32 2
  %tmp1238 = load <4 x float>, <4 x float>* undef, align 16
  %vecext13812 = extractelement <4 x float> %tmp1238, i32 3
  %add13813 = fadd float %vecext13812, -3.615000e+02
  %vecins13814 = insertelement <4 x float> undef, float %add13813, i32 3
  store <4 x float> %vecins13814, <4 x float>* undef, align 16
  store <4 x float> <float -2.270000e+02, float -1.500000e+01, float 0x407084CCC0000000, float -1.425000e+02>, <4 x float>* undef
  %tmp1239 = load <4 x float>, <4 x float>* undef
  store <4 x float> undef, <4 x float>* undef, align 16
  %tmp1240 = load <4 x float>, <4 x float>* undef, align 16
  %vecext13817 = extractelement <4 x float> %tmp1240, i32 0
  %vecins13856 = insertelement <4 x float> undef, float undef, i32 3
  store <4 x float> %vecins13856, <4 x float>* undef, align 16
  store <4 x float> <float 0x40656CCCC0000000, float 0xC0656999A0000000, float 0x40778E6660000000, float 0x407ECE6660000000>, <4 x float>* undef
  %tmp1241 = load <4 x float>, <4 x float>* undef
  %tmp1242 = load <4 x float>, <4 x float>* undef, align 16
  store <4 x float> undef, <4 x float>* undef, align 16
  %tmp1243 = load <4 x float>, <4 x float>* undef, align 16
  %vecext13859 = extractelement <4 x float> %tmp1243, i32 0
  %tmp1244 = load <4 x float>, <4 x float>* undef, align 16
  %vecins13861 = insertelement <4 x float> %tmp1244, float undef, i32 0
  %tmp1245 = load <4 x float>, <4 x float>* undef, align 16
  %vecext13862 = extractelement <4 x float> %tmp1245, i32 1
  %add13863 = fadd float %vecext13862, -1.380000e+02
  %vecins13864 = insertelement <4 x float> undef, float %add13863, i32 1
  %vecins13867 = insertelement <4 x float> undef, float undef, i32 2
  store <4 x float> %vecins13867, <4 x float>* undef, align 16
  %tmp1246 = load <4 x float>, <4 x float>* undef, align 16
  %tmp1247 = load <4 x float>, <4 x float>* undef, align 16
  ret <4 x float> undef
}

declare i32 @printf(i8*, ...)
