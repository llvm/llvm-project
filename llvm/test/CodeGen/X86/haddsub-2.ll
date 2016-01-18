; RUN: llc < %s -march=x86-64 -mattr=+sse2,+sse3 | FileCheck %s -check-prefix=CHECK -check-prefix=SSE3
; RUN: llc < %s -march=x86-64 -mattr=+sse2,+sse3,+ssse3 | FileCheck %s -check-prefix=CHECK -check-prefix=SSSE3
; RUN: llc < %s -march=x86-64 -mattr=+avx | FileCheck %s -check-prefix=CHECK -check-prefix=AVX
; RUN: llc < %s -march=x86-64 -mattr=+avx2 | FileCheck %s -check-prefix=CHECK -check-prefix=AVX2



define <4 x float> @hadd_ps_test1(<4 x float> %A, <4 x float> %B) {
  %vecext = extractelement <4 x float> %A, i32 0
  %vecext1 = extractelement <4 x float> %A, i32 1
  %add = fadd float %vecext, %vecext1
  %vecinit = insertelement <4 x float> undef, float %add, i32 0
  %vecext2 = extractelement <4 x float> %A, i32 2
  %vecext3 = extractelement <4 x float> %A, i32 3
  %add4 = fadd float %vecext2, %vecext3
  %vecinit5 = insertelement <4 x float> %vecinit, float %add4, i32 1
  %vecext6 = extractelement <4 x float> %B, i32 0
  %vecext7 = extractelement <4 x float> %B, i32 1
  %add8 = fadd float %vecext6, %vecext7
  %vecinit9 = insertelement <4 x float> %vecinit5, float %add8, i32 2
  %vecext10 = extractelement <4 x float> %B, i32 2
  %vecext11 = extractelement <4 x float> %B, i32 3
  %add12 = fadd float %vecext10, %vecext11
  %vecinit13 = insertelement <4 x float> %vecinit9, float %add12, i32 3
  ret <4 x float> %vecinit13
}
; CHECK-LABEL: hadd_ps_test1
; CHECK: haddps
; CHECK-NEXT: ret


define <4 x float> @hadd_ps_test2(<4 x float> %A, <4 x float> %B) {
  %vecext = extractelement <4 x float> %A, i32 2
  %vecext1 = extractelement <4 x float> %A, i32 3
  %add = fadd float %vecext, %vecext1
  %vecinit = insertelement <4 x float> undef, float %add, i32 1
  %vecext2 = extractelement <4 x float> %A, i32 0
  %vecext3 = extractelement <4 x float> %A, i32 1
  %add4 = fadd float %vecext2, %vecext3
  %vecinit5 = insertelement <4 x float> %vecinit, float %add4, i32 0
  %vecext6 = extractelement <4 x float> %B, i32 2
  %vecext7 = extractelement <4 x float> %B, i32 3
  %add8 = fadd float %vecext6, %vecext7
  %vecinit9 = insertelement <4 x float> %vecinit5, float %add8, i32 3
  %vecext10 = extractelement <4 x float> %B, i32 0
  %vecext11 = extractelement <4 x float> %B, i32 1
  %add12 = fadd float %vecext10, %vecext11
  %vecinit13 = insertelement <4 x float> %vecinit9, float %add12, i32 2
  ret <4 x float> %vecinit13
}
; CHECK-LABEL: hadd_ps_test2
; CHECK: haddps
; CHECK-NEXT: ret


define <4 x float> @hsub_ps_test1(<4 x float> %A, <4 x float> %B) {
  %vecext = extractelement <4 x float> %A, i32 0
  %vecext1 = extractelement <4 x float> %A, i32 1
  %sub = fsub float %vecext, %vecext1
  %vecinit = insertelement <4 x float> undef, float %sub, i32 0
  %vecext2 = extractelement <4 x float> %A, i32 2
  %vecext3 = extractelement <4 x float> %A, i32 3
  %sub4 = fsub float %vecext2, %vecext3
  %vecinit5 = insertelement <4 x float> %vecinit, float %sub4, i32 1
  %vecext6 = extractelement <4 x float> %B, i32 0
  %vecext7 = extractelement <4 x float> %B, i32 1
  %sub8 = fsub float %vecext6, %vecext7
  %vecinit9 = insertelement <4 x float> %vecinit5, float %sub8, i32 2
  %vecext10 = extractelement <4 x float> %B, i32 2
  %vecext11 = extractelement <4 x float> %B, i32 3
  %sub12 = fsub float %vecext10, %vecext11
  %vecinit13 = insertelement <4 x float> %vecinit9, float %sub12, i32 3
  ret <4 x float> %vecinit13
}
; CHECK-LABEL: hsub_ps_test1
; CHECK: hsubps
; CHECK-NEXT: ret


define <4 x float> @hsub_ps_test2(<4 x float> %A, <4 x float> %B) {
  %vecext = extractelement <4 x float> %A, i32 2
  %vecext1 = extractelement <4 x float> %A, i32 3
  %sub = fsub float %vecext, %vecext1
  %vecinit = insertelement <4 x float> undef, float %sub, i32 1
  %vecext2 = extractelement <4 x float> %A, i32 0
  %vecext3 = extractelement <4 x float> %A, i32 1
  %sub4 = fsub float %vecext2, %vecext3
  %vecinit5 = insertelement <4 x float> %vecinit, float %sub4, i32 0
  %vecext6 = extractelement <4 x float> %B, i32 2
  %vecext7 = extractelement <4 x float> %B, i32 3
  %sub8 = fsub float %vecext6, %vecext7
  %vecinit9 = insertelement <4 x float> %vecinit5, float %sub8, i32 3
  %vecext10 = extractelement <4 x float> %B, i32 0
  %vecext11 = extractelement <4 x float> %B, i32 1
  %sub12 = fsub float %vecext10, %vecext11
  %vecinit13 = insertelement <4 x float> %vecinit9, float %sub12, i32 2
  ret <4 x float> %vecinit13
}
; CHECK-LABEL: hsub_ps_test2
; CHECK: hsubps
; CHECK-NEXT: ret


define <4 x i32> @phadd_d_test1(<4 x i32> %A, <4 x i32> %B) {
  %vecext = extractelement <4 x i32> %A, i32 0
  %vecext1 = extractelement <4 x i32> %A, i32 1
  %add = add i32 %vecext, %vecext1
  %vecinit = insertelement <4 x i32> undef, i32 %add, i32 0
  %vecext2 = extractelement <4 x i32> %A, i32 2
  %vecext3 = extractelement <4 x i32> %A, i32 3
  %add4 = add i32 %vecext2, %vecext3
  %vecinit5 = insertelement <4 x i32> %vecinit, i32 %add4, i32 1
  %vecext6 = extractelement <4 x i32> %B, i32 0
  %vecext7 = extractelement <4 x i32> %B, i32 1
  %add8 = add i32 %vecext6, %vecext7
  %vecinit9 = insertelement <4 x i32> %vecinit5, i32 %add8, i32 2
  %vecext10 = extractelement <4 x i32> %B, i32 2
  %vecext11 = extractelement <4 x i32> %B, i32 3
  %add12 = add i32 %vecext10, %vecext11
  %vecinit13 = insertelement <4 x i32> %vecinit9, i32 %add12, i32 3
  ret <4 x i32> %vecinit13
}
; CHECK-LABEL: phadd_d_test1
; SSE3-NOT: phaddd
; SSSE3: phaddd
; AVX: vphaddd
; AVX2 vphaddd
; CHECK: ret


define <4 x i32> @phadd_d_test2(<4 x i32> %A, <4 x i32> %B) {
  %vecext = extractelement <4 x i32> %A, i32 2
  %vecext1 = extractelement <4 x i32> %A, i32 3
  %add = add i32 %vecext, %vecext1
  %vecinit = insertelement <4 x i32> undef, i32 %add, i32 1
  %vecext2 = extractelement <4 x i32> %A, i32 0
  %vecext3 = extractelement <4 x i32> %A, i32 1
  %add4 = add i32 %vecext2, %vecext3
  %vecinit5 = insertelement <4 x i32> %vecinit, i32 %add4, i32 0
  %vecext6 = extractelement <4 x i32> %B, i32 3
  %vecext7 = extractelement <4 x i32> %B, i32 2
  %add8 = add i32 %vecext6, %vecext7
  %vecinit9 = insertelement <4 x i32> %vecinit5, i32 %add8, i32 3
  %vecext10 = extractelement <4 x i32> %B, i32 1
  %vecext11 = extractelement <4 x i32> %B, i32 0
  %add12 = add i32 %vecext10, %vecext11
  %vecinit13 = insertelement <4 x i32> %vecinit9, i32 %add12, i32 2
  ret <4 x i32> %vecinit13
}
; CHECK-LABEL: phadd_d_test2
; SSE3-NOT: phaddd
; SSSE3: phaddd
; AVX: vphaddd
; AVX2 vphaddd
; CHECK: ret


define <4 x i32> @phsub_d_test1(<4 x i32> %A, <4 x i32> %B) {
  %vecext = extractelement <4 x i32> %A, i32 0
  %vecext1 = extractelement <4 x i32> %A, i32 1
  %sub = sub i32 %vecext, %vecext1
  %vecinit = insertelement <4 x i32> undef, i32 %sub, i32 0
  %vecext2 = extractelement <4 x i32> %A, i32 2
  %vecext3 = extractelement <4 x i32> %A, i32 3
  %sub4 = sub i32 %vecext2, %vecext3
  %vecinit5 = insertelement <4 x i32> %vecinit, i32 %sub4, i32 1
  %vecext6 = extractelement <4 x i32> %B, i32 0
  %vecext7 = extractelement <4 x i32> %B, i32 1
  %sub8 = sub i32 %vecext6, %vecext7
  %vecinit9 = insertelement <4 x i32> %vecinit5, i32 %sub8, i32 2
  %vecext10 = extractelement <4 x i32> %B, i32 2
  %vecext11 = extractelement <4 x i32> %B, i32 3
  %sub12 = sub i32 %vecext10, %vecext11
  %vecinit13 = insertelement <4 x i32> %vecinit9, i32 %sub12, i32 3
  ret <4 x i32> %vecinit13
}
; CHECK-LABEL: phsub_d_test1
; SSE3-NOT: phsubd
; SSSE3: phsubd
; AVX: vphsubd
; AVX2 vphsubd
; CHECK: ret


define <4 x i32> @phsub_d_test2(<4 x i32> %A, <4 x i32> %B) {
  %vecext = extractelement <4 x i32> %A, i32 2
  %vecext1 = extractelement <4 x i32> %A, i32 3
  %sub = sub i32 %vecext, %vecext1
  %vecinit = insertelement <4 x i32> undef, i32 %sub, i32 1
  %vecext2 = extractelement <4 x i32> %A, i32 0
  %vecext3 = extractelement <4 x i32> %A, i32 1
  %sub4 = sub i32 %vecext2, %vecext3
  %vecinit5 = insertelement <4 x i32> %vecinit, i32 %sub4, i32 0
  %vecext6 = extractelement <4 x i32> %B, i32 2
  %vecext7 = extractelement <4 x i32> %B, i32 3
  %sub8 = sub i32 %vecext6, %vecext7
  %vecinit9 = insertelement <4 x i32> %vecinit5, i32 %sub8, i32 3
  %vecext10 = extractelement <4 x i32> %B, i32 0
  %vecext11 = extractelement <4 x i32> %B, i32 1
  %sub12 = sub i32 %vecext10, %vecext11
  %vecinit13 = insertelement <4 x i32> %vecinit9, i32 %sub12, i32 2
  ret <4 x i32> %vecinit13
}
; CHECK-LABEL: phsub_d_test2
; SSE3-NOT: phsubd
; SSSE3: phsubd
; AVX: vphsubd
; AVX2 vphsubd
; CHECK: ret


define <2 x double> @hadd_pd_test1(<2 x double> %A, <2 x double> %B) {
  %vecext = extractelement <2 x double> %A, i32 0
  %vecext1 = extractelement <2 x double> %A, i32 1
  %add = fadd double %vecext, %vecext1
  %vecinit = insertelement <2 x double> undef, double %add, i32 0
  %vecext2 = extractelement <2 x double> %B, i32 0
  %vecext3 = extractelement <2 x double> %B, i32 1
  %add2 = fadd double %vecext2, %vecext3
  %vecinit2 = insertelement <2 x double> %vecinit, double %add2, i32 1
  ret <2 x double> %vecinit2
}
; CHECK-LABEL: hadd_pd_test1
; CHECK: haddpd
; CHECK-NEXT: ret


define <2 x double> @hadd_pd_test2(<2 x double> %A, <2 x double> %B) {
  %vecext = extractelement <2 x double> %A, i32 1
  %vecext1 = extractelement <2 x double> %A, i32 0
  %add = fadd double %vecext, %vecext1
  %vecinit = insertelement <2 x double> undef, double %add, i32 0
  %vecext2 = extractelement <2 x double> %B, i32 1
  %vecext3 = extractelement <2 x double> %B, i32 0
  %add2 = fadd double %vecext2, %vecext3
  %vecinit2 = insertelement <2 x double> %vecinit, double %add2, i32 1
  ret <2 x double> %vecinit2
}
; CHECK-LABEL: hadd_pd_test2
; CHECK: haddpd
; CHECK-NEXT: ret


define <2 x double> @hsub_pd_test1(<2 x double> %A, <2 x double> %B) {
  %vecext = extractelement <2 x double> %A, i32 0
  %vecext1 = extractelement <2 x double> %A, i32 1
  %sub = fsub double %vecext, %vecext1
  %vecinit = insertelement <2 x double> undef, double %sub, i32 0
  %vecext2 = extractelement <2 x double> %B, i32 0
  %vecext3 = extractelement <2 x double> %B, i32 1
  %sub2 = fsub double %vecext2, %vecext3
  %vecinit2 = insertelement <2 x double> %vecinit, double %sub2, i32 1
  ret <2 x double> %vecinit2
}
; CHECK-LABEL: hsub_pd_test1
; CHECK: hsubpd
; CHECK-NEXT: ret


define <2 x double> @hsub_pd_test2(<2 x double> %A, <2 x double> %B) {
  %vecext = extractelement <2 x double> %B, i32 0
  %vecext1 = extractelement <2 x double> %B, i32 1
  %sub = fsub double %vecext, %vecext1
  %vecinit = insertelement <2 x double> undef, double %sub, i32 1
  %vecext2 = extractelement <2 x double> %A, i32 0
  %vecext3 = extractelement <2 x double> %A, i32 1
  %sub2 = fsub double %vecext2, %vecext3
  %vecinit2 = insertelement <2 x double> %vecinit, double %sub2, i32 0
  ret <2 x double> %vecinit2
}
; CHECK-LABEL: hsub_pd_test2
; CHECK: hsubpd
; CHECK-NEXT: ret


define <4 x double> @avx_vhadd_pd_test(<4 x double> %A, <4 x double> %B) {
  %vecext = extractelement <4 x double> %A, i32 0
  %vecext1 = extractelement <4 x double> %A, i32 1
  %add = fadd double %vecext, %vecext1
  %vecinit = insertelement <4 x double> undef, double %add, i32 0
  %vecext2 = extractelement <4 x double> %A, i32 2
  %vecext3 = extractelement <4 x double> %A, i32 3
  %add4 = fadd double %vecext2, %vecext3
  %vecinit5 = insertelement <4 x double> %vecinit, double %add4, i32 1
  %vecext6 = extractelement <4 x double> %B, i32 0
  %vecext7 = extractelement <4 x double> %B, i32 1
  %add8 = fadd double %vecext6, %vecext7
  %vecinit9 = insertelement <4 x double> %vecinit5, double %add8, i32 2
  %vecext10 = extractelement <4 x double> %B, i32 2
  %vecext11 = extractelement <4 x double> %B, i32 3
  %add12 = fadd double %vecext10, %vecext11
  %vecinit13 = insertelement <4 x double> %vecinit9, double %add12, i32 3
  ret <4 x double> %vecinit13
}
; CHECK-LABEL: avx_vhadd_pd_test
; SSE3: haddpd
; SSE3-NEXT: haddpd
; SSSE3: haddpd
; SSSE3: haddpd
; AVX: vhaddpd
; AVX: vhaddpd
; AVX2: vhaddpd
; AVX2: vhaddpd
; CHECK: ret


define <4 x double> @avx_vhsub_pd_test(<4 x double> %A, <4 x double> %B) {
  %vecext = extractelement <4 x double> %A, i32 0
  %vecext1 = extractelement <4 x double> %A, i32 1
  %sub = fsub double %vecext, %vecext1
  %vecinit = insertelement <4 x double> undef, double %sub, i32 0
  %vecext2 = extractelement <4 x double> %A, i32 2
  %vecext3 = extractelement <4 x double> %A, i32 3
  %sub4 = fsub double %vecext2, %vecext3
  %vecinit5 = insertelement <4 x double> %vecinit, double %sub4, i32 1
  %vecext6 = extractelement <4 x double> %B, i32 0
  %vecext7 = extractelement <4 x double> %B, i32 1
  %sub8 = fsub double %vecext6, %vecext7
  %vecinit9 = insertelement <4 x double> %vecinit5, double %sub8, i32 2
  %vecext10 = extractelement <4 x double> %B, i32 2
  %vecext11 = extractelement <4 x double> %B, i32 3
  %sub12 = fsub double %vecext10, %vecext11
  %vecinit13 = insertelement <4 x double> %vecinit9, double %sub12, i32 3
  ret <4 x double> %vecinit13
}
; CHECK-LABEL: avx_vhsub_pd_test
; SSE3: hsubpd
; SSE3-NEXT: hsubpd
; SSSE3: hsubpd
; SSSE3-NEXT: hsubpd
; AVX: vhsubpd
; AVX: vhsubpd
; AVX2: vhsubpd
; AVX2: vhsubpd
; CHECK: ret


define <8 x i32> @avx2_vphadd_d_test(<8 x i32> %A, <8 x i32> %B) {
  %vecext = extractelement <8 x i32> %A, i32 0
  %vecext1 = extractelement <8 x i32> %A, i32 1
  %add = add i32 %vecext, %vecext1
  %vecinit = insertelement <8 x i32> undef, i32 %add, i32 0
  %vecext2 = extractelement <8 x i32> %A, i32 2
  %vecext3 = extractelement <8 x i32> %A, i32 3
  %add4 = add i32 %vecext2, %vecext3
  %vecinit5 = insertelement <8 x i32> %vecinit, i32 %add4, i32 1
  %vecext6 = extractelement <8 x i32> %A, i32 4
  %vecext7 = extractelement <8 x i32> %A, i32 5
  %add8 = add i32 %vecext6, %vecext7
  %vecinit9 = insertelement <8 x i32> %vecinit5, i32 %add8, i32 2
  %vecext10 = extractelement <8 x i32> %A, i32 6
  %vecext11 = extractelement <8 x i32> %A, i32 7
  %add12 = add i32 %vecext10, %vecext11
  %vecinit13 = insertelement <8 x i32> %vecinit9, i32 %add12, i32 3
  %vecext14 = extractelement <8 x i32> %B, i32 0
  %vecext15 = extractelement <8 x i32> %B, i32 1
  %add16 = add i32 %vecext14, %vecext15
  %vecinit17 = insertelement <8 x i32> %vecinit13, i32 %add16, i32 4
  %vecext18 = extractelement <8 x i32> %B, i32 2
  %vecext19 = extractelement <8 x i32> %B, i32 3
  %add20 = add i32 %vecext18, %vecext19
  %vecinit21 = insertelement <8 x i32> %vecinit17, i32 %add20, i32 5
  %vecext22 = extractelement <8 x i32> %B, i32 4
  %vecext23 = extractelement <8 x i32> %B, i32 5
  %add24 = add i32 %vecext22, %vecext23
  %vecinit25 = insertelement <8 x i32> %vecinit21, i32 %add24, i32 6
  %vecext26 = extractelement <8 x i32> %B, i32 6
  %vecext27 = extractelement <8 x i32> %B, i32 7
  %add28 = add i32 %vecext26, %vecext27
  %vecinit29 = insertelement <8 x i32> %vecinit25, i32 %add28, i32 7
  ret <8 x i32> %vecinit29
}
; CHECK-LABEL: avx2_vphadd_d_test
; SSE3-NOT: phaddd
; SSSE3: phaddd
; SSSE3-NEXT: phaddd
; AVX: vphaddd
; AVX: vphaddd
; AVX2: vphaddd
; AVX2: vphaddd
; CHECK: ret

define <16 x i16> @avx2_vphadd_w_test(<16 x i16> %a, <16 x i16> %b) {
  %vecext = extractelement <16 x i16> %a, i32 0
  %vecext1 = extractelement <16 x i16> %a, i32 1
  %add = add i16 %vecext, %vecext1
  %vecinit = insertelement <16 x i16> undef, i16 %add, i32 0
  %vecext4 = extractelement <16 x i16> %a, i32 2
  %vecext6 = extractelement <16 x i16> %a, i32 3
  %add8 = add i16 %vecext4, %vecext6
  %vecinit10 = insertelement <16 x i16> %vecinit, i16 %add8, i32 1
  %vecext11 = extractelement <16 x i16> %a, i32 4
  %vecext13 = extractelement <16 x i16> %a, i32 5
  %add15 = add i16 %vecext11, %vecext13
  %vecinit17 = insertelement <16 x i16> %vecinit10, i16 %add15, i32 2
  %vecext18 = extractelement <16 x i16> %a, i32 6
  %vecext20 = extractelement <16 x i16> %a, i32 7
  %add22 = add i16 %vecext18, %vecext20
  %vecinit24 = insertelement <16 x i16> %vecinit17, i16 %add22, i32 3
  %vecext25 = extractelement <16 x i16> %a, i32 8
  %vecext27 = extractelement <16 x i16> %a, i32 9
  %add29 = add i16 %vecext25, %vecext27
  %vecinit31 = insertelement <16 x i16> %vecinit24, i16 %add29, i32 4
  %vecext32 = extractelement <16 x i16> %a, i32 10
  %vecext34 = extractelement <16 x i16> %a, i32 11
  %add36 = add i16 %vecext32, %vecext34
  %vecinit38 = insertelement <16 x i16> %vecinit31, i16 %add36, i32 5
  %vecext39 = extractelement <16 x i16> %a, i32 12
  %vecext41 = extractelement <16 x i16> %a, i32 13
  %add43 = add i16 %vecext39, %vecext41
  %vecinit45 = insertelement <16 x i16> %vecinit38, i16 %add43, i32 6
  %vecext46 = extractelement <16 x i16> %a, i32 14
  %vecext48 = extractelement <16 x i16> %a, i32 15
  %add50 = add i16 %vecext46, %vecext48
  %vecinit52 = insertelement <16 x i16> %vecinit45, i16 %add50, i32 7
  %vecext53 = extractelement <16 x i16> %b, i32 0
  %vecext55 = extractelement <16 x i16> %b, i32 1
  %add57 = add i16 %vecext53, %vecext55
  %vecinit59 = insertelement <16 x i16> %vecinit52, i16 %add57, i32 8
  %vecext60 = extractelement <16 x i16> %b, i32 2
  %vecext62 = extractelement <16 x i16> %b, i32 3
  %add64 = add i16 %vecext60, %vecext62
  %vecinit66 = insertelement <16 x i16> %vecinit59, i16 %add64, i32 9
  %vecext67 = extractelement <16 x i16> %b, i32 4
  %vecext69 = extractelement <16 x i16> %b, i32 5
  %add71 = add i16 %vecext67, %vecext69
  %vecinit73 = insertelement <16 x i16> %vecinit66, i16 %add71, i32 10
  %vecext74 = extractelement <16 x i16> %b, i32 6
  %vecext76 = extractelement <16 x i16> %b, i32 7
  %add78 = add i16 %vecext74, %vecext76
  %vecinit80 = insertelement <16 x i16> %vecinit73, i16 %add78, i32 11
  %vecext81 = extractelement <16 x i16> %b, i32 8
  %vecext83 = extractelement <16 x i16> %b, i32 9
  %add85 = add i16 %vecext81, %vecext83
  %vecinit87 = insertelement <16 x i16> %vecinit80, i16 %add85, i32 12
  %vecext88 = extractelement <16 x i16> %b, i32 10
  %vecext90 = extractelement <16 x i16> %b, i32 11
  %add92 = add i16 %vecext88, %vecext90
  %vecinit94 = insertelement <16 x i16> %vecinit87, i16 %add92, i32 13
  %vecext95 = extractelement <16 x i16> %b, i32 12
  %vecext97 = extractelement <16 x i16> %b, i32 13
  %add99 = add i16 %vecext95, %vecext97
  %vecinit101 = insertelement <16 x i16> %vecinit94, i16 %add99, i32 14
  %vecext102 = extractelement <16 x i16> %b, i32 14
  %vecext104 = extractelement <16 x i16> %b, i32 15
  %add106 = add i16 %vecext102, %vecext104
  %vecinit108 = insertelement <16 x i16> %vecinit101, i16 %add106, i32 15
  ret <16 x i16> %vecinit108
}
; CHECK-LABEL: avx2_vphadd_w_test
; SSE3-NOT: phaddw
; SSSE3: phaddw
; SSSE3-NEXT: phaddw
; AVX: vphaddw
; AVX: vphaddw
; AVX2: vphaddw
; AVX2: vphaddw
; CHECK: ret


; Verify that we don't select horizontal subs in the following functions.

define <4 x i32> @not_a_hsub_1(<4 x i32> %A, <4 x i32> %B) {
  %vecext = extractelement <4 x i32> %A, i32 0
  %vecext1 = extractelement <4 x i32> %A, i32 1
  %sub = sub i32 %vecext, %vecext1
  %vecinit = insertelement <4 x i32> undef, i32 %sub, i32 0
  %vecext2 = extractelement <4 x i32> %A, i32 2
  %vecext3 = extractelement <4 x i32> %A, i32 3
  %sub4 = sub i32 %vecext2, %vecext3
  %vecinit5 = insertelement <4 x i32> %vecinit, i32 %sub4, i32 1
  %vecext6 = extractelement <4 x i32> %B, i32 1
  %vecext7 = extractelement <4 x i32> %B, i32 0
  %sub8 = sub i32 %vecext6, %vecext7
  %vecinit9 = insertelement <4 x i32> %vecinit5, i32 %sub8, i32 2
  %vecext10 = extractelement <4 x i32> %B, i32 3
  %vecext11 = extractelement <4 x i32> %B, i32 2
  %sub12 = sub i32 %vecext10, %vecext11
  %vecinit13 = insertelement <4 x i32> %vecinit9, i32 %sub12, i32 3
  ret <4 x i32> %vecinit13
}
; CHECK-LABEL: not_a_hsub_1
; CHECK-NOT: phsubd
; CHECK: ret


define <4 x float> @not_a_hsub_2(<4 x float> %A, <4 x float> %B) {
  %vecext = extractelement <4 x float> %A, i32 2
  %vecext1 = extractelement <4 x float> %A, i32 3
  %sub = fsub float %vecext, %vecext1
  %vecinit = insertelement <4 x float> undef, float %sub, i32 1
  %vecext2 = extractelement <4 x float> %A, i32 0
  %vecext3 = extractelement <4 x float> %A, i32 1
  %sub4 = fsub float %vecext2, %vecext3
  %vecinit5 = insertelement <4 x float> %vecinit, float %sub4, i32 0
  %vecext6 = extractelement <4 x float> %B, i32 3
  %vecext7 = extractelement <4 x float> %B, i32 2
  %sub8 = fsub float %vecext6, %vecext7
  %vecinit9 = insertelement <4 x float> %vecinit5, float %sub8, i32 3
  %vecext10 = extractelement <4 x float> %B, i32 0
  %vecext11 = extractelement <4 x float> %B, i32 1
  %sub12 = fsub float %vecext10, %vecext11
  %vecinit13 = insertelement <4 x float> %vecinit9, float %sub12, i32 2
  ret <4 x float> %vecinit13
}
; CHECK-LABEL: not_a_hsub_2
; CHECK-NOT: hsubps
; CHECK: ret


define <2 x double> @not_a_hsub_3(<2 x double> %A, <2 x double> %B) {
  %vecext = extractelement <2 x double> %B, i32 0
  %vecext1 = extractelement <2 x double> %B, i32 1
  %sub = fsub double %vecext, %vecext1
  %vecinit = insertelement <2 x double> undef, double %sub, i32 1
  %vecext2 = extractelement <2 x double> %A, i32 1
  %vecext3 = extractelement <2 x double> %A, i32 0
  %sub2 = fsub double %vecext2, %vecext3
  %vecinit2 = insertelement <2 x double> %vecinit, double %sub2, i32 0
  ret <2 x double> %vecinit2
}
; CHECK-LABEL: not_a_hsub_3
; CHECK-NOT: hsubpd
; CHECK: ret


; Test AVX horizontal add/sub of packed single/double precision
; floating point values from 256-bit vectors.

define <8 x float> @avx_vhadd_ps(<8 x float> %a, <8 x float> %b) {
  %vecext = extractelement <8 x float> %a, i32 0
  %vecext1 = extractelement <8 x float> %a, i32 1
  %add = fadd float %vecext, %vecext1
  %vecinit = insertelement <8 x float> undef, float %add, i32 0
  %vecext2 = extractelement <8 x float> %a, i32 2
  %vecext3 = extractelement <8 x float> %a, i32 3
  %add4 = fadd float %vecext2, %vecext3
  %vecinit5 = insertelement <8 x float> %vecinit, float %add4, i32 1
  %vecext6 = extractelement <8 x float> %b, i32 0
  %vecext7 = extractelement <8 x float> %b, i32 1
  %add8 = fadd float %vecext6, %vecext7
  %vecinit9 = insertelement <8 x float> %vecinit5, float %add8, i32 2
  %vecext10 = extractelement <8 x float> %b, i32 2
  %vecext11 = extractelement <8 x float> %b, i32 3
  %add12 = fadd float %vecext10, %vecext11
  %vecinit13 = insertelement <8 x float> %vecinit9, float %add12, i32 3
  %vecext14 = extractelement <8 x float> %a, i32 4
  %vecext15 = extractelement <8 x float> %a, i32 5
  %add16 = fadd float %vecext14, %vecext15
  %vecinit17 = insertelement <8 x float> %vecinit13, float %add16, i32 4
  %vecext18 = extractelement <8 x float> %a, i32 6
  %vecext19 = extractelement <8 x float> %a, i32 7
  %add20 = fadd float %vecext18, %vecext19
  %vecinit21 = insertelement <8 x float> %vecinit17, float %add20, i32 5
  %vecext22 = extractelement <8 x float> %b, i32 4
  %vecext23 = extractelement <8 x float> %b, i32 5
  %add24 = fadd float %vecext22, %vecext23
  %vecinit25 = insertelement <8 x float> %vecinit21, float %add24, i32 6
  %vecext26 = extractelement <8 x float> %b, i32 6
  %vecext27 = extractelement <8 x float> %b, i32 7
  %add28 = fadd float %vecext26, %vecext27
  %vecinit29 = insertelement <8 x float> %vecinit25, float %add28, i32 7
  ret <8 x float> %vecinit29
}
; CHECK-LABEL: avx_vhadd_ps
; SSE3: haddps
; SSE3-NEXT: haddps
; SSSE3: haddps
; SSSE3-NEXT: haddps
; AVX: vhaddps
; AVX2: vhaddps
; CHECK: ret


define <8 x float> @avx_vhsub_ps(<8 x float> %a, <8 x float> %b) {
  %vecext = extractelement <8 x float> %a, i32 0
  %vecext1 = extractelement <8 x float> %a, i32 1
  %sub = fsub float %vecext, %vecext1
  %vecinit = insertelement <8 x float> undef, float %sub, i32 0
  %vecext2 = extractelement <8 x float> %a, i32 2
  %vecext3 = extractelement <8 x float> %a, i32 3
  %sub4 = fsub float %vecext2, %vecext3
  %vecinit5 = insertelement <8 x float> %vecinit, float %sub4, i32 1
  %vecext6 = extractelement <8 x float> %b, i32 0
  %vecext7 = extractelement <8 x float> %b, i32 1
  %sub8 = fsub float %vecext6, %vecext7
  %vecinit9 = insertelement <8 x float> %vecinit5, float %sub8, i32 2
  %vecext10 = extractelement <8 x float> %b, i32 2
  %vecext11 = extractelement <8 x float> %b, i32 3
  %sub12 = fsub float %vecext10, %vecext11
  %vecinit13 = insertelement <8 x float> %vecinit9, float %sub12, i32 3
  %vecext14 = extractelement <8 x float> %a, i32 4
  %vecext15 = extractelement <8 x float> %a, i32 5
  %sub16 = fsub float %vecext14, %vecext15
  %vecinit17 = insertelement <8 x float> %vecinit13, float %sub16, i32 4
  %vecext18 = extractelement <8 x float> %a, i32 6
  %vecext19 = extractelement <8 x float> %a, i32 7
  %sub20 = fsub float %vecext18, %vecext19
  %vecinit21 = insertelement <8 x float> %vecinit17, float %sub20, i32 5
  %vecext22 = extractelement <8 x float> %b, i32 4
  %vecext23 = extractelement <8 x float> %b, i32 5
  %sub24 = fsub float %vecext22, %vecext23
  %vecinit25 = insertelement <8 x float> %vecinit21, float %sub24, i32 6
  %vecext26 = extractelement <8 x float> %b, i32 6
  %vecext27 = extractelement <8 x float> %b, i32 7
  %sub28 = fsub float %vecext26, %vecext27
  %vecinit29 = insertelement <8 x float> %vecinit25, float %sub28, i32 7
  ret <8 x float> %vecinit29
}
; CHECK-LABEL: avx_vhsub_ps
; SSE3: hsubps
; SSE3-NEXT: hsubps
; SSSE3: hsubps
; SSSE3-NEXT: hsubps
; AVX: vhsubps
; AVX2: vhsubps
; CHECK: ret


define <4 x double> @avx_hadd_pd(<4 x double> %a, <4 x double> %b) {
  %vecext = extractelement <4 x double> %a, i32 0
  %vecext1 = extractelement <4 x double> %a, i32 1
  %add = fadd double %vecext, %vecext1
  %vecinit = insertelement <4 x double> undef, double %add, i32 0
  %vecext2 = extractelement <4 x double> %b, i32 0
  %vecext3 = extractelement <4 x double> %b, i32 1
  %add4 = fadd double %vecext2, %vecext3
  %vecinit5 = insertelement <4 x double> %vecinit, double %add4, i32 1
  %vecext6 = extractelement <4 x double> %a, i32 2
  %vecext7 = extractelement <4 x double> %a, i32 3
  %add8 = fadd double %vecext6, %vecext7
  %vecinit9 = insertelement <4 x double> %vecinit5, double %add8, i32 2
  %vecext10 = extractelement <4 x double> %b, i32 2
  %vecext11 = extractelement <4 x double> %b, i32 3
  %add12 = fadd double %vecext10, %vecext11
  %vecinit13 = insertelement <4 x double> %vecinit9, double %add12, i32 3
  ret <4 x double> %vecinit13
}
; CHECK-LABEL: avx_hadd_pd
; SSE3: haddpd
; SSE3-NEXT: haddpd
; SSSE3: haddpd
; SSSE3-NEXT: haddpd
; AVX: vhaddpd
; AVX2: vhaddpd
; CHECK: ret


define <4 x double> @avx_hsub_pd(<4 x double> %a, <4 x double> %b) {
  %vecext = extractelement <4 x double> %a, i32 0
  %vecext1 = extractelement <4 x double> %a, i32 1
  %sub = fsub double %vecext, %vecext1
  %vecinit = insertelement <4 x double> undef, double %sub, i32 0
  %vecext2 = extractelement <4 x double> %b, i32 0
  %vecext3 = extractelement <4 x double> %b, i32 1
  %sub4 = fsub double %vecext2, %vecext3
  %vecinit5 = insertelement <4 x double> %vecinit, double %sub4, i32 1
  %vecext6 = extractelement <4 x double> %a, i32 2
  %vecext7 = extractelement <4 x double> %a, i32 3
  %sub8 = fsub double %vecext6, %vecext7
  %vecinit9 = insertelement <4 x double> %vecinit5, double %sub8, i32 2
  %vecext10 = extractelement <4 x double> %b, i32 2
  %vecext11 = extractelement <4 x double> %b, i32 3
  %sub12 = fsub double %vecext10, %vecext11
  %vecinit13 = insertelement <4 x double> %vecinit9, double %sub12, i32 3
  ret <4 x double> %vecinit13
}
; CHECK-LABEL: avx_hsub_pd
; SSE3: hsubpd
; SSE3-NEXT: hsubpd
; SSSE3: hsubpd
; SSSE3-NEXT: hsubpd
; AVX: vhsubpd
; AVX2: vhsubpd
; CHECK: ret


; Test AVX2 horizontal add of packed integer values from 256-bit vectors.

define <8 x i32> @avx2_hadd_d(<8 x i32> %a, <8 x i32> %b) {
  %vecext = extractelement <8 x i32> %a, i32 0
  %vecext1 = extractelement <8 x i32> %a, i32 1
  %add = add i32 %vecext, %vecext1
  %vecinit = insertelement <8 x i32> undef, i32 %add, i32 0
  %vecext2 = extractelement <8 x i32> %a, i32 2
  %vecext3 = extractelement <8 x i32> %a, i32 3
  %add4 = add i32 %vecext2, %vecext3
  %vecinit5 = insertelement <8 x i32> %vecinit, i32 %add4, i32 1
  %vecext6 = extractelement <8 x i32> %b, i32 0
  %vecext7 = extractelement <8 x i32> %b, i32 1
  %add8 = add i32 %vecext6, %vecext7
  %vecinit9 = insertelement <8 x i32> %vecinit5, i32 %add8, i32 2
  %vecext10 = extractelement <8 x i32> %b, i32 2
  %vecext11 = extractelement <8 x i32> %b, i32 3
  %add12 = add i32 %vecext10, %vecext11
  %vecinit13 = insertelement <8 x i32> %vecinit9, i32 %add12, i32 3
  %vecext14 = extractelement <8 x i32> %a, i32 4
  %vecext15 = extractelement <8 x i32> %a, i32 5
  %add16 = add i32 %vecext14, %vecext15
  %vecinit17 = insertelement <8 x i32> %vecinit13, i32 %add16, i32 4
  %vecext18 = extractelement <8 x i32> %a, i32 6
  %vecext19 = extractelement <8 x i32> %a, i32 7
  %add20 = add i32 %vecext18, %vecext19
  %vecinit21 = insertelement <8 x i32> %vecinit17, i32 %add20, i32 5
  %vecext22 = extractelement <8 x i32> %b, i32 4
  %vecext23 = extractelement <8 x i32> %b, i32 5
  %add24 = add i32 %vecext22, %vecext23
  %vecinit25 = insertelement <8 x i32> %vecinit21, i32 %add24, i32 6
  %vecext26 = extractelement <8 x i32> %b, i32 6
  %vecext27 = extractelement <8 x i32> %b, i32 7
  %add28 = add i32 %vecext26, %vecext27
  %vecinit29 = insertelement <8 x i32> %vecinit25, i32 %add28, i32 7
  ret <8 x i32> %vecinit29
}
; CHECK-LABEL: avx2_hadd_d
; SSE3-NOT: phaddd
; SSSE3: phaddd
; SSSE3-NEXT: phaddd
; AVX: vphaddd
; AVX: vphaddd
; AVX2: vphaddd
; AVX2-NOT: vphaddd
; CHECK: ret


define <16 x i16> @avx2_hadd_w(<16 x i16> %a, <16 x i16> %b) {
  %vecext = extractelement <16 x i16> %a, i32 0
  %vecext1 = extractelement <16 x i16> %a, i32 1
  %add = add i16 %vecext, %vecext1
  %vecinit = insertelement <16 x i16> undef, i16 %add, i32 0
  %vecext4 = extractelement <16 x i16> %a, i32 2
  %vecext6 = extractelement <16 x i16> %a, i32 3
  %add8 = add i16 %vecext4, %vecext6
  %vecinit10 = insertelement <16 x i16> %vecinit, i16 %add8, i32 1
  %vecext11 = extractelement <16 x i16> %a, i32 4
  %vecext13 = extractelement <16 x i16> %a, i32 5
  %add15 = add i16 %vecext11, %vecext13
  %vecinit17 = insertelement <16 x i16> %vecinit10, i16 %add15, i32 2
  %vecext18 = extractelement <16 x i16> %a, i32 6
  %vecext20 = extractelement <16 x i16> %a, i32 7
  %add22 = add i16 %vecext18, %vecext20
  %vecinit24 = insertelement <16 x i16> %vecinit17, i16 %add22, i32 3
  %vecext25 = extractelement <16 x i16> %a, i32 8
  %vecext27 = extractelement <16 x i16> %a, i32 9
  %add29 = add i16 %vecext25, %vecext27
  %vecinit31 = insertelement <16 x i16> %vecinit24, i16 %add29, i32 8
  %vecext32 = extractelement <16 x i16> %a, i32 10
  %vecext34 = extractelement <16 x i16> %a, i32 11
  %add36 = add i16 %vecext32, %vecext34
  %vecinit38 = insertelement <16 x i16> %vecinit31, i16 %add36, i32 9
  %vecext39 = extractelement <16 x i16> %a, i32 12
  %vecext41 = extractelement <16 x i16> %a, i32 13
  %add43 = add i16 %vecext39, %vecext41
  %vecinit45 = insertelement <16 x i16> %vecinit38, i16 %add43, i32 10
  %vecext46 = extractelement <16 x i16> %a, i32 14
  %vecext48 = extractelement <16 x i16> %a, i32 15
  %add50 = add i16 %vecext46, %vecext48
  %vecinit52 = insertelement <16 x i16> %vecinit45, i16 %add50, i32 11
  %vecext53 = extractelement <16 x i16> %b, i32 0
  %vecext55 = extractelement <16 x i16> %b, i32 1
  %add57 = add i16 %vecext53, %vecext55
  %vecinit59 = insertelement <16 x i16> %vecinit52, i16 %add57, i32 4
  %vecext60 = extractelement <16 x i16> %b, i32 2
  %vecext62 = extractelement <16 x i16> %b, i32 3
  %add64 = add i16 %vecext60, %vecext62
  %vecinit66 = insertelement <16 x i16> %vecinit59, i16 %add64, i32 5
  %vecext67 = extractelement <16 x i16> %b, i32 4
  %vecext69 = extractelement <16 x i16> %b, i32 5
  %add71 = add i16 %vecext67, %vecext69
  %vecinit73 = insertelement <16 x i16> %vecinit66, i16 %add71, i32 6
  %vecext74 = extractelement <16 x i16> %b, i32 6
  %vecext76 = extractelement <16 x i16> %b, i32 7
  %add78 = add i16 %vecext74, %vecext76
  %vecinit80 = insertelement <16 x i16> %vecinit73, i16 %add78, i32 7
  %vecext81 = extractelement <16 x i16> %b, i32 8
  %vecext83 = extractelement <16 x i16> %b, i32 9
  %add85 = add i16 %vecext81, %vecext83
  %vecinit87 = insertelement <16 x i16> %vecinit80, i16 %add85, i32 12
  %vecext88 = extractelement <16 x i16> %b, i32 10
  %vecext90 = extractelement <16 x i16> %b, i32 11
  %add92 = add i16 %vecext88, %vecext90
  %vecinit94 = insertelement <16 x i16> %vecinit87, i16 %add92, i32 13
  %vecext95 = extractelement <16 x i16> %b, i32 12
  %vecext97 = extractelement <16 x i16> %b, i32 13
  %add99 = add i16 %vecext95, %vecext97
  %vecinit101 = insertelement <16 x i16> %vecinit94, i16 %add99, i32 14
  %vecext102 = extractelement <16 x i16> %b, i32 14
  %vecext104 = extractelement <16 x i16> %b, i32 15
  %add106 = add i16 %vecext102, %vecext104
  %vecinit108 = insertelement <16 x i16> %vecinit101, i16 %add106, i32 15
  ret <16 x i16> %vecinit108
}
; CHECK-LABEL: avx2_hadd_w
; SSE3-NOT: phaddw
; SSSE3: phaddw
; SSSE3-NEXT: phaddw
; AVX: vphaddw
; AVX: vphaddw
; AVX2: vphaddw
; AVX2-NOT: vphaddw
; CHECK: ret

