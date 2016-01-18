; RUN: llc < %s -march=x86-64 -mattr=ssse3 | FileCheck %s -check-prefix=CHECK -check-prefix=SSE
; RUN: llc < %s -march=x86-64 -mattr=avx | FileCheck %s -check-prefix=CHECK -check-prefix=AVX
; RUN: llc < %s -march=x86-64 -mattr=avx2 | FileCheck %s -check-prefix=CHECK -check-prefix=AVX2

; Verify that we correctly fold horizontal binop even in the presence of UNDEFs.

define <4 x float> @test1_undef(<4 x float> %a, <4 x float> %b) {
  %vecext = extractelement <4 x float> %a, i32 0
  %vecext1 = extractelement <4 x float> %a, i32 1
  %add = fadd float %vecext, %vecext1
  %vecinit = insertelement <4 x float> undef, float %add, i32 0
  %vecext2 = extractelement <4 x float> %a, i32 2
  %vecext3 = extractelement <4 x float> %a, i32 3
  %add4 = fadd float %vecext2, %vecext3
  %vecinit5 = insertelement <4 x float> %vecinit, float %add4, i32 1
  %vecext10 = extractelement <4 x float> %b, i32 2
  %vecext11 = extractelement <4 x float> %b, i32 3
  %add12 = fadd float %vecext10, %vecext11
  %vecinit13 = insertelement <4 x float> %vecinit5, float %add12, i32 3
  ret <4 x float> %vecinit13
}
; CHECK-LABEL: test1_undef
; SSE: haddps
; AVX: vhaddps
; AVX2: vhaddps
; CHECK-NEXT: ret


define <4 x float> @test2_undef(<4 x float> %a, <4 x float> %b) {
  %vecext = extractelement <4 x float> %a, i32 0
  %vecext1 = extractelement <4 x float> %a, i32 1
  %add = fadd float %vecext, %vecext1
  %vecinit = insertelement <4 x float> undef, float %add, i32 0
  %vecext6 = extractelement <4 x float> %b, i32 0
  %vecext7 = extractelement <4 x float> %b, i32 1
  %add8 = fadd float %vecext6, %vecext7
  %vecinit9 = insertelement <4 x float> %vecinit, float %add8, i32 2
  %vecext10 = extractelement <4 x float> %b, i32 2
  %vecext11 = extractelement <4 x float> %b, i32 3
  %add12 = fadd float %vecext10, %vecext11
  %vecinit13 = insertelement <4 x float> %vecinit9, float %add12, i32 3
  ret <4 x float> %vecinit13
}
; CHECK-LABEL: test2_undef
; SSE: haddps
; AVX: vhaddps
; AVX2: vhaddps
; CHECK-NEXT: ret


define <4 x float> @test3_undef(<4 x float> %a, <4 x float> %b) {
  %vecext = extractelement <4 x float> %a, i32 0
  %vecext1 = extractelement <4 x float> %a, i32 1
  %add = fadd float %vecext, %vecext1
  %vecinit = insertelement <4 x float> undef, float %add, i32 0
  %vecext2 = extractelement <4 x float> %a, i32 2
  %vecext3 = extractelement <4 x float> %a, i32 3
  %add4 = fadd float %vecext2, %vecext3
  %vecinit5 = insertelement <4 x float> %vecinit, float %add4, i32 1
  %vecext6 = extractelement <4 x float> %b, i32 0
  %vecext7 = extractelement <4 x float> %b, i32 1
  %add8 = fadd float %vecext6, %vecext7
  %vecinit9 = insertelement <4 x float> %vecinit5, float %add8, i32 2
  ret <4 x float> %vecinit9
}
; CHECK-LABEL: test3_undef
; SSE: haddps
; AVX: vhaddps
; AVX2: vhaddps
; CHECK-NEXT: ret


define <4 x float> @test4_undef(<4 x float> %a, <4 x float> %b) {
  %vecext = extractelement <4 x float> %a, i32 0
  %vecext1 = extractelement <4 x float> %a, i32 1
  %add = fadd float %vecext, %vecext1
  %vecinit = insertelement <4 x float> undef, float %add, i32 0
  ret <4 x float> %vecinit
}
; CHECK-LABEL: test4_undef
; CHECK-NOT: haddps
; CHECK: ret


define <2 x double> @test5_undef(<2 x double> %a, <2 x double> %b) {
  %vecext = extractelement <2 x double> %a, i32 0
  %vecext1 = extractelement <2 x double> %a, i32 1
  %add = fadd double %vecext, %vecext1
  %vecinit = insertelement <2 x double> undef, double %add, i32 0
  ret <2 x double> %vecinit
}
; CHECK-LABEL: test5_undef
; CHECK-NOT: haddpd
; CHECK: ret


define <4 x float> @test6_undef(<4 x float> %a, <4 x float> %b) {
  %vecext = extractelement <4 x float> %a, i32 0
  %vecext1 = extractelement <4 x float> %a, i32 1
  %add = fadd float %vecext, %vecext1
  %vecinit = insertelement <4 x float> undef, float %add, i32 0
  %vecext2 = extractelement <4 x float> %a, i32 2
  %vecext3 = extractelement <4 x float> %a, i32 3
  %add4 = fadd float %vecext2, %vecext3
  %vecinit5 = insertelement <4 x float> %vecinit, float %add4, i32 1
  ret <4 x float> %vecinit5
}
; CHECK-LABEL: test6_undef
; SSE: haddps
; AVX: vhaddps
; AVX2: vhaddps
; CHECK-NEXT: ret


define <4 x float> @test7_undef(<4 x float> %a, <4 x float> %b) {
  %vecext = extractelement <4 x float> %b, i32 0
  %vecext1 = extractelement <4 x float> %b, i32 1
  %add = fadd float %vecext, %vecext1
  %vecinit = insertelement <4 x float> undef, float %add, i32 2
  %vecext2 = extractelement <4 x float> %b, i32 2
  %vecext3 = extractelement <4 x float> %b, i32 3
  %add4 = fadd float %vecext2, %vecext3
  %vecinit5 = insertelement <4 x float> %vecinit, float %add4, i32 3
  ret <4 x float> %vecinit5
}
; CHECK-LABEL: test7_undef
; SSE: haddps
; AVX: vhaddps
; AVX2: vhaddps
; CHECK-NEXT: ret


define <4 x float> @test8_undef(<4 x float> %a, <4 x float> %b) {
  %vecext = extractelement <4 x float> %a, i32 0
  %vecext1 = extractelement <4 x float> %a, i32 1
  %add = fadd float %vecext, %vecext1
  %vecinit = insertelement <4 x float> undef, float %add, i32 0
  %vecext2 = extractelement <4 x float> %a, i32 2
  %vecext3 = extractelement <4 x float> %a, i32 3
  %add4 = fadd float %vecext2, %vecext3
  %vecinit5 = insertelement <4 x float> %vecinit, float %add4, i32 2
  ret <4 x float> %vecinit5
}
; CHECK-LABEL: test8_undef
; CHECK-NOT: haddps
; CHECK: ret


define <4 x float> @test9_undef(<4 x float> %a, <4 x float> %b) {
  %vecext = extractelement <4 x float> %a, i32 0
  %vecext1 = extractelement <4 x float> %a, i32 1
  %add = fadd float %vecext, %vecext1
  %vecinit = insertelement <4 x float> undef, float %add, i32 0
  %vecext2 = extractelement <4 x float> %b, i32 2
  %vecext3 = extractelement <4 x float> %b, i32 3
  %add4 = fadd float %vecext2, %vecext3
  %vecinit5 = insertelement <4 x float> %vecinit, float %add4, i32 3
  ret <4 x float> %vecinit5
}
; CHECK-LABEL: test9_undef
; CHECK: haddps
; CHECK-NEXT: ret

define <8 x float> @test10_undef(<8 x float> %a, <8 x float> %b) {
  %vecext = extractelement <8 x float> %a, i32 0
  %vecext1 = extractelement <8 x float> %a, i32 1
  %add = fadd float %vecext, %vecext1
  %vecinit = insertelement <8 x float> undef, float %add, i32 0
  %vecext2 = extractelement <8 x float> %b, i32 2
  %vecext3 = extractelement <8 x float> %b, i32 3
  %add4 = fadd float %vecext2, %vecext3
  %vecinit5 = insertelement <8 x float> %vecinit, float %add4, i32 3
  ret <8 x float> %vecinit5
}
; CHECK-LABEL: test10_undef
; SSE: haddps
; AVX: vhaddps
; AVX2: vhaddps
; CHECK-NOT: haddps
; CHECK: ret

define <8 x float> @test11_undef(<8 x float> %a, <8 x float> %b) {
  %vecext = extractelement <8 x float> %a, i32 0
  %vecext1 = extractelement <8 x float> %a, i32 1
  %add = fadd float %vecext, %vecext1
  %vecinit = insertelement <8 x float> undef, float %add, i32 0
  %vecext2 = extractelement <8 x float> %b, i32 4
  %vecext3 = extractelement <8 x float> %b, i32 5
  %add4 = fadd float %vecext2, %vecext3
  %vecinit5 = insertelement <8 x float> %vecinit, float %add4, i32 6
  ret <8 x float> %vecinit5
}
; CHECK-LABEL: test11_undef
; SSE-NOT: haddps
; AVX: vhaddps
; AVX2: vhaddps
; CHECK: ret

define <8 x float> @test12_undef(<8 x float> %a, <8 x float> %b) {
  %vecext = extractelement <8 x float> %a, i32 0
  %vecext1 = extractelement <8 x float> %a, i32 1
  %add = fadd float %vecext, %vecext1
  %vecinit = insertelement <8 x float> undef, float %add, i32 0
  %vecext2 = extractelement <8 x float> %a, i32 2
  %vecext3 = extractelement <8 x float> %a, i32 3
  %add4 = fadd float %vecext2, %vecext3
  %vecinit5 = insertelement <8 x float> %vecinit, float %add4, i32 1
  ret <8 x float> %vecinit5
}
; CHECK-LABEL: test12_undef
; SSE: haddps
; AVX: vhaddps
; AVX2: vhaddps
; CHECK-NOT: haddps
; CHECK: ret

define <8 x float> @test13_undef(<8 x float> %a, <8 x float> %b) {
  %vecext = extractelement <8 x float> %a, i32 0
  %vecext1 = extractelement <8 x float> %a, i32 1
  %add1 = fadd float %vecext, %vecext1
  %vecinit1 = insertelement <8 x float> undef, float %add1, i32 0
  %vecext2 = extractelement <8 x float> %a, i32 2
  %vecext3 = extractelement <8 x float> %a, i32 3
  %add2 = fadd float %vecext2, %vecext3
  %vecinit2 = insertelement <8 x float> %vecinit1, float %add2, i32 1
  %vecext4 = extractelement <8 x float> %a, i32 4
  %vecext5 = extractelement <8 x float> %a, i32 5
  %add3 = fadd float %vecext4, %vecext5
  %vecinit3 = insertelement <8 x float> %vecinit2, float %add3, i32 2
  %vecext6 = extractelement <8 x float> %a, i32 6
  %vecext7 = extractelement <8 x float> %a, i32 7
  %add4 = fadd float %vecext6, %vecext7
  %vecinit4 = insertelement <8 x float> %vecinit3, float %add4, i32 3
  ret <8 x float> %vecinit4
}
; CHECK-LABEL: test13_undef
; SSE: haddps
; SSE-NOT: haddps
; AVX: vhaddps
; AVX2: vhaddps
; CHECK-NOT: haddps
; CHECK: ret

define <8 x i32> @test14_undef(<8 x i32> %a, <8 x i32> %b) {
  %vecext = extractelement <8 x i32> %a, i32 0
  %vecext1 = extractelement <8 x i32> %a, i32 1
  %add = add i32 %vecext, %vecext1
  %vecinit = insertelement <8 x i32> undef, i32 %add, i32 0
  %vecext2 = extractelement <8 x i32> %b, i32 2
  %vecext3 = extractelement <8 x i32> %b, i32 3
  %add4 = add i32 %vecext2, %vecext3
  %vecinit5 = insertelement <8 x i32> %vecinit, i32 %add4, i32 3
  ret <8 x i32> %vecinit5
}
; CHECK-LABEL: test14_undef
; SSE: phaddd
; AVX: vphaddd
; AVX2: vphaddd
; CHECK-NOT: phaddd
; CHECK: ret

; On AVX2, the following sequence can be folded into a single horizontal add.
; If the Subtarget doesn't support AVX2, then we avoid emitting two packed 
; integer horizontal adds instead of two scalar adds followed by vector inserts.
define <8 x i32> @test15_undef(<8 x i32> %a, <8 x i32> %b) {
  %vecext = extractelement <8 x i32> %a, i32 0
  %vecext1 = extractelement <8 x i32> %a, i32 1
  %add = add i32 %vecext, %vecext1
  %vecinit = insertelement <8 x i32> undef, i32 %add, i32 0
  %vecext2 = extractelement <8 x i32> %b, i32 4
  %vecext3 = extractelement <8 x i32> %b, i32 5
  %add4 = add i32 %vecext2, %vecext3
  %vecinit5 = insertelement <8 x i32> %vecinit, i32 %add4, i32 6
  ret <8 x i32> %vecinit5
}
; CHECK-LABEL: test15_undef
; SSE-NOT: phaddd
; AVX-NOT: vphaddd
; AVX2: vphaddd
; CHECK: ret

define <8 x i32> @test16_undef(<8 x i32> %a, <8 x i32> %b) {
  %vecext = extractelement <8 x i32> %a, i32 0
  %vecext1 = extractelement <8 x i32> %a, i32 1
  %add = add i32 %vecext, %vecext1
  %vecinit = insertelement <8 x i32> undef, i32 %add, i32 0
  %vecext2 = extractelement <8 x i32> %a, i32 2
  %vecext3 = extractelement <8 x i32> %a, i32 3
  %add4 = add i32 %vecext2, %vecext3
  %vecinit5 = insertelement <8 x i32> %vecinit, i32 %add4, i32 1
  ret <8 x i32> %vecinit5
}
; CHECK-LABEL: test16_undef
; SSE: phaddd
; AVX: vphaddd
; AVX2: vphaddd
; CHECK-NOT: haddps
; CHECK: ret

define <8 x i32> @test17_undef(<8 x i32> %a, <8 x i32> %b) {
  %vecext = extractelement <8 x i32> %a, i32 0
  %vecext1 = extractelement <8 x i32> %a, i32 1
  %add1 = add i32 %vecext, %vecext1
  %vecinit1 = insertelement <8 x i32> undef, i32 %add1, i32 0
  %vecext2 = extractelement <8 x i32> %a, i32 2
  %vecext3 = extractelement <8 x i32> %a, i32 3
  %add2 = add i32 %vecext2, %vecext3
  %vecinit2 = insertelement <8 x i32> %vecinit1, i32 %add2, i32 1
  %vecext4 = extractelement <8 x i32> %a, i32 4
  %vecext5 = extractelement <8 x i32> %a, i32 5
  %add3 = add i32 %vecext4, %vecext5
  %vecinit3 = insertelement <8 x i32> %vecinit2, i32 %add3, i32 2
  %vecext6 = extractelement <8 x i32> %a, i32 6
  %vecext7 = extractelement <8 x i32> %a, i32 7
  %add4 = add i32 %vecext6, %vecext7
  %vecinit4 = insertelement <8 x i32> %vecinit3, i32 %add4, i32 3
  ret <8 x i32> %vecinit4
}
; CHECK-LABEL: test17_undef
; SSE: phaddd
; AVX: vphaddd
; AVX2: vphaddd
; CHECK-NOT: haddps
; CHECK: ret

