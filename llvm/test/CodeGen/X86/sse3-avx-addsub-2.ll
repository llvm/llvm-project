; RUN: llc < %s -march=x86-64 -mtriple=x86_64-unknown-linux-gnu -mcpu=core2 | FileCheck %s -check-prefix=CHECK -check-prefix=SSE
; RUN: llc < %s -march=x86-64 -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7-avx | FileCheck %s -check-prefix=CHECK -check-prefix=AVX


; Verify that we correctly generate 'addsub' instructions from
; a sequence of vector extracts + float add/sub + vector inserts.

define <4 x float> @test1(<4 x float> %A, <4 x float> %B) {
  %1 = extractelement <4 x float> %A, i32 0
  %2 = extractelement <4 x float> %B, i32 0
  %sub = fsub float %1, %2
  %3 = extractelement <4 x float> %A, i32 2
  %4 = extractelement <4 x float> %B, i32 2
  %sub2 = fsub float %3, %4
  %5 = extractelement <4 x float> %A, i32 1
  %6 = extractelement <4 x float> %B, i32 1
  %add = fadd float %5, %6
  %7 = extractelement <4 x float> %A, i32 3
  %8 = extractelement <4 x float> %B, i32 3
  %add2 = fadd float %7, %8
  %vecinsert1 = insertelement <4 x float> undef, float %add, i32 1
  %vecinsert2 = insertelement <4 x float> %vecinsert1, float %add2, i32 3
  %vecinsert3 = insertelement <4 x float> %vecinsert2, float %sub, i32 0
  %vecinsert4 = insertelement <4 x float> %vecinsert3, float %sub2, i32 2
  ret <4 x float> %vecinsert4
}
; CHECK-LABEL: test1
; SSE: addsubps
; AVX: vaddsubps
; CHECK-NEXT: ret


define <4 x float> @test2(<4 x float> %A, <4 x float> %B) {
  %1 = extractelement <4 x float> %A, i32 2
  %2 = extractelement <4 x float> %B, i32 2
  %sub2 = fsub float %1, %2
  %3 = extractelement <4 x float> %A, i32 3
  %4 = extractelement <4 x float> %B, i32 3
  %add2 = fadd float %3, %4
  %vecinsert1 = insertelement <4 x float> undef, float %sub2, i32 2
  %vecinsert2 = insertelement <4 x float> %vecinsert1, float %add2, i32 3
  ret <4 x float> %vecinsert2
}
; CHECK-LABEL: test2
; SSE: addsubps
; AVX: vaddsubps
; CHECK-NEXT: ret


define <4 x float> @test3(<4 x float> %A, <4 x float> %B) {
  %1 = extractelement <4 x float> %A, i32 0
  %2 = extractelement <4 x float> %B, i32 0
  %sub = fsub float %1, %2
  %3 = extractelement <4 x float> %A, i32 3
  %4 = extractelement <4 x float> %B, i32 3
  %add = fadd float %4, %3
  %vecinsert1 = insertelement <4 x float> undef, float %sub, i32 0
  %vecinsert2 = insertelement <4 x float> %vecinsert1, float %add, i32 3
  ret <4 x float> %vecinsert2
}
; CHECK-LABEL: test3
; SSE: addsubps
; AVX: vaddsubps
; CHECK-NEXT: ret


define <4 x float> @test4(<4 x float> %A, <4 x float> %B) {
  %1 = extractelement <4 x float> %A, i32 2
  %2 = extractelement <4 x float> %B, i32 2
  %sub = fsub float %1, %2
  %3 = extractelement <4 x float> %A, i32 1
  %4 = extractelement <4 x float> %B, i32 1
  %add = fadd float %3, %4
  %vecinsert1 = insertelement <4 x float> undef, float %sub, i32 2
  %vecinsert2 = insertelement <4 x float> %vecinsert1, float %add, i32 1
  ret <4 x float> %vecinsert2
}
; CHECK-LABEL: test4
; SSE: addsubps
; AVX: vaddsubps
; CHECK-NEXT: ret


define <4 x float> @test5(<4 x float> %A, <4 x float> %B) {
  %1 = extractelement <4 x float> %A, i32 0
  %2 = extractelement <4 x float> %B, i32 0
  %sub2 = fsub float %1, %2
  %3 = extractelement <4 x float> %A, i32 1
  %4 = extractelement <4 x float> %B, i32 1
  %add2 = fadd float %3, %4
  %vecinsert1 = insertelement <4 x float> undef, float %sub2, i32 0
  %vecinsert2 = insertelement <4 x float> %vecinsert1, float %add2, i32 1
  ret <4 x float> %vecinsert2
}
; CHECK-LABEL: test5
; SSE: addsubps
; AVX: vaddsubps
; CHECK-NEXT: ret


define <4 x float> @test6(<4 x float> %A, <4 x float> %B) {
  %1 = extractelement <4 x float> %A, i32 0
  %2 = extractelement <4 x float> %B, i32 0
  %sub = fsub float %1, %2
  %3 = extractelement <4 x float> %A, i32 2
  %4 = extractelement <4 x float> %B, i32 2
  %sub2 = fsub float %3, %4
  %5 = extractelement <4 x float> %A, i32 1
  %6 = extractelement <4 x float> %B, i32 1
  %add = fadd float %5, %6
  %7 = extractelement <4 x float> %A, i32 3
  %8 = extractelement <4 x float> %B, i32 3
  %add2 = fadd float %7, %8
  %vecinsert1 = insertelement <4 x float> undef, float %add, i32 1
  %vecinsert2 = insertelement <4 x float> %vecinsert1, float %add2, i32 3
  %vecinsert3 = insertelement <4 x float> %vecinsert2, float %sub, i32 0
  %vecinsert4 = insertelement <4 x float> %vecinsert3, float %sub2, i32 2
  ret <4 x float> %vecinsert4
}
; CHECK-LABEL: test6
; SSE: addsubps
; AVX: vaddsubps
; CHECK-NEXT: ret


define <4 x double> @test7(<4 x double> %A, <4 x double> %B) {
  %1 = extractelement <4 x double> %A, i32 0
  %2 = extractelement <4 x double> %B, i32 0
  %sub = fsub double %1, %2
  %3 = extractelement <4 x double> %A, i32 2
  %4 = extractelement <4 x double> %B, i32 2
  %sub2 = fsub double %3, %4
  %5 = extractelement <4 x double> %A, i32 1
  %6 = extractelement <4 x double> %B, i32 1
  %add = fadd double %5, %6
  %7 = extractelement <4 x double> %A, i32 3
  %8 = extractelement <4 x double> %B, i32 3
  %add2 = fadd double %7, %8
  %vecinsert1 = insertelement <4 x double> undef, double %add, i32 1
  %vecinsert2 = insertelement <4 x double> %vecinsert1, double %add2, i32 3
  %vecinsert3 = insertelement <4 x double> %vecinsert2, double %sub, i32 0
  %vecinsert4 = insertelement <4 x double> %vecinsert3, double %sub2, i32 2
  ret <4 x double> %vecinsert4
}
; CHECK-LABEL: test7
; SSE: addsubpd
; SSE-NEXT: addsubpd
; AVX: vaddsubpd
; AVX-NOT: vaddsubpd
; CHECK: ret


define <2 x double> @test8(<2 x double> %A, <2 x double> %B) {
  %1 = extractelement <2 x double> %A, i32 0
  %2 = extractelement <2 x double> %B, i32 0
  %sub = fsub double %1, %2
  %3 = extractelement <2 x double> %A, i32 1
  %4 = extractelement <2 x double> %B, i32 1
  %add = fadd double %3, %4
  %vecinsert1 = insertelement <2 x double> undef, double %sub, i32 0
  %vecinsert2 = insertelement <2 x double> %vecinsert1, double %add, i32 1
  ret <2 x double> %vecinsert2
}
; CHECK-LABEL: test8
; SSE: addsubpd
; AVX: vaddsubpd
; CHECK: ret


define <8 x float> @test9(<8 x float> %A, <8 x float> %B) {
  %1 = extractelement <8 x float> %A, i32 0
  %2 = extractelement <8 x float> %B, i32 0
  %sub = fsub float %1, %2
  %3 = extractelement <8 x float> %A, i32 2
  %4 = extractelement <8 x float> %B, i32 2
  %sub2 = fsub float %3, %4
  %5 = extractelement <8 x float> %A, i32 1
  %6 = extractelement <8 x float> %B, i32 1
  %add = fadd float %5, %6
  %7 = extractelement <8 x float> %A, i32 3
  %8 = extractelement <8 x float> %B, i32 3
  %add2 = fadd float %7, %8
  %9 = extractelement <8 x float> %A, i32 4
  %10 = extractelement <8 x float> %B, i32 4
  %sub3 = fsub float %9, %10
  %11 = extractelement <8 x float> %A, i32 6
  %12 = extractelement <8 x float> %B, i32 6
  %sub4 = fsub float %11, %12
  %13 = extractelement <8 x float> %A, i32 5
  %14 = extractelement <8 x float> %B, i32 5
  %add3 = fadd float %13, %14
  %15 = extractelement <8 x float> %A, i32 7
  %16 = extractelement <8 x float> %B, i32 7
  %add4 = fadd float %15, %16
  %vecinsert1 = insertelement <8 x float> undef, float %add, i32 1
  %vecinsert2 = insertelement <8 x float> %vecinsert1, float %add2, i32 3
  %vecinsert3 = insertelement <8 x float> %vecinsert2, float %sub, i32 0
  %vecinsert4 = insertelement <8 x float> %vecinsert3, float %sub2, i32 2
  %vecinsert5 = insertelement <8 x float> %vecinsert4, float %add3, i32 5
  %vecinsert6 = insertelement <8 x float> %vecinsert5, float %add4, i32 7
  %vecinsert7 = insertelement <8 x float> %vecinsert6, float %sub3, i32 4
  %vecinsert8 = insertelement <8 x float> %vecinsert7, float %sub4, i32 6
  ret <8 x float> %vecinsert8
}
; CHECK-LABEL: test9
; SSE: addsubps
; SSE-NEXT: addsubps
; AVX: vaddsubps
; AVX-NOT: vaddsubps
; CHECK: ret


; Verify that we don't generate addsub instruction for the following
; functions.
define <4 x float> @test10(<4 x float> %A, <4 x float> %B) {
  %1 = extractelement <4 x float> %A, i32 0
  %2 = extractelement <4 x float> %B, i32 0
  %sub = fsub float %1, %2
  %vecinsert1 = insertelement <4 x float> undef, float %sub, i32 0
  ret <4 x float> %vecinsert1
}
; CHECK-LABEL: test10
; CHECK-NOT: addsubps
; CHECK: ret


define <4 x float> @test11(<4 x float> %A, <4 x float> %B) {
  %1 = extractelement <4 x float> %A, i32 2
  %2 = extractelement <4 x float> %B, i32 2
  %sub = fsub float %1, %2
  %vecinsert1 = insertelement <4 x float> undef, float %sub, i32 2
  ret <4 x float> %vecinsert1
}
; CHECK-LABEL: test11
; CHECK-NOT: addsubps
; CHECK: ret


define <4 x float> @test12(<4 x float> %A, <4 x float> %B) {
  %1 = extractelement <4 x float> %A, i32 1
  %2 = extractelement <4 x float> %B, i32 1
  %add = fadd float %1, %2
  %vecinsert1 = insertelement <4 x float> undef, float %add, i32 1
  ret <4 x float> %vecinsert1
}
; CHECK-LABEL: test12
; CHECK-NOT: addsubps
; CHECK: ret


define <4 x float> @test13(<4 x float> %A, <4 x float> %B) {
  %1 = extractelement <4 x float> %A, i32 3
  %2 = extractelement <4 x float> %B, i32 3
  %add = fadd float %1, %2
  %vecinsert1 = insertelement <4 x float> undef, float %add, i32 3
  ret <4 x float> %vecinsert1
}
; CHECK-LABEL: test13
; CHECK-NOT: addsubps
; CHECK: ret


define <4 x float> @test14(<4 x float> %A, <4 x float> %B) {
  %1 = extractelement <4 x float> %A, i32 0
  %2 = extractelement <4 x float> %B, i32 0
  %sub = fsub float %1, %2
  %3 = extractelement <4 x float> %A, i32 2
  %4 = extractelement <4 x float> %B, i32 2
  %sub2 = fsub float %3, %4
  %vecinsert1 = insertelement <4 x float> undef, float %sub, i32 0
  %vecinsert2 = insertelement <4 x float> %vecinsert1, float %sub2, i32 2
  ret <4 x float> %vecinsert2
}
; CHECK-LABEL: test14
; CHECK-NOT: addsubps
; CHECK: ret


define <4 x float> @test15(<4 x float> %A, <4 x float> %B) {
  %1 = extractelement <4 x float> %A, i32 1
  %2 = extractelement <4 x float> %B, i32 1
  %add = fadd float %1, %2
  %3 = extractelement <4 x float> %A, i32 3
  %4 = extractelement <4 x float> %B, i32 3
  %add2 = fadd float %3, %4
  %vecinsert1 = insertelement <4 x float> undef, float %add, i32 1
  %vecinsert2 = insertelement <4 x float> %vecinsert1, float %add2, i32 3
  ret <4 x float> %vecinsert2
}
; CHECK-LABEL: test15
; CHECK-NOT: addsubps
; CHECK: ret


define <4 x float> @test16(<4 x float> %A, <4 x float> %B) {
  %1 = extractelement <4 x float> %A, i32 0
  %2 = extractelement <4 x float> %B, i32 0
  %sub = fsub float %1, undef
  %3 = extractelement <4 x float> %A, i32 2
  %4 = extractelement <4 x float> %B, i32 2
  %sub2 = fsub float %3, %4
  %5 = extractelement <4 x float> %A, i32 1
  %6 = extractelement <4 x float> %B, i32 1
  %add = fadd float %5, undef
  %7 = extractelement <4 x float> %A, i32 3
  %8 = extractelement <4 x float> %B, i32 3
  %add2 = fadd float %7, %8
  %vecinsert1 = insertelement <4 x float> undef, float %add, i32 1
  %vecinsert2 = insertelement <4 x float> %vecinsert1, float %add2, i32 3
  %vecinsert3 = insertelement <4 x float> %vecinsert2, float %sub, i32 0
  %vecinsert4 = insertelement <4 x float> %vecinsert3, float %sub2, i32 2
  ret <4 x float> %vecinsert4
}
; CHECK-LABEL: test16
; CHECK-NOT: addsubps
; CHECK: ret

define <2 x float> @test_v2f32(<2 x float> %v0, <2 x float> %v1) {
  %v2 = extractelement <2 x float> %v0, i32 0
  %v3 = extractelement <2 x float> %v1, i32 0
  %v4 = extractelement <2 x float> %v0, i32 1
  %v5 = extractelement <2 x float> %v1, i32 1
  %sub = fsub float %v2, %v3
  %add = fadd float %v5, %v4
  %res0 = insertelement <2 x float> undef, float %sub, i32 0
  %res1 = insertelement <2 x float> %res0, float %add, i32 1
  ret <2 x float> %res1
}
; CHECK-LABEL: test_v2f32
; CHECK: addsubps %xmm1, %xmm0
; CHECK-NEXT: retq
