; RUN: llc -mtriple=x86_64-pc-windows-msvc < %s | FileCheck %s

; Test 1st and 2nd arguments passed in XMM0 and XMM1.
; Test 7nd argument passed by reference in stack: 56(%rsp).
define x86_vectorcallcc <4 x float> @test_m128_7(<4 x float> %a, <4 x float> %b, <4 x float> %c, <4 x float> %d, <4 x float> %e, <4 x float> %f, <4 x float> %g) #0 {
  ; CHECK-LABEL: test_m128_7@@112:
  ; CHECK: movq 56(%rsp), %rax
  ; CHECK: vaddps %xmm1, %xmm0, %xmm0
  ; CHECK: vsubps (%rax), %xmm0, %xmm0
  %add.i = fadd <4 x float> %a, %b
  %sub.i = fsub <4 x float> %add.i, %g
  ret <4 x float> %sub.i
}

; Test 1st and 2nd arguments passed in YMM0 and YMM1.
; Test 7nd argument passed by reference in stack: 56(%rsp).
define x86_vectorcallcc <8 x float> @test_m256_7(<8 x float> %a, <8 x float> %b, <8 x float> %c, <8 x float> %d, <8 x float> %e, <8 x float> %f, <8 x float> %g) #0 {
  ; CHECK-LABEL: test_m256_7@@224:
  ; CHECK: movq 56(%rsp), %rax
  ; CHECK: vaddps %ymm1, %ymm0, %ymm0
  ; CHECK: vsubps (%rax), %ymm0, %ymm0
  %add.i = fadd <8 x float> %a, %b
  %sub.i = fsub <8 x float> %add.i, %g
  ret <8 x float> %sub.i
}

attributes #0 = { nounwind "target-cpu"="core-avx2" }
