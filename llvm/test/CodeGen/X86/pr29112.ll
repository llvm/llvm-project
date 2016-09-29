; RUN: llc < %s -march=x86-64 -mattr=+avx512f | FileCheck %s

declare <4 x float> @foo(<4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>)

; In AVX512 without VLX we can't spill XMM16-31 with vmovaps as its not available. Instead we need to use vextractf32x4 to spill since it can encode the ZMM super register and can store the lower 128-bits.

define <4 x float> @bar(<4 x float>* %a1p, <4 x float>* %a2p, <4 x float> %a3, <4 x float> %a4, <16 x float>%c1, <16 x float>%c2) {
; CHECK: vextractf32x4	$0, %zmm16, {{[0-9]+}}(%rsp) {{.*#+}} 16-byte Folded Spill
  %a1 = shufflevector <16 x float>%c1, <16 x float>%c2, <4 x i32> <i32 4, i32 20, i32 1, i32 17>

  %a2 = shufflevector <16 x float>%c1, <16 x float>%c2, <4 x i32> <i32 4, i32 21, i32 1, i32 17>
  %a5 = shufflevector <16 x float>%c1, <16 x float>%c2, <4 x i32> <i32 4, i32 20, i32 1, i32 27>
  %a6 = shufflevector <16 x float>%c1, <16 x float>%c2, <4 x i32> <i32 3, i32 20, i32 1, i32 17>
  %a7 = shufflevector <16 x float>%c1, <16 x float>%c2, <4 x i32> <i32 4, i32 21, i32 1, i32 17>
  %a8 = shufflevector <16 x float>%c1, <16 x float>%c2, <4 x i32> <i32 5, i32 20, i32 1, i32 19>
  %a9 = shufflevector <16 x float>%c1, <16 x float>%c2, <4 x i32> <i32 4, i32 20, i32 1, i32 17>
  %a10 = shufflevector <16 x float>%c1, <16 x float>%c2, <4 x i32> <i32 4, i32 20, i32 1, i32 17>
  %ax2 = shufflevector <16 x float>%c1, <16 x float>%c2, <4 x i32> <i32 4, i32 20, i32 1, i32 19>
  %ax5 = shufflevector <16 x float>%c1, <16 x float>%c2, <4 x i32> <i32 4, i32 20, i32 1, i32 17>
  %ax6 = shufflevector <16 x float>%c1, <16 x float>%c2, <4 x i32> <i32 4, i32 22, i32 1, i32 18>
  %ax7 = shufflevector <16 x float>%c1, <16 x float>%c2, <4 x i32> <i32 1, i32 20, i32 1, i32 17>
  %ax8 = shufflevector <16 x float>%c1, <16 x float>%c2, <4 x i32> <i32 4, i32 20, i32 1, i32 19>
  %ax9 = shufflevector <16 x float>%c1, <16 x float>%c2, <4 x i32> <i32 4, i32 20, i32 1, i32 17>
  %ax10 = shufflevector <16 x float>%c1, <16 x float>%c2, <4 x i32> <i32 4, i32 20, i32 1, i32 17>
  %ay2 = shufflevector <16 x float>%c1, <16 x float>%c2, <4 x i32> <i32 4, i32 20, i32 1, i32 17>
  %ay5 = shufflevector <16 x float>%c1, <16 x float>%c2, <4 x i32> <i32 4, i32 28, i32 1, i32 17>
  %ay6 = shufflevector <16 x float>%c1, <16 x float>%c2, <4 x i32> <i32 5, i32 20, i32 1, i32 17>
  %ay7 = shufflevector <16 x float>%c1, <16 x float>%c2, <4 x i32> <i32 4, i32 30, i32 1, i32 22>
  %ay8 = shufflevector <16 x float>%c1, <16 x float>%c2, <4 x i32> <i32 4, i32 20, i32 1, i32 17>
  %ay9 = shufflevector <16 x float>%c1, <16 x float>%c2, <4 x i32> <i32 4, i32 22, i32 1, i32 17>
  %ay10 = shufflevector <16 x float>%c1, <16 x float>%c2, <4 x i32> <i32 4, i32 20, i32 3, i32 18>

  %r1 = fadd <4 x float> %ay10, %ay9
  %r2 = fadd <4 x float> %ay8, %ay7
  %r3 = fadd <4 x float> %ay6, %ay5
  %r4 = fadd <4 x float> %ay2, %ax10
  %r5 = fadd <4 x float> %ay9, %ax8
  %r6 = fadd <4 x float> %r5, %r3
  %r7 = fadd <4 x float> %a9, %r6
  %a11 = call <4 x float> @foo(<4 x float> %r7, <4 x float> %a10, <4 x float> %r1, <4 x float> %a4, <4 x float> %a5, <4 x float> %a6, <4 x float> %a7, <4 x float> %a8, <4 x float> %r2, <4 x float> %r4)
  %a12 = fadd <4 x float> %a2, %a1
  %a13 = fadd <4 x float> %a12, %a11

  ret <4 x float> %a13
}
