; RUN: llc < %s -march=x86-64 -mattr=+sse3,-avx | FileCheck %s -check-prefix=SSE3
; RUN: llc < %s -march=x86-64 -mattr=-sse3,+avx | FileCheck %s -check-prefix=AVX

; SSE3-LABEL: haddpd1:
; SSE3-NOT: vhaddpd
; SSE3: haddpd
; AVX-LABEL: haddpd1:
; AVX: vhaddpd
define <2 x double> @haddpd1(<2 x double> %x, <2 x double> %y) {
  %a = shufflevector <2 x double> %x, <2 x double> %y, <2 x i32> <i32 0, i32 2>
  %b = shufflevector <2 x double> %x, <2 x double> %y, <2 x i32> <i32 1, i32 3>
  %r = fadd <2 x double> %a, %b
  ret <2 x double> %r
}

; SSE3-LABEL: haddpd2:
; SSE3-NOT: vhaddpd
; SSE3: haddpd
; AVX-LABEL: haddpd2:
; AVX: vhaddpd
define <2 x double> @haddpd2(<2 x double> %x, <2 x double> %y) {
  %a = shufflevector <2 x double> %x, <2 x double> %y, <2 x i32> <i32 1, i32 2>
  %b = shufflevector <2 x double> %y, <2 x double> %x, <2 x i32> <i32 2, i32 1>
  %r = fadd <2 x double> %a, %b
  ret <2 x double> %r
}

; SSE3-LABEL: haddpd3:
; SSE3-NOT: vhaddpd
; SSE3: haddpd
; AVX-LABEL: haddpd3:
; AVX: vhaddpd
define <2 x double> @haddpd3(<2 x double> %x) {
  %a = shufflevector <2 x double> %x, <2 x double> undef, <2 x i32> <i32 0, i32 undef>
  %b = shufflevector <2 x double> %x, <2 x double> undef, <2 x i32> <i32 1, i32 undef>
  %r = fadd <2 x double> %a, %b
  ret <2 x double> %r
}

; SSE3-LABEL: haddps1:
; SSE3-NOT: vhaddps
; SSE3: haddps
; AVX-LABEL: haddps1:
; AVX: vhaddps
define <4 x float> @haddps1(<4 x float> %x, <4 x float> %y) {
  %a = shufflevector <4 x float> %x, <4 x float> %y, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %b = shufflevector <4 x float> %x, <4 x float> %y, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %r = fadd <4 x float> %a, %b
  ret <4 x float> %r
}

; SSE3-LABEL: haddps2:
; SSE3-NOT: vhaddps
; SSE3: haddps
; AVX-LABEL: haddps2:
; AVX: vhaddps
define <4 x float> @haddps2(<4 x float> %x, <4 x float> %y) {
  %a = shufflevector <4 x float> %x, <4 x float> %y, <4 x i32> <i32 1, i32 2, i32 5, i32 6>
  %b = shufflevector <4 x float> %y, <4 x float> %x, <4 x i32> <i32 4, i32 7, i32 0, i32 3>
  %r = fadd <4 x float> %a, %b
  ret <4 x float> %r
}

; SSE3-LABEL: haddps3:
; SSE3-NOT: vhaddps
; SSE3: haddps
; AVX-LABEL: haddps3:
; AVX: vhaddps
define <4 x float> @haddps3(<4 x float> %x) {
  %a = shufflevector <4 x float> %x, <4 x float> undef, <4 x i32> <i32 undef, i32 2, i32 4, i32 6>
  %b = shufflevector <4 x float> %x, <4 x float> undef, <4 x i32> <i32 undef, i32 3, i32 5, i32 7>
  %r = fadd <4 x float> %a, %b
  ret <4 x float> %r
}

; SSE3-LABEL: haddps4:
; SSE3-NOT: vhaddps
; SSE3: haddps
; AVX-LABEL: haddps4:
; AVX: vhaddps
define <4 x float> @haddps4(<4 x float> %x) {
  %a = shufflevector <4 x float> %x, <4 x float> undef, <4 x i32> <i32 0, i32 2, i32 undef, i32 undef>
  %b = shufflevector <4 x float> %x, <4 x float> undef, <4 x i32> <i32 1, i32 3, i32 undef, i32 undef>
  %r = fadd <4 x float> %a, %b
  ret <4 x float> %r
}

; SSE3-LABEL: haddps5:
; SSE3-NOT: vhaddps
; SSE3: haddps
; AVX-LABEL: haddps5:
; AVX: vhaddps
define <4 x float> @haddps5(<4 x float> %x) {
  %a = shufflevector <4 x float> %x, <4 x float> undef, <4 x i32> <i32 0, i32 3, i32 undef, i32 undef>
  %b = shufflevector <4 x float> %x, <4 x float> undef, <4 x i32> <i32 1, i32 2, i32 undef, i32 undef>
  %r = fadd <4 x float> %a, %b
  ret <4 x float> %r
}

; SSE3-LABEL: haddps6:
; SSE3-NOT: vhaddps
; SSE3: haddps
; AVX-LABEL: haddps6:
; AVX: vhaddps
define <4 x float> @haddps6(<4 x float> %x) {
  %a = shufflevector <4 x float> %x, <4 x float> undef, <4 x i32> <i32 0, i32 undef, i32 undef, i32 undef>
  %b = shufflevector <4 x float> %x, <4 x float> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %r = fadd <4 x float> %a, %b
  ret <4 x float> %r
}

; SSE3-LABEL: haddps7:
; SSE3-NOT: vhaddps
; SSE3: haddps
; AVX-LABEL: haddps7:
; AVX: vhaddps
define <4 x float> @haddps7(<4 x float> %x) {
  %a = shufflevector <4 x float> %x, <4 x float> undef, <4 x i32> <i32 undef, i32 3, i32 undef, i32 undef>
  %b = shufflevector <4 x float> %x, <4 x float> undef, <4 x i32> <i32 undef, i32 2, i32 undef, i32 undef>
  %r = fadd <4 x float> %a, %b
  ret <4 x float> %r
}

; SSE3-LABEL: hsubpd1:
; SSE3-NOT: vhsubpd
; SSE3: hsubpd
; AVX-LABEL: hsubpd1:
; AVX: vhsubpd
define <2 x double> @hsubpd1(<2 x double> %x, <2 x double> %y) {
  %a = shufflevector <2 x double> %x, <2 x double> %y, <2 x i32> <i32 0, i32 2>
  %b = shufflevector <2 x double> %x, <2 x double> %y, <2 x i32> <i32 1, i32 3>
  %r = fsub <2 x double> %a, %b
  ret <2 x double> %r
}

; SSE3-LABEL: hsubpd2:
; SSE3-NOT: vhsubpd
; SSE3: hsubpd
; AVX-LABEL: hsubpd2:
; AVX: vhsubpd
define <2 x double> @hsubpd2(<2 x double> %x) {
  %a = shufflevector <2 x double> %x, <2 x double> undef, <2 x i32> <i32 0, i32 undef>
  %b = shufflevector <2 x double> %x, <2 x double> undef, <2 x i32> <i32 1, i32 undef>
  %r = fsub <2 x double> %a, %b
  ret <2 x double> %r
}

; SSE3-LABEL: hsubps1:
; SSE3-NOT: vhsubps
; SSE3: hsubps
; AVX-LABEL: hsubps1:
; AVX: vhsubps
define <4 x float> @hsubps1(<4 x float> %x, <4 x float> %y) {
  %a = shufflevector <4 x float> %x, <4 x float> %y, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %b = shufflevector <4 x float> %x, <4 x float> %y, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %r = fsub <4 x float> %a, %b
  ret <4 x float> %r
}

; SSE3-LABEL: hsubps2:
; SSE3-NOT: vhsubps
; SSE3: hsubps
; AVX-LABEL: hsubps2:
; AVX: vhsubps
define <4 x float> @hsubps2(<4 x float> %x) {
  %a = shufflevector <4 x float> %x, <4 x float> undef, <4 x i32> <i32 undef, i32 2, i32 4, i32 6>
  %b = shufflevector <4 x float> %x, <4 x float> undef, <4 x i32> <i32 undef, i32 3, i32 5, i32 7>
  %r = fsub <4 x float> %a, %b
  ret <4 x float> %r
}

; SSE3-LABEL: hsubps3:
; SSE3-NOT: vhsubps
; SSE3: hsubps
; AVX-LABEL: hsubps3:
; AVX: vhsubps
define <4 x float> @hsubps3(<4 x float> %x) {
  %a = shufflevector <4 x float> %x, <4 x float> undef, <4 x i32> <i32 0, i32 2, i32 undef, i32 undef>
  %b = shufflevector <4 x float> %x, <4 x float> undef, <4 x i32> <i32 1, i32 3, i32 undef, i32 undef>
  %r = fsub <4 x float> %a, %b
  ret <4 x float> %r
}

; SSE3-LABEL: hsubps4:
; SSE3-NOT: vhsubps
; SSE3: hsubps
; AVX-LABEL: hsubps4:
; AVX: vhsubps
define <4 x float> @hsubps4(<4 x float> %x) {
  %a = shufflevector <4 x float> %x, <4 x float> undef, <4 x i32> <i32 0, i32 undef, i32 undef, i32 undef>
  %b = shufflevector <4 x float> %x, <4 x float> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %r = fsub <4 x float> %a, %b
  ret <4 x float> %r
}

; SSE3-LABEL: vhaddps1:
; SSE3-NOT: vhaddps
; SSE3: haddps
; SSE3: haddps
; AVX-LABEL: vhaddps1:
; AVX: vhaddps
define <8 x float> @vhaddps1(<8 x float> %x, <8 x float> %y) {
  %a = shufflevector <8 x float> %x, <8 x float> %y, <8 x i32> <i32 0, i32 2, i32 8, i32 10, i32 4, i32 6, i32 12, i32 14>
  %b = shufflevector <8 x float> %x, <8 x float> %y, <8 x i32> <i32 1, i32 3, i32 9, i32 11, i32 5, i32 7, i32 13, i32 15>
  %r = fadd <8 x float> %a, %b
  ret <8 x float> %r
}

; SSE3-LABEL: vhaddps2:
; SSE3-NOT: vhaddps
; SSE3: haddps
; SSE3: haddps
; AVX-LABEL: vhaddps2:
; AVX: vhaddps
define <8 x float> @vhaddps2(<8 x float> %x, <8 x float> %y) {
  %a = shufflevector <8 x float> %x, <8 x float> %y, <8 x i32> <i32 1, i32 2, i32 9, i32 10, i32 5, i32 6, i32 13, i32 14>
  %b = shufflevector <8 x float> %y, <8 x float> %x, <8 x i32> <i32 8, i32 11, i32 0, i32 3, i32 12, i32 15, i32 4, i32 7>
  %r = fadd <8 x float> %a, %b
  ret <8 x float> %r
}

; SSE3-LABEL: vhaddps3:
; SSE3-NOT: vhaddps
; SSE3: haddps
; SSE3: haddps
; AVX-LABEL: vhaddps3:
; AVX: vhaddps
define <8 x float> @vhaddps3(<8 x float> %x) {
  %a = shufflevector <8 x float> %x, <8 x float> undef, <8 x i32> <i32 undef, i32 2, i32 8, i32 10, i32 4, i32 6, i32 undef, i32 14>
  %b = shufflevector <8 x float> %x, <8 x float> undef, <8 x i32> <i32 1, i32 3, i32 9, i32 undef, i32 5, i32 7, i32 13, i32 15>
  %r = fadd <8 x float> %a, %b
  ret <8 x float> %r
}

; SSE3-LABEL: vhsubps1:
; SSE3-NOT: vhsubps
; SSE3: hsubps
; SSE3: hsubps
; AVX-LABEL: vhsubps1:
; AVX: vhsubps
define <8 x float> @vhsubps1(<8 x float> %x, <8 x float> %y) {
  %a = shufflevector <8 x float> %x, <8 x float> %y, <8 x i32> <i32 0, i32 2, i32 8, i32 10, i32 4, i32 6, i32 12, i32 14>
  %b = shufflevector <8 x float> %x, <8 x float> %y, <8 x i32> <i32 1, i32 3, i32 9, i32 11, i32 5, i32 7, i32 13, i32 15>
  %r = fsub <8 x float> %a, %b
  ret <8 x float> %r
}

; SSE3-LABEL: vhsubps3:
; SSE3-NOT: vhsubps
; SSE3: hsubps
; SSE3: hsubps
; AVX-LABEL: vhsubps3:
; AVX: vhsubps
define <8 x float> @vhsubps3(<8 x float> %x) {
  %a = shufflevector <8 x float> %x, <8 x float> undef, <8 x i32> <i32 undef, i32 2, i32 8, i32 10, i32 4, i32 6, i32 undef, i32 14>
  %b = shufflevector <8 x float> %x, <8 x float> undef, <8 x i32> <i32 1, i32 3, i32 9, i32 undef, i32 5, i32 7, i32 13, i32 15>
  %r = fsub <8 x float> %a, %b
  ret <8 x float> %r
}

; SSE3-LABEL: vhaddpd1:
; SSE3-NOT: vhaddpd
; SSE3: haddpd
; SSE3: haddpd
; AVX-LABEL: vhaddpd1:
; AVX: vhaddpd
define <4 x double> @vhaddpd1(<4 x double> %x, <4 x double> %y) {
  %a = shufflevector <4 x double> %x, <4 x double> %y, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
  %b = shufflevector <4 x double> %x, <4 x double> %y, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
  %r = fadd <4 x double> %a, %b
  ret <4 x double> %r
}

; SSE3-LABEL: vhsubpd1:
; SSE3-NOT: vhsubpd
; SSE3: hsubpd
; SSE3: hsubpd
; AVX-LABEL: vhsubpd1:
; AVX: vhsubpd
define <4 x double> @vhsubpd1(<4 x double> %x, <4 x double> %y) {
  %a = shufflevector <4 x double> %x, <4 x double> %y, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
  %b = shufflevector <4 x double> %x, <4 x double> %y, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
  %r = fsub <4 x double> %a, %b
  ret <4 x double> %r
}

; CHECK-LABEL: haddps_v2f32
; CHECK: haddps %xmm{{[0-9]+}}, %xmm0
; CHECK-NEXT: retq
define <2 x float> @haddps_v2f32(<4 x float> %v0) {
  %v0.0 = extractelement <4 x float> %v0, i32 0
  %v0.1 = extractelement <4 x float> %v0, i32 1
  %v0.2 = extractelement <4 x float> %v0, i32 2
  %v0.3 = extractelement <4 x float> %v0, i32 3
  %op0 = fadd float %v0.0, %v0.1
  %op1 = fadd float %v0.2, %v0.3
  %res0 = insertelement <2 x float> undef, float %op0, i32 0
  %res1 = insertelement <2 x float> %res0, float %op1, i32 1
  ret <2 x float> %res1
}
