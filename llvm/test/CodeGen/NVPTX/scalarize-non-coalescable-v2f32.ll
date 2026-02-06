; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_100 -fp-contract=fast | FileCheck %s
; RUN: %if ptxas-sm_100 %{ llc < %s -mtriple=nvptx64 -mcpu=sm_100 -fp-contract=fast | %ptxas-verify -arch sm_100 %}

target triple = "nvptx64-nvidia-cuda"

define <2 x float> @fma_non_coalescable(
    ptr addrspace(3) %p, <2 x float> %a, <2 x float> %c) {
; CHECK-LABEL: fma_non_coalescable(
; CHECK:     fma.rn.f32 %r
; CHECK:     fma.rn.f32 %r
; CHECK-NOT: fma{{.*}}f32x2
; CHECK:     ret
  %ld  = load <4 x float>, ptr addrspace(3) %p, align 16
  %e0  = extractelement <4 x float> %ld, i32 0
  %e2  = extractelement <4 x float> %ld, i32 2
  %bv0 = insertelement <2 x float> poison, float %e0, i32 0
  %bv  = insertelement <2 x float> %bv0,  float %e2, i32 1
  %mul = fmul <2 x float> %a, %bv
  %res = fadd <2 x float> %mul, %c
  ret <2 x float> %res
}

define <2 x float> @fadd_non_coalescable(
    ptr addrspace(3) %p, <2 x float> %a) {
; CHECK-LABEL: fadd_non_coalescable(
; CHECK:     add.f32 %r
; CHECK:     add.f32 %r
; CHECK-NOT: add{{.*}}f32x2
; CHECK:     ret
  %ld  = load <4 x float>, ptr addrspace(3) %p, align 16
  %e0  = extractelement <4 x float> %ld, i32 0
  %e2  = extractelement <4 x float> %ld, i32 2
  %bv0 = insertelement <2 x float> poison, float %e0, i32 0
  %bv  = insertelement <2 x float> %bv0,  float %e2, i32 1
  %add = fadd <2 x float> %a, %bv
  ret <2 x float> %add
}

define <2 x float> @fmul_non_coalescable(
    ptr addrspace(3) %p, <2 x float> %a) {
; CHECK-LABEL: fmul_non_coalescable(
; CHECK:     mul.f32 %r
; CHECK:     mul.f32 %r
; CHECK-NOT: mul{{.*}}f32x2
; CHECK:     ret
  %ld  = load <4 x float>, ptr addrspace(3) %p, align 16
  %e0  = extractelement <4 x float> %ld, i32 0
  %e2  = extractelement <4 x float> %ld, i32 2
  %bv0 = insertelement <2 x float> poison, float %e0, i32 0
  %bv  = insertelement <2 x float> %bv0,  float %e2, i32 1
  %mul = fmul <2 x float> %a, %bv
  ret <2 x float> %mul
}

define <2 x float> @fsub_non_coalescable(
    ptr addrspace(3) %p, <2 x float> %a) {
; CHECK-LABEL: fsub_non_coalescable(
; CHECK:     sub.f32 %r
; CHECK:     sub.f32 %r
; CHECK-NOT: sub{{.*}}f32x2
; CHECK:     ret
  %ld  = load <4 x float>, ptr addrspace(3) %p, align 16
  %e0  = extractelement <4 x float> %ld, i32 0
  %e2  = extractelement <4 x float> %ld, i32 2
  %bv0 = insertelement <2 x float> poison, float %e0, i32 0
  %bv  = insertelement <2 x float> %bv0,  float %e2, i32 1
  %sub = fsub <2 x float> %a, %bv
  ret <2 x float> %sub
}

; Should remain vectorized

define <2 x float> @fma_adjacent_elements(
    ptr addrspace(3) %p, <2 x float> %a, <2 x float> %c) {
; CHECK-LABEL: fma_adjacent_elements(
; CHECK:     fma.rn.f32x2
; CHECK:     ret
  %ld  = load <4 x float>, ptr addrspace(3) %p, align 16
  %e0  = extractelement <4 x float> %ld, i32 0
  %e1  = extractelement <4 x float> %ld, i32 1
  %bv0 = insertelement <2 x float> poison, float %e0, i32 0
  %bv  = insertelement <2 x float> %bv0,  float %e1, i32 1
  %mul = fmul <2 x float> %a, %bv
  %res = fadd <2 x float> %mul, %c
  ret <2 x float> %res
}

define <2 x float> @fma_naturally_paired(
    <2 x float> %a, <2 x float> %b, <2 x float> %c) {
; CHECK-LABEL: fma_naturally_paired(
; CHECK:     fma.rn.f32x2
; CHECK:     ret
  %mul = fmul <2 x float> %a, %b
  %res = fadd <2 x float> %mul, %c
  ret <2 x float> %res
}

define <2 x float> @fma_coalescable_scalars(
    float %x, float %y,
    <2 x float> %b, <2 x float> %c) {
; CHECK-LABEL: fma_coalescable_scalars(
; CHECK:     fma.rn.f32x2
; CHECK:     ret
  %v0  = insertelement <2 x float> poison, float %x, i32 0
  %v   = insertelement <2 x float> %v0,  float %y, i32 1
  %mul = fmul <2 x float> %v, %b
  %res = fadd <2 x float> %mul, %c
  ret <2 x float> %res
}
