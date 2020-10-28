; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900 -denormal-fp-math-f32=preserve-sign < %s | FileCheck -check-prefixes=FUSED,ALL %s
; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900 -denormal-fp-math-f32=ieee < %s | FileCheck -check-prefixes=SLOW,ALL %s
; RUN: opt -cost-model -cost-kind=code-size -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900 -denormal-fp-math-f32=preserve-sign < %s | FileCheck -check-prefixes=FUSED,ALL %s
; RUN: opt -cost-model -cost-kind=code-size -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900 -denormal-fp-math-f32=ieee < %s | FileCheck -check-prefixes=SLOW,ALL %s

target triple = "amdgcn--"

; ALL-LABEL: 'fmul_fadd_f32':
; FUSED: estimated cost of 0 for instruction:   %mul = fmul float
; SLOW: estimated cost of 1 for instruction:   %mul = fmul float
; ALL: estimated cost of 1 for instruction:   %add = fadd float
define float @fmul_fadd_f32(float %r0, float %r1, float %r2) #0 {
  %mul = fmul float %r0, %r1
  %add = fadd float %mul, %r2
  ret float %add
}

; ALL-LABEL: 'fmul_fadd_v2f32':
; FUSED: estimated cost of 0 for instruction:   %mul = fmul <2 x float>
; SLOW: estimated cost of 2 for instruction:   %mul = fmul <2 x float>
; ALL: estimated cost of 2 for instruction:   %add = fadd <2 x float>
define <2 x float> @fmul_fadd_v2f32(<2 x float> %r0, <2 x float> %r1, <2 x float> %r2) #0 {
  %mul = fmul <2 x float> %r0, %r1
  %add = fadd <2 x float> %mul, %r2
  ret <2 x float> %add
}

; ALL-LABEL: 'fmul_fsub_f32':
; FUSED: estimated cost of 0 for instruction:   %mul = fmul float
; SLOW: estimated cost of 1 for instruction:   %mul = fmul float
; ALL: estimated cost of 1 for instruction:   %sub = fsub float
define float @fmul_fsub_f32(float %r0, float %r1, float %r2) #0 {
  %mul = fmul float %r0, %r1
  %sub = fsub float %mul, %r2
  ret float %sub
}

; ALL-LABEL: 'fmul_fsub_v2f32':
; FUSED: estimated cost of 0 for instruction:   %mul = fmul <2 x float>
; SLOW: estimated cost of 2 for instruction:   %mul = fmul <2 x float>
; ALL: estimated cost of 2 for instruction:   %sub = fsub <2 x float>
define <2 x float> @fmul_fsub_v2f32(<2 x float> %r0, <2 x float> %r1, <2 x float> %r2) #0 {
  %mul = fmul <2 x float> %r0, %r1
  %sub = fsub <2 x float> %mul, %r2
  ret <2 x float> %sub
}

attributes #0 = { nounwind }
