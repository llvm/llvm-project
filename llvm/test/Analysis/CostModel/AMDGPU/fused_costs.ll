; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900 -denormal-fp-math-f32=preserve-sign -denormal-fp-math=preserve-sign -fp-contract=on < %s | FileCheck -check-prefixes=FUSED,NOCONTRACT,THRPTALL,ALL %s
; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900 -denormal-fp-math-f32=ieee -denormal-fp-math=ieee -fp-contract=on < %s | FileCheck -check-prefixes=SLOW,NOCONTRACT,THRPTALL,ALL %s
; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900 -denormal-fp-math-f32=ieee -denormal-fp-math=ieee -fp-contract=fast < %s | FileCheck -check-prefixes=FUSED,CONTRACT,THRPTALL,ALL %s
; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx1030 -denormal-fp-math-f32=preserve-sign -denormal-fp-math=preserve-sign -fp-contract=on < %s | FileCheck -check-prefixes=GFX1030,NOCONTRACT,THRPTALL,ALL %s
; RUN: opt -cost-model -cost-kind=code-size -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900 -denormal-fp-math-f32=preserve-sign -denormal-fp-math=preserve-sign -fp-contract=on < %s | FileCheck -check-prefixes=FUSED,SZNOCONTRACT,SIZEALL,ALL %s
; RUN: opt -cost-model -cost-kind=code-size -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900 -denormal-fp-math-f32=ieee -denormal-fp-math=ieee -fp-contract=on < %s | FileCheck -check-prefixes=SLOW,SZNOCONTRACT,SIZEALL,ALL %s
; RUN: opt -cost-model -cost-kind=code-size -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900 -denormal-fp-math-f32=ieee -denormal-fp-math=ieee -fp-contract=fast < %s | FileCheck -check-prefixes=FUSED,CONTRACT,SIZEALL,ALL %s
; RUN: opt -cost-model -cost-kind=code-size -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx1030 -denormal-fp-math-f32=preserve-sign -denormal-fp-math=preserve-sign -fp-contract=on < %s | FileCheck -check-prefixes=GFX1030,SZNOCONTRACT,SIZEALL,ALL %s

target triple = "amdgcn--"

; ALL-LABEL: 'fmul_fadd_f32':
; FUSED: estimated cost of 0 for instruction:   %mul = fmul float
; SLOW: estimated cost of 1 for instruction:   %mul = fmul float
; GFX1030: estimated cost of 1 for instruction:   %mul = fmul float
; ALL: estimated cost of 1 for instruction:   %add = fadd float
define float @fmul_fadd_f32(float %r0, float %r1, float %r2) #0 {
  %mul = fmul float %r0, %r1
  %add = fadd float %mul, %r2
  ret float %add
}

; ALL-LABEL: 'fmul_fadd_contract_f32':
; ALL: estimated cost of 0 for instruction:   %mul = fmul contract float
; ALL: estimated cost of 1 for instruction:   %add = fadd contract float
define float @fmul_fadd_contract_f32(float %r0, float %r1, float %r2) #0 {
  %mul = fmul contract float %r0, %r1
  %add = fadd contract float %mul, %r2
  ret float %add
}

; ALL-LABEL: 'fmul_fadd_v2f32':
; FUSED: estimated cost of 0 for instruction:   %mul = fmul <2 x float>
; SLOW: estimated cost of 2 for instruction:   %mul = fmul <2 x float>
; GFX1030: estimated cost of 2 for instruction:   %mul = fmul <2 x float>
; ALL: estimated cost of 2 for instruction:   %add = fadd <2 x float>
define <2 x float> @fmul_fadd_v2f32(<2 x float> %r0, <2 x float> %r1, <2 x float> %r2) #0 {
  %mul = fmul <2 x float> %r0, %r1
  %add = fadd <2 x float> %mul, %r2
  ret <2 x float> %add
}

; ALL-LABEL: 'fmul_fsub_f32':
; FUSED: estimated cost of 0 for instruction:   %mul = fmul float
; SLOW: estimated cost of 1 for instruction:   %mul = fmul float
; GFX1030: estimated cost of 1 for instruction:   %mul = fmul float
; ALL: estimated cost of 1 for instruction:   %sub = fsub float
define float @fmul_fsub_f32(float %r0, float %r1, float %r2) #0 {
  %mul = fmul float %r0, %r1
  %sub = fsub float %mul, %r2
  ret float %sub
}

; ALL-LABEL: 'fmul_fsub_v2f32':
; FUSED: estimated cost of 0 for instruction:   %mul = fmul <2 x float>
; SLOW: estimated cost of 2 for instruction:   %mul = fmul <2 x float>
; GFX1030: estimated cost of 2 for instruction:   %mul = fmul <2 x float>
; ALL: estimated cost of 2 for instruction:   %sub = fsub <2 x float>
define <2 x float> @fmul_fsub_v2f32(<2 x float> %r0, <2 x float> %r1, <2 x float> %r2) #0 {
  %mul = fmul <2 x float> %r0, %r1
  %sub = fsub <2 x float> %mul, %r2
  ret <2 x float> %sub
}

; ALL-LABEL: 'fmul_fadd_f16':
; FUSED: estimated cost of 0 for instruction:   %mul = fmul half
; SLOW: estimated cost of 1 for instruction:   %mul = fmul half
; ALL: estimated cost of 1 for instruction:   %add = fadd half
define half @fmul_fadd_f16(half %r0, half %r1, half %r2) #0 {
  %mul = fmul half %r0, %r1
  %add = fadd half %mul, %r2
  ret half %add
}

; ALL-LABEL: 'fmul_fadd_contract_f16':
; ALL: estimated cost of 0 for instruction:   %mul = fmul contract half
; ALL: estimated cost of 1 for instruction:   %add = fadd contract half
define half @fmul_fadd_contract_f16(half %r0, half %r1, half %r2) #0 {
  %mul = fmul contract half %r0, %r1
  %add = fadd contract half %mul, %r2
  ret half %add
}

; ALL-LABEL: 'fmul_fadd_v2f16':
; FUSED: estimated cost of 0 for instruction:   %mul = fmul <2 x half>
; SLOW: estimated cost of 1 for instruction:   %mul = fmul <2 x half>
; ALL: estimated cost of 1 for instruction:   %add = fadd <2 x half>
define <2 x half> @fmul_fadd_v2f16(<2 x half> %r0, <2 x half> %r1, <2 x half> %r2) #0 {
  %mul = fmul <2 x half> %r0, %r1
  %add = fadd <2 x half> %mul, %r2
  ret <2 x half> %add
}

; ALL-LABEL: 'fmul_fsub_f16':
; FUSED: estimated cost of 0 for instruction:   %mul = fmul half
; SLOW: estimated cost of 1 for instruction:   %mul = fmul half
; ALL: estimated cost of 1 for instruction:   %sub = fsub half
define half @fmul_fsub_f16(half %r0, half %r1, half %r2) #0 {
  %mul = fmul half %r0, %r1
  %sub = fsub half %mul, %r2
  ret half %sub
}

; ALL-LABEL: 'fmul_fsub_v2f16':
; FUSED: estimated cost of 0 for instruction:   %mul = fmul <2 x half>
; SLOW: estimated cost of 1 for instruction:   %mul = fmul <2 x half>
; ALL: estimated cost of 1 for instruction:   %sub = fsub <2 x half>
define <2 x half> @fmul_fsub_v2f16(<2 x half> %r0, <2 x half> %r1, <2 x half> %r2) #0 {
  %mul = fmul <2 x half> %r0, %r1
  %sub = fsub <2 x half> %mul, %r2
  ret <2 x half> %sub
}

; ALL-LABEL: 'fmul_fadd_f64':
; CONTRACT: estimated cost of 0 for instruction:   %mul = fmul double
; NOCONTRACT: estimated cost of 4 for instruction:   %mul = fmul double
; SZNOCONTRACT: estimated cost of 2 for instruction:   %mul = fmul double
; THRPTALL: estimated cost of 4 for instruction:   %add = fadd double
; SIZEALL: estimated cost of 2 for instruction:   %add = fadd double
define double @fmul_fadd_f64(double %r0, double %r1, double %r2) #0 {
  %mul = fmul double %r0, %r1
  %add = fadd double %mul, %r2
  ret double %add
}

; ALL-LABEL: 'fmul_fadd_contract_f64':
; ALL: estimated cost of 0 for instruction:   %mul = fmul contract double
; THRPTALL: estimated cost of 4 for instruction:   %add = fadd contract double
; SIZEALL: estimated cost of 2 for instruction:   %add = fadd contract double
define double @fmul_fadd_contract_f64(double %r0, double %r1, double %r2) #0 {
  %mul = fmul contract double %r0, %r1
  %add = fadd contract double %mul, %r2
  ret double %add
}

; ALL-LABEL: 'fmul_fadd_v2f64':
; CONTRACT: estimated cost of 0 for instruction:   %mul = fmul <2 x double>
; NOCONTRACT: estimated cost of 8 for instruction:   %mul = fmul <2 x double>
; SZNOCONTRACT: estimated cost of 4 for instruction:   %mul = fmul <2 x double>
; THRPTALL: estimated cost of 8 for instruction:   %add = fadd <2 x double>
; SIZEALL: estimated cost of 4 for instruction:   %add = fadd <2 x double>
define <2 x double> @fmul_fadd_v2f64(<2 x double> %r0, <2 x double> %r1, <2 x double> %r2) #0 {
  %mul = fmul <2 x double> %r0, %r1
  %add = fadd <2 x double> %mul, %r2
  ret <2 x double> %add
}

; ALL-LABEL: 'fmul_fsub_f64':
; CONTRACT: estimated cost of 0 for instruction:   %mul = fmul double
; NOCONTRACT: estimated cost of 4 for instruction:   %mul = fmul double
; SZNOCONTRACT: estimated cost of 2 for instruction:   %mul = fmul double
; THRPTALL: estimated cost of 4 for instruction:   %sub = fsub double
; SIZEALL: estimated cost of 2 for instruction:   %sub = fsub double
define double @fmul_fsub_f64(double %r0, double %r1, double %r2) #0 {
  %mul = fmul double %r0, %r1
  %sub = fsub double %mul, %r2
  ret double %sub
}

; ALL-LABEL: 'fmul_fsub_v2f64':
; CONTRACT: estimated cost of 0 for instruction:   %mul = fmul <2 x double>
; NOCONTRACT: estimated cost of 8 for instruction:   %mul = fmul <2 x double>
; SZNOCONTRACT: estimated cost of 4 for instruction:   %mul = fmul <2 x double>
; THRPTALL: estimated cost of 8 for instruction:   %sub = fsub <2 x double>
; SIZEALL: estimated cost of 4 for instruction:   %sub = fsub <2 x double>
define <2 x double> @fmul_fsub_v2f64(<2 x double> %r0, <2 x double> %r1, <2 x double> %r2) #0 {
  %mul = fmul <2 x double> %r0, %r1
  %sub = fsub <2 x double> %mul, %r2
  ret <2 x double> %sub
}

attributes #0 = { nounwind }
