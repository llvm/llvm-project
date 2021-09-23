; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900 -denormal-fp-math-f32=preserve-sign -denormal-fp-math=preserve-sign -fp-contract=on < %s | FileCheck -check-prefixes=FUSED,NOCONTRACT,THRPTALL,ALL %s
; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900 -denormal-fp-math-f32=ieee -denormal-fp-math=ieee -fp-contract=on < %s | FileCheck -check-prefixes=SLOW,NOCONTRACT,THRPTALL,ALL %s
; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900 -denormal-fp-math-f32=ieee -denormal-fp-math=ieee -fp-contract=fast < %s | FileCheck -check-prefixes=FUSED,CONTRACT,THRPTALL,ALL %s
; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx1030 -denormal-fp-math-f32=preserve-sign -denormal-fp-math=preserve-sign -fp-contract=on < %s | FileCheck -check-prefixes=GFX1030,NOCONTRACT,THRPTALL,ALL %s
; RUN: opt -cost-model -cost-kind=code-size -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900 -denormal-fp-math-f32=preserve-sign -denormal-fp-math=preserve-sign -fp-contract=on < %s | FileCheck -check-prefixes=FUSED,SZNOCONTRACT,SIZEALL,ALL %s
; RUN: opt -cost-model -cost-kind=code-size -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900 -denormal-fp-math-f32=ieee -denormal-fp-math=ieee -fp-contract=on < %s | FileCheck -check-prefixes=SLOW,SZNOCONTRACT,SIZEALL,ALL %s
; RUN: opt -cost-model -cost-kind=code-size -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900 -denormal-fp-math-f32=ieee -denormal-fp-math=ieee -fp-contract=fast < %s | FileCheck -check-prefixes=FUSED,CONTRACT,SIZEALL,ALL %s
; RUN: opt -cost-model -cost-kind=code-size -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx1030 -denormal-fp-math-f32=preserve-sign -denormal-fp-math=preserve-sign -fp-contract=on < %s | FileCheck -check-prefixes=GFX1030,SZNOCONTRACT,SIZEALL,ALL %s
; END.

target triple = "amdgcn--"

; ALL-LABEL: 'fmul_fadd_f32':
; FUSED: estimated cost of 0 for {{.*}} fmul float
; SLOW: estimated cost of 1 for {{.*}} fmul float
; GFX1030: estimated cost of 1 for {{.*}} fmul float
; ALL: estimated cost of 1 for {{.*}} fadd float
; ALL: estimated cost of 0 for {{.*}} fmul contract float
; ALL: estimated cost of 1 for {{.*}} fadd contract float
; FUSED: estimated cost of 0 for {{.*}} fmul <2 x float>
; SLOW: estimated cost of 2 for {{.*}} fmul <2 x float>
; GFX1030: estimated cost of 2 for {{.*}} fmul <2 x float>
; ALL: estimated cost of 2 for {{.*}} fadd <2 x float>
; FUSED: estimated cost of 0 for {{.*}} fmul float
; SLOW: estimated cost of 1 for {{.*}} fmul float
; GFX1030: estimated cost of 1 for {{.*}} fmul float
; ALL: estimated cost of 1 for {{.*}} fsub float
; FUSED: estimated cost of 0 for {{.*}} fmul <2 x float>
; SLOW: estimated cost of 2 for {{.*}} fmul <2 x float>
; GFX1030: estimated cost of 2 for {{.*}} fmul <2 x float>
; ALL: estimated cost of 2 for {{.*}} fsub <2 x float>
define void @fmul_fadd_f32() #0 {
  %f32 = fmul float undef, undef
  %f32add = fadd float %f32, undef
  %f32c = fmul contract float undef, undef
  %f32cadd = fadd contract float %f32c, undef
  %v2f32 = fmul <2 x float> undef, undef
  %v2f32add = fadd <2 x float> %v2f32, undef
  %f32_2 = fmul float undef, undef
  %f32sub = fsub float %f32_2, undef
  %v2f32_2 = fmul <2 x float> undef, undef
  %v2f32sub = fsub <2 x float> %v2f32_2, undef
  ret void
}

; ALL-LABEL: 'fmul_fadd_f16':
; FUSED: estimated cost of 0 for {{.*}} fmul half
; SLOW: estimated cost of 1 for {{.*}} fmul half
; ALL: estimated cost of 1 for {{.*}} fadd half
; ALL: estimated cost of 0 for {{.*}} fmul contract half
; ALL: estimated cost of 1 for {{.*}} fadd contract half
; FUSED: estimated cost of 0 for {{.*}} fmul <2 x half>
; SLOW: estimated cost of 1 for {{.*}} fmul <2 x half>
; ALL: estimated cost of 1 for {{.*}} fadd <2 x half>
; FUSED: estimated cost of 0 for {{.*}} fmul half
; SLOW: estimated cost of 1 for {{.*}} fmul half
; ALL: estimated cost of 1 for {{.*}} fsub half
; FUSED: estimated cost of 0 for {{.*}} fmul <2 x half>
; SLOW: estimated cost of 1 for {{.*}} fmul <2 x half>
; ALL: estimated cost of 1 for {{.*}} fsub <2 x half>
define void @fmul_fadd_f16() #0 {
  %f16 = fmul half undef, undef
  %f16add = fadd half %f16, undef
  %f16c = fmul contract half undef, undef
  %f15cadd = fadd contract half %f16c, undef
  %v2f16 = fmul <2 x half> undef, undef
  %v2f16add = fadd <2 x half> %v2f16, undef
  %f16_2 = fmul half undef, undef
  %f16sub = fsub half %f16_2, undef
  %v2f16_2 = fmul <2 x half> undef, undef
  %v2f16sub = fsub <2 x half> %v2f16_2, undef
  ret void
}

; ALL-LABEL: 'fmul_fadd_f64':
; CONTRACT: estimated cost of 0 for {{.*}} fmul double
; NOCONTRACT: estimated cost of 4 for {{.*}} fmul double
; SZNOCONTRACT: estimated cost of 2 for {{.*}} fmul double
; THRPTALL: estimated cost of 4 for {{.*}} fadd double
; SIZEALL: estimated cost of 2 for {{.*}} fadd double
; ALL: estimated cost of 0 for {{.*}} fmul contract double
; THRPTALL: estimated cost of 4 for {{.*}} fadd contract double
; SIZEALL: estimated cost of 2 for {{.*}} fadd contract double
; CONTRACT: estimated cost of 0 for {{.*}} fmul <2 x double>
; NOCONTRACT: estimated cost of 8 for {{.*}} fmul <2 x double>
; SZNOCONTRACT: estimated cost of 4 for {{.*}} fmul <2 x double>
; THRPTALL: estimated cost of 8 for {{.*}} fadd <2 x double>
; SIZEALL: estimated cost of 4 for {{.*}} fadd <2 x double>
; CONTRACT: estimated cost of 0 for {{.*}} fmul double
; NOCONTRACT: estimated cost of 4 for {{.*}} fmul double
; SZNOCONTRACT: estimated cost of 2 for {{.*}} fmul double
; THRPTALL: estimated cost of 4 for {{.*}} fsub double
; SIZEALL: estimated cost of 2 for {{.*}} fsub double
; CONTRACT: estimated cost of 0 for {{.*}} fmul <2 x double>
; NOCONTRACT: estimated cost of 8 for {{.*}} fmul <2 x double>
; SZNOCONTRACT: estimated cost of 4 for {{.*}} fmul <2 x double>
; THRPTALL: estimated cost of 8 for {{.*}} fsub <2 x double>
; SIZEALL: estimated cost of 4 for {{.*}} fsub <2 x double>
define void @fmul_fadd_f64() #0 {
  %f64 = fmul double undef, undef
  %f64add = fadd double %f64, undef
  %f64c = fmul contract double undef, undef
  %f64cadd = fadd contract double %f64c, undef
  %v2f64 = fmul <2 x double> undef, undef
  %v2f64add = fadd <2 x double> %v2f64, undef
  %f64_2 = fmul double undef, undef
  %f64sub = fsub double %f64_2, undef
  %v2f64_2 = fmul <2 x double> undef, undef
  %v2f64sub = fsub <2 x double> %v2f64_2, undef
  ret void
}

attributes #0 = { nounwind }

