; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx803 -filetype=obj < %s | llvm-objdump --triple=amdgcn-amd-amdhsa --mcpu=gfx803 --disassemble - | FileCheck %s

; Make sure computed instruction sizes for madak/madmk are correct and
; pass the instruction size verifier.

; CHECK: v_madak_f32 v0, v0, v1, 0x41200000 // 000000000004: 30000300 41200000
define float @v_madak_f32(float %a, float %b) #0 {
  %mul = fmul float %a, %b
  %madmk = fadd float %mul, 10.0
  ret float %madmk
}

; CHECK: v_madmk_f32 v0, v0, 0x41200000, v1 // 000000000044: 2E000300 41200000
define float @v_madmk_f32(float %a, float %b) #0 {
  %mul = fmul float %a, 10.0
  %madmk = fadd float %mul, %b
  ret float %madmk
}

; CHECK: v_madak_f16 v0, v0, v1, 0x4900 // 000000000084: 4A000300 00004900
define half @v_madak_f16(half %a, half %b) #0 {
  %mul = fmul half %a, %b
  %madmk = fadd half %mul, 10.0
  ret half %madmk
}

; CHECK: v_madmk_f16 v0, v0, 0x4900, v1 // 0000000000C4: 48000300 00004900
define half @v_madmk_f16(half %a, half %b) #0 {
  %mul = fmul half %a, 10.0
  %madmk = fadd half %mul, %b
  ret half %madmk
}

attributes #0 = { nounwind denormal_fpenv(preservesign) }
