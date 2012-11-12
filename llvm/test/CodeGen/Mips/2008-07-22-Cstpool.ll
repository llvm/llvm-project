; RUN: llc -march=mips < %s | FileCheck %s

define float @F(float %a) nounwind {
; CHECK: .rodata.cst4,"aM",@progbits 
entry:
; CHECK: ($CPI0_{{[0-1]}})
; CHECK: ($CPI0_{{[0,1]}}) 
; CHECK: ($CPI0_{{[0,1]}}) 
; CHECK: ($CPI0_{{[0,1]}}) 
  fadd float %a, 0x4011333340000000		; <float>:0 [#uses=1]
  fadd float %0, 0x4010666660000000		; <float>:1 [#uses=1]
  ret float %1
}
