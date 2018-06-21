; RUN: llc < %s -O0 -march=nvptx -mcpu=sm_20 -asm-verbose=1 | FileCheck %s

; CHECK: // implicit-def: %f[[F0:[0-9]+]]
; CHECK: add.rn.f32         %f{{[0-9]+}}, %f{{[0-9]+}}, %f[[F0]];
define float @foo(float %a) {
  %ret = fadd float %a, undef
  ret float %ret
}

