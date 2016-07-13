; RUN: llc %s -O0 -march=sparc -mcpu=ut699 -o - | FileCheck %s

; CHECK:        ld [%o0+%lo(.LCPI0_0)], %f0
; CHECK-NEXT:   nop


define float @X() #0 {
entry:
  %f = alloca float, align 4
  store float 0x3FF3C08320000000, float* %f, align 4
  %0 = load float, float* %f, align 4
  ret float %0
}
