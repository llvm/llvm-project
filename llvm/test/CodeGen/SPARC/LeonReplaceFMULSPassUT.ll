; RUN: llc %s -O0 -march=sparc -mcpu=ut699 -o - | FileCheck %s

; CHECK-LABEL: fmuls_fix_test
; CHECK:       fstod %f20, %f2
; CHECK:       fstod %f21, %f3
; CHECK:       fmuld %f2, %f3, %f8
; CHECK:       fstod %f20, %f0
define double @fmuls_fix_test() {
entry:
  %a = alloca float, align 4
  %b = alloca float, align 4
  store float 0x402ECCCCC0000000, float* %a, align 4
  store float 0x4022333340000000, float* %b, align 4
  %0 = load float, float* %b, align 4
  %1 = load float, float* %a, align 4
  %mul = tail call double asm sideeffect "fmuls $0, $1, $2", "={f20},{f21},{f8}"(float* %a, float* %b)

  ret double %mul
}
