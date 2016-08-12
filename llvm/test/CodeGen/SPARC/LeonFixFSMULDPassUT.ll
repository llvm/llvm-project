; RUN: llc %s -O0 -march=sparc -mattr=fixfsmuld -o - | FileCheck %s
; RUN: llc %s -O0 -march=sparc -o - | FileCheck %s --check-prefix=NOFIX

; CHECK-LABEL: test_fix_fsmuld_1
; CHECK:       fstod %f1, %f2
; CHECK:       fstod %f0, %f4
; CHECK:       fmuld %f2, %f4, %f0
; NOFIX-LABEL: test_fix_fsmuld_1
; NOFIX:       fsmuld %f1, %f0, %f0
define double @test_fix_fsmuld_1(float %a, float %b) {
entry:
  %0 = fpext float %a to double
  %1 = fpext float %b to double
  %mul = fmul double %0, %1

  ret double %mul
}
