; RUN: llc -mtriple=mipsel-linux-gnu -mcpu=mips32 -mattr=+single-float < %s | FileCheck %s
; RUN: llc -mtriple=mipsel-linux-gnu -mcpu=mips32r2 -mattr=+single-float < %s | FileCheck %s

define double @dofloat(double %a, double %b) nounwind {
; CHECK-LABEL: dofloat:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    addiu $sp, $sp, -24
; CHECK-NEXT:    sw    $ra, 20($sp)
; CHECK-NEXT:    jal   __adddf3
; CHECK-NEXT:    nop
; CHECK-NEXT:    lw    $ra, 20($sp)
; CHECK-NEXT:    jr    $ra
; CHECK-NEXT:    addiu $sp, $sp, 24

entry:
  fadd double %a, %b ; <double>:0 [#uses=1]
  ret double %0
}
