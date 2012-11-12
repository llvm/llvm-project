; RUN: llc  < %s -march=mips64el -mcpu=mips64 -mattr=n64 | FileCheck %s -check-prefix=64
; RUN: llc  < %s -march=mips64el -mcpu=mips64r2 -mattr=n64 | FileCheck %s -check-prefix=64R2

declare double @copysign(double, double) nounwind readnone

declare float @copysignf(float, float) nounwind readnone

define float @func2(float %d, double %f) nounwind readnone {
entry:
; 64: func2
; 64: lui  $[[T0:[0-9]+]], 32767
; 64: ori  $[[MSK0:[0-9]+]], $[[T0]], 65535
; 64: and  $[[AND0:[0-9]+]], ${{[0-9]+}}, $[[MSK0]]
; 64: dsrl ${{[0-9]+}}, ${{[0-9]+}}, 63
; 64: sll  $[[SLL:[0-9]+]], ${{[0-9]+}}, 31
; 64: or   $[[OR:[0-9]+]], $[[AND0]], $[[SLL]]
; 64: mtc1 $[[OR]], $f0

; 64R2: dext ${{[0-9]+}}, ${{[0-9]+}}, 63, 1
; 64R2: ins  $[[INS:[0-9]+]], ${{[0-9]+}}, 31, 1
; 64R2: mtc1 $[[INS]], $f0

  %add = fadd float %d, 1.000000e+00
  %conv = fptrunc double %f to float
  %call = tail call float @copysignf(float %add, float %conv) nounwind readnone
  ret float %call
}

define double @func3(double %d, float %f) nounwind readnone {
entry:

; 64: daddiu $[[T0:[0-9]+]], $zero, 1
; 64: dsll   $[[T1:[0-9]+]], $[[T0]], 63
; 64: daddiu $[[MSK0:[0-9]+]], $[[T1]], -1
; 64: and    $[[AND0:[0-9]+]], ${{[0-9]+}}, $[[MSK0]]
; 64: srl    ${{[0-9]+}}, ${{[0-9]+}}, 31
; 64: dsll   $[[DSLL:[0-9]+]], ${{[0-9]+}}, 63
; 64: or     $[[OR:[0-9]+]], $[[AND0]], $[[DSLL]]
; 64: dmtc1  $[[OR]], $f0

; 64R2: ext   ${{[0-9]+}}, ${{[0-9]+}}, 31, 1
; 64R2: dins  $[[INS:[0-9]+]], ${{[0-9]+}}, 63, 1
; 64R2: dmtc1 $[[INS]], $f0

  %add = fadd double %d, 1.000000e+00
  %conv = fpext float %f to double
  %call = tail call double @copysign(double %add, double %conv) nounwind readnone
  ret double %call
}

