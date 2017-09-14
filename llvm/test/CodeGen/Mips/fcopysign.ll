; RUN: llc  < %s -march=mipsel -mcpu=mips32 | FileCheck %s -check-prefix=32
; RUN: llc  < %s -march=mipsel -mcpu=mips32r2 | FileCheck %s -check-prefix=32R2
; RUN: llc  < %s -march=mips64el -mcpu=mips4 -target-abi=n64 | FileCheck %s -check-prefix=64
; RUN: llc  < %s -march=mips64el -mcpu=mips64 -target-abi=n64 | FileCheck %s -check-prefix=64
; RUN: llc  < %s -march=mips64el -mcpu=mips64r2 -target-abi=n64 | FileCheck %s -check-prefix=64R2

define double @func0(double %d0, double %d1) nounwind readnone {
entry:
;
; 32: lui  $[[MSK1:[0-9]+]], 32768
; 32: and  $[[AND1:[0-9]+]], ${{[0-9]+}}, $[[MSK1]]
; 32: lui  $[[T0:[0-9]+]], 32767
; 32: ori  $[[MSK0:[0-9]+]], $[[T0]], 65535
; 32: and  $[[AND0:[0-9]+]], ${{[0-9]+}}, $[[MSK0]]
; 32: or   $[[OR:[0-9]+]], $[[AND0]], $[[AND1]]
; 32: mtc1 $[[OR]], $f1

; 32R2: ext  $[[EXT:[0-9]+]], ${{[0-9]+}}, 31, 1
; 32R2: ins  $[[INS:[0-9]+]], $[[EXT]], 31, 1
; 32R2: mthc1 $[[INS]], $f0

; 64: daddiu $[[T0:[0-9]+]], $zero, 1
; 64: dsll   $[[MSK1:[0-9]+]], $[[T0]], 63
; 64: and    $[[AND1:[0-9]+]], ${{[0-9]+}}, $[[MSK1]]
; 64: daddiu $[[MSK0:[0-9]+]], $[[MSK1]], -1
; 64: and    $[[AND0:[0-9]+]], ${{[0-9]+}}, $[[MSK0]]
; 64: or     $[[OR:[0-9]+]], $[[AND0]], $[[AND1]]
; 64: dmtc1  $[[OR]], $f0

; 64R2: dextu  $[[EXT:[0-9]+]], ${{[0-9]+}}, 63, 1
; 64R2: dinsu  $[[INS:[0-9]+]], $[[EXT]], 63, 1
; 64R2: dmtc1  $[[INS]], $f0

  %call = tail call double @copysign(double %d0, double %d1) nounwind readnone
  ret double %call
}

declare double @copysign(double, double) nounwind readnone

define float @func1(float %f0, float %f1) nounwind readnone {
entry:

; 32: lui  $[[MSK1:[0-9]+]], 32768
; 32: and  $[[AND1:[0-9]+]], ${{[0-9]+}}, $[[MSK1]]
; 32: lui  $[[T0:[0-9]+]], 32767
; 32: ori  $[[MSK0:[0-9]+]], $[[T0]], 65535
; 32: and  $[[AND0:[0-9]+]], ${{[0-9]+}}, $[[MSK0]]
; 32: or   $[[OR:[0-9]+]], $[[AND0]], $[[AND1]]
; 32: mtc1 $[[OR]], $f0

; 32R2: ext  $[[EXT:[0-9]+]], ${{[0-9]+}}, 31, 1
; 32R2: ins  $[[INS:[0-9]+]], $[[EXT]], 31, 1
; 32R2: mtc1 $[[INS]], $f0

  %call = tail call float @copysignf(float %f0, float %f1) nounwind readnone
  ret float %call
}

declare float @copysignf(float, float) nounwind readnone

