; RUN: llc  < %s -mtriple=mipsel-linux-gnu -mcpu=mips32 | FileCheck %s -check-prefix=32
; RUN: llc  < %s -mtriple=mipsel-linux-gnu -mcpu=mips32r2 | FileCheck %s -check-prefix=32R2
; RUN: llc  < %s -mtriple=mips64el-linux-gnu -mcpu=mips64 -mattr=n64 | FileCheck %s -check-prefix=64
; RUN: llc  < %s -mtriple=mips64el-linux-gnu -mcpu=mips64r2 -mattr=n64 | FileCheck %s -check-prefix=64R2
; RUN: llc  < %s -mtriple=mipsel-linux-gnu -mcpu=mips32 -enable-no-nans-fp-math | FileCheck %s -check-prefix=NO-NAN

define float @foo0(float %a) nounwind readnone {
entry:

; 32: lui  $[[T0:[0-9]+]], 32767
; 32: ori  $[[MSK0:[0-9]+]], $[[T0]], 65535
; 32: and  $[[AND:[0-9]+]], ${{[0-9]+}}, $[[MSK0]]
; 32: mtc1 $[[AND]], $f0

; 32R2: ins  $[[INS:[0-9]+]], $zero, 31, 1
; 32R2: mtc1 $[[INS]], $f0

; NO-NAN: abs.s

  %call = tail call float @fabsf(float %a) nounwind readnone
  ret float %call
}

declare float @fabsf(float) nounwind readnone

define double @foo1(double %a) nounwind readnone {
entry:

; 32: lui  $[[T0:[0-9]+]], 32767
; 32: ori  $[[MSK0:[0-9]+]], $[[T0]], 65535
; 32: and  $[[AND:[0-9]+]], ${{[0-9]+}}, $[[MSK0]]
; 32: mtc1 $[[AND]], $f1

; 32R2: ins  $[[INS:[0-9]+]], $zero, 31, 1
; 32R2: mtc1 $[[INS]], $f1

; 64: daddiu  $[[T0:[0-9]+]], $zero, 1
; 64: dsll    $[[T1:[0-9]+]], ${{[0-9]+}}, 63
; 64: daddiu  $[[MSK0:[0-9]+]], $[[T1]], -1
; 64: and     $[[AND:[0-9]+]], ${{[0-9]+}}, $[[MSK0]]
; 64: dmtc1   $[[AND]], $f0

; 64R2: dins  $[[INS:[0-9]+]], $zero, 63, 1
; 64R2: dmtc1 $[[INS]], $f0

; NO-NAN: abs.d

  %call = tail call double @fabs(double %a) nounwind readnone
  ret double %call
}

declare double @fabs(double) nounwind readnone
