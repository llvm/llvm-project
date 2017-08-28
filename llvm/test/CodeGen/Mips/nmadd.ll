; Check whether nmadd/nmsub instructions are properly generated
; RUN: llc < %s -march=mipsel   -mcpu=mips32r2 -enable-no-nans-fp-math | FileCheck %s -check-prefixes=ALL,CHECK-NM
; RUN: llc < %s -march=mipsel   -mcpu=mips32r2 -mattr=+fp64 -enable-no-nans-fp-math | FileCheck %s -check-prefixes=ALL,CHECK-NM
; RUN: llc < %s -march=mipsel   -mcpu=mips32r2 -mattr=micromips -enable-no-nans-fp-math -asm-show-inst | FileCheck %s -check-prefixes=ALL,CHECK-NM,CHECK-MM
; RUN: llc < %s -march=mips64el -mcpu=mips64   -target-abi=n64 -enable-no-nans-fp-math | FileCheck %s -check-prefixes=ALL,CHECK-NM-64
; RUN: llc < %s -march=mips64el -mcpu=mips64r2 -target-abi=n64 -enable-no-nans-fp-math | FileCheck %s -check-prefixes=ALL,CHECK-NM-64
; RUN: llc < %s -march=mips64el -mcpu=mips4    -target-abi=n64 -enable-no-nans-fp-math | FileCheck %s -check-prefixes=ALL,CHECK-NM-64
; RUN: llc < %s -march=mipsel   -mcpu=mips32   -enable-no-nans-fp-math | FileCheck %s -check-prefixes=ALL,CHECK-NOT-NM
; RUN: llc < %s -march=mipsel   -mcpu=mips32r6 -enable-no-nans-fp-math | FileCheck %s -check-prefixes=ALL,CHECK-NOT-NM
; RUN: llc < %s -march=mips64el -mcpu=mips3    -target-abi=n64 -enable-no-nans-fp-math | FileCheck %s -check-prefixes=ALL,CHECK-NOT-NM-64
; RUN-TODO: llc < %s -march=mipsel   -mcpu=mips32r6 -mattr=micromips -enable-no-nans-fp-math | FileCheck %s -check-prefixes=ALL,CHECK-NOT-NM

define float @add1(float %f, float %g, float %h) local_unnamed_addr #0 {
entry:
; ALL-LABEL: add1

; CHECK-NM-64:             nmadd.s $f0, $f14, $f12, $f13
; CHECK-NM:                nmadd.s $f0, $f0, $f12, $f14
; CHECK-MM:                NMADD_S_MM
; CHECK-NOT-NM-64          mul.s $f0, $f12, $f13
; CHECK-NOT-NM-64:         neg.s $f0, $f0
; CHECK-NOT-NM:            mul.s $f0, $f12, $f14
; CHECK-NOT-NM:            neg.s $f0, $f0

  %mul = fmul nnan float %f, %g
  %add = fadd nnan float %mul, %h
  %sub = fsub nnan float -0.000000e+00, %add
  ret float %sub
}

define double @add2(double %f, double %g, double %h) local_unnamed_addr #0 {
entry:
; ALL-LABEL: add2

; CHECK-NM-64:             nmadd.d $f0, $f14, $f12, $f13
; CHECK-NM:                nmadd.d $f0, $f0, $f12, $f14
; CHECK-MM:                NMADD_D32_MM
; CHECK-NOT-NM-64          mul.d $f0, $f12, $f13
; CHECK-NOT-NM-64:         neg.d $f0, $f0
; CHECK-NOT-NM:            mul.d $f0, $f12, $f14
; CHECK-NOT-NM:            neg.d $f0, $f0

  %mul = fmul nnan double %f, %g
  %add = fadd nnan double %mul, %h
  %sub = fsub nnan double -0.000000e+00, %add
  ret double %sub
}

define float @sub1(float %f, float %g, float %h) local_unnamed_addr #0 {
entry:
; ALL-LABEL: sub1

; CHECK-NM-64:             nmsub.s $f0, $f14, $f12, $f13
; CHECK-NM:                nmsub.s $f0, $f0, $f12, $f14
; CHECK-MM:                NMSUB_S_MM
; CHECK-NOT-NM-64          mul.s $f0, $f12, $f13
; CHECK-NOT-NM-64:         neg.s $f0, $f0
; CHECK-NOT-NM:            mul.s $f0, $f12, $f14
; CHECK-NOT-NM:            neg.s $f0, $f0

  %mul = fmul nnan float %f, %g
  %sub = fsub nnan float %mul, %h
  %sub1 = fsub nnan float -0.000000e+00, %sub
  ret float %sub1
}

define double @sub2(double %f, double %g, double %h) local_unnamed_addr #0 {
entry:
; ALL-LABEL: sub2

; CHECK-NM-64:             nmsub.d $f0, $f14, $f12, $f13
; CHECK-NM:                nmsub.d $f0, $f0, $f12, $f14
; CHECK-MM:                NMSUB_D32_MM
; CHECK-NOT-NM-64          mul.d $f0, $f12, $f13
; CHECK-NOT-NM-64:         neg.d $f0, $f0
; CHECK-NOT-NM:            mul.d $f0, $f12, $f14
; CHECK-NOT-NM:            neg.d $f0, $f0

  %mul = fmul nnan double %f, %g
  %sub = fsub nnan double %mul, %h
  %sub1 = fsub nnan double -0.000000e+00, %sub
  ret double %sub1
}
