; Check that abs.[ds] is selected and does not depend on -enable-no-nans-fp-math
; They obey the Has2008 and ABS2008 configuration bits which govern the
; conformance to IEEE 754 (1985) and IEEE 754 (2008). When these bits are not
; present, they confirm to 1985.
; In 1985 mode, abs.[ds] are arithmetic (i.e. they raise invalid operation
; exceptions when given NaN's). In 2008 mode, they are non-arithmetic (i.e.
; they are copies and don't raise any exceptions).

; RUN: llc  < %s -mtriple=mipsel-linux-gnu -mcpu=mips32 | FileCheck %s
; RUN: llc  < %s -mtriple=mipsel-linux-gnu -mcpu=mips32r2 | FileCheck %s
; RUN: llc  < %s -mtriple=mipsel-linux-gnu -mcpu=mips32 -enable-no-nans-fp-math | FileCheck %s

; RUN: llc  < %s -mtriple=mips64el-linux-gnu -mcpu=mips64 | FileCheck %s
; RUN: llc  < %s -mtriple=mips64el-linux-gnu -mcpu=mips64 -enable-no-nans-fp-math | FileCheck %s

define float @foo0(float %d) nounwind readnone {
entry:
; CHECK-LABEL: foo0:
; CHECK: neg.s
  %sub = fsub float -0.000000e+00, %d
  ret float %sub
}

define double @foo1(double %d) nounwind readnone {
entry:
; CHECK-LABEL: foo1:
; CHECK: neg.d
  %sub = fsub double -0.000000e+00, %d
  ret double %sub
}
