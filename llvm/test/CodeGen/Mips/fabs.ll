; Check that abs.[ds] is only selected for mips32r6 or mips64r6 when no
; additional options are passed. For revisions prior mips32r6 and mips64r6,
; abs.[ds] does not generate the correct result when working with NaNs, and
; should be explicitly enabled with -enable-no-nans-fp-math or +abs2008 options.

; In 1985 mode, abs.[ds] are arithmetic (i.e. they raise invalid operation
; exceptions when given NaN's). In 2008 mode, they are non-arithmetic (i.e.
; they are copies and don't raise any exceptions).

; Testing default values
; RUN: llc  < %s -mtriple=mipsel-linux-gnu -mcpu=mips32 | FileCheck %s \
; RUN:    -check-prefix=CHECK-ABSLEGACY
; RUN: llc  < %s -mtriple=mips64el-linux-gnu -mcpu=mips64 | FileCheck %s \
; RUN:    -check-prefix=CHECK-ABSLEGACY
; RUN: llc  < %s -mtriple=mipsel-linux-gnu -mcpu=mips32r2 | FileCheck %s \
; RUN:    -check-prefix=CHECK-ABSLEGACY
; RUN: llc  < %s -mtriple=mips64el-linux-gnu -mcpu=mips64r2 | FileCheck %s \
; RUN:    -check-prefix=CHECK-ABSLEGACY
; RUN: llc  < %s -mtriple=mipsel-linux-gnu -mcpu=mips32r6 | FileCheck %s \
; RUN:    -check-prefix=CHECK-ABS2008
; RUN: llc  < %s -mtriple=mips64el-linux-gnu -mcpu=mips64r6 | FileCheck %s \
; RUN:    -check-prefix=CHECK-ABS2008
; RUN: llc  < %s -mtriple=mips-linux-gnu -mcpu=mips32 | FileCheck %s \
; RUN:    -check-prefix=CHECK-ABSLEGACY
; RUN: llc  < %s -mtriple=mips64-linux-gnu -mcpu=mips64 | FileCheck %s \
; RUN:    -check-prefix=CHECK-ABSLEGACY
; RUN: llc  < %s -mtriple=mips-linux-gnu -mcpu=mips32r2 | FileCheck %s \
; RUN:    -check-prefix=CHECK-ABSLEGACY
; RUN: llc  < %s -mtriple=mips64-linux-gnu -mcpu=mips64r2 | FileCheck %s \
; RUN:    -check-prefix=CHECK-ABSLEGACY
; RUN: llc  < %s -mtriple=mips-linux-gnu -mcpu=mips32r6 | FileCheck %s \
; RUN:    -check-prefix=CHECK-ABS2008
; RUN: llc  < %s -mtriple=mips64-linux-gnu -mcpu=mips64r6 | FileCheck %s \
; RUN:    -check-prefix=CHECK-ABS2008
; Testing non-default values
; RUN: llc  < %s -mtriple=mipsel-linux-gnu -mcpu=mips32r2 -mattr=+abs2008 \
; RUN:    | FileCheck %s -check-prefix=CHECK-ABS2008
; RUN: llc  < %s -mtriple=mips64el-linux-gnu -mcpu=mips64r2 -mattr=+abs2008 \
; RUN:    | FileCheck %s -check-prefix=CHECK-ABS2008
; RUN: llc  < %s -mtriple=mips-linux-gnu -mcpu=mips32r2 -mattr=+abs2008 \
; RUN:    | FileCheck %s -check-prefix=CHECK-ABS2008
; RUN: llc  < %s -mtriple=mips64-linux-gnu -mcpu=mips64r2 -mattr=+abs2008 \
; RUN:    | FileCheck %s -check-prefix=CHECK-ABS2008
; Testing -enable-no-nans-fp-math
; RUN: llc  < %s -mtriple=mipsel-linux-gnu -mcpu=mips32 \
; RUN:    -enable-no-nans-fp-math | FileCheck %s -check-prefix=CHECK-ABS2008
; RUN: llc  < %s -mtriple=mips64el-linux-gnu -mcpu=mips64 \
; RUN:    -enable-no-nans-fp-math | FileCheck %s -check-prefix=CHECK-ABS2008
; RUN: llc  < %s -mtriple=mips-linux-gnu -mcpu=mips32 \
; RUN:    -enable-no-nans-fp-math | FileCheck %s -check-prefix=CHECK-ABS2008
; RUN: llc  < %s -mtriple=mips64-linux-gnu -mcpu=mips64 \
; RUN:    -enable-no-nans-fp-math | FileCheck %s -check-prefix=CHECK-ABS2008

; microMIPS
; Testing default values
; RUN: llc  < %s -mtriple=mipsel-linux-gnu -mcpu=mips32 -mattr=+micromips \
; RUN:    | FileCheck %s -check-prefix=CHECK-ABSLEGACY
; RUN: llc  < %s -mtriple=mipsel-linux-gnu -mcpu=mips32r2 -mattr=+micromips \
; RUN:    | FileCheck %s -check-prefix=CHECK-ABSLEGACY
; RUN: llc  < %s -mtriple=mips-linux-gnu -mcpu=mips32 -mattr=+micromips \
; RUN:    | FileCheck %s -check-prefix=CHECK-ABSLEGACY
; RUN: llc  < %s -mtriple=mips-linux-gnu -mcpu=mips32r2 -mattr=+micromips \
; RUN:    | FileCheck %s -check-prefix=CHECK-ABSLEGACY
; Testing non-default values
; RUN: llc  < %s -mtriple=mipsel-linux-gnu -mcpu=mips32r2 \
; RUN:    -mattr=+abs2008,+micromips | FileCheck %s -check-prefix=CHECK-ABS2008
; RUN: llc  < %s -mtriple=mips-linux-gnu -mcpu=mips32r2 \
; RUN:    -mattr=+abs2008,+micromips | FileCheck %s -check-prefix=CHECK-ABS2008
; Testing -enable-no-nans-fp-math
; RUN: llc  < %s -mtriple=mipsel-linux-gnu -mcpu=mips32 -mattr=+micromips \
; RUN:    -enable-no-nans-fp-math | FileCheck %s -check-prefix=CHECK-ABS2008
; RUN: llc  < %s -mtriple=mips-linux-gnu -mcpu=mips32 -mattr=+micromips \
; RUN:    -enable-no-nans-fp-math | FileCheck %s -check-prefix=CHECK-ABS2008

define float @foo0(float %a) nounwind readnone {
entry:

; CHECK-LABEL: foo0
; CHECK-ABS2008: abs.s
; CHECK-ABSLEGACY: {{(ori|ins)}}
; CHECK-ABSLEGACY-NOT: abs.s

  %call = tail call float @fabsf(float %a) nounwind readnone
  ret float %call
}

declare float @fabsf(float) nounwind readnone

define double @foo1(double %a) nounwind readnone {
entry:

; CHECK-LABEL: foo1:
; CHECK-ABS2008: abs.d
; CHECK-ABSLEGACY: {{(ori|ins|dsll)}}
; CHECK-ABSLEGACY-NOT: abs.d

  %call = tail call double @fabs(double %a) nounwind readnone
  ret double %call
}

declare double @fabs(double) nounwind readnone
