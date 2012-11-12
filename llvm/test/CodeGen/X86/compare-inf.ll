; RUN: llc < %s -march=x86-64 | FileCheck %s

; Convert oeq and une to ole/oge/ule/uge when comparing with infinity
; and negative infinity, because those are more efficient on x86.

; CHECK: oeq_inff:
; CHECK: ucomiss
; CHECK: jb
define float @oeq_inff(float %x, float %y) nounwind readonly {
  %t0 = fcmp oeq float %x, 0x7FF0000000000000
  %t1 = select i1 %t0, float 1.0, float %y
  ret float %t1
}

; CHECK: oeq_inf:
; CHECK: ucomisd
; CHECK: jb
define double @oeq_inf(double %x, double %y) nounwind readonly {
  %t0 = fcmp oeq double %x, 0x7FF0000000000000
  %t1 = select i1 %t0, double 1.0, double %y
  ret double %t1
}

; CHECK: une_inff:
; CHECK: ucomiss
; CHECK: jae
define float @une_inff(float %x, float %y) nounwind readonly {
  %t0 = fcmp une float %x, 0x7FF0000000000000
  %t1 = select i1 %t0, float 1.0, float %y
  ret float %t1
}

; CHECK: une_inf:
; CHECK: ucomisd
; CHECK: jae
define double @une_inf(double %x, double %y) nounwind readonly {
  %t0 = fcmp une double %x, 0x7FF0000000000000
  %t1 = select i1 %t0, double 1.0, double %y
  ret double %t1
}

; CHECK: oeq_neg_inff:
; CHECK: ucomiss
; CHECK: jb
define float @oeq_neg_inff(float %x, float %y) nounwind readonly {
  %t0 = fcmp oeq float %x, 0xFFF0000000000000
  %t1 = select i1 %t0, float 1.0, float %y
  ret float %t1
}

; CHECK: oeq_neg_inf:
; CHECK: ucomisd
; CHECK: jb
define double @oeq_neg_inf(double %x, double %y) nounwind readonly {
  %t0 = fcmp oeq double %x, 0xFFF0000000000000
  %t1 = select i1 %t0, double 1.0, double %y
  ret double %t1
}

; CHECK: une_neg_inff:
; CHECK: ucomiss
; CHECK: jae
define float @une_neg_inff(float %x, float %y) nounwind readonly {
  %t0 = fcmp une float %x, 0xFFF0000000000000
  %t1 = select i1 %t0, float 1.0, float %y
  ret float %t1
}

; CHECK: une_neg_inf:
; CHECK: ucomisd
; CHECK: jae
define double @une_neg_inf(double %x, double %y) nounwind readonly {
  %t0 = fcmp une double %x, 0xFFF0000000000000
  %t1 = select i1 %t0, double 1.0, double %y
  ret double %t1
}
