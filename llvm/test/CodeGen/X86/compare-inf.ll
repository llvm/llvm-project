; RUN: llc < %s -mtriple=x86_64-- | FileCheck %s

; Convert oeq and une to ole/oge/ule/uge when comparing with infinity
; and negative infinity, because those are more efficient on x86.

declare void @f() nounwind

; CHECK-LABEL: oeq_inff:
; CHECK: ucomiss
; CHECK: jb
define void @oeq_inff(float %x) nounwind {
  %t0 = fcmp oeq float %x, 0x7FF0000000000000
  br i1 %t0, label %true, label %false

true:
  call void @f() nounwind
  br label %false

false:
  ret void
}

; CHECK-LABEL: oeq_inf:
; CHECK: ucomisd
; CHECK: jb
define void @oeq_inf(double %x) nounwind {
  %t0 = fcmp oeq double %x, 0x7FF0000000000000
  br i1 %t0, label %true, label %false

true:
  call void @f() nounwind
  br label %false

false:
  ret void
}

; CHECK-LABEL: une_inff:
; CHECK: ucomiss
; CHECK: jae
define void @une_inff(float %x) nounwind {
  %t0 = fcmp une float %x, 0x7FF0000000000000
  br i1 %t0, label %true, label %false

true:
  call void @f() nounwind
  br label %false

false:
  ret void
}

; CHECK-LABEL: une_inf:
; CHECK: ucomisd
; CHECK: jae
define void @une_inf(double %x) nounwind {
  %t0 = fcmp une double %x, 0x7FF0000000000000
  br i1 %t0, label %true, label %false

true:
  call void @f() nounwind
  br label %false

false:
  ret void
}

; CHECK-LABEL: oeq_neg_inff:
; CHECK: ucomiss
; CHECK: jb
define void @oeq_neg_inff(float %x) nounwind {
  %t0 = fcmp oeq float %x, 0xFFF0000000000000
  br i1 %t0, label %true, label %false

true:
  call void @f() nounwind
  br label %false

false:
  ret void
}

; CHECK-LABEL: oeq_neg_inf:
; CHECK: ucomisd
; CHECK: jb
define void @oeq_neg_inf(double %x) nounwind {
  %t0 = fcmp oeq double %x, 0xFFF0000000000000
  br i1 %t0, label %true, label %false

true:
  call void @f() nounwind
  br label %false

false:
  ret void
}

; CHECK-LABEL: une_neg_inff:
; CHECK: ucomiss
; CHECK: jae
define void @une_neg_inff(float %x) nounwind {
  %t0 = fcmp une float %x, 0xFFF0000000000000
  br i1 %t0, label %true, label %false

true:
  call void @f() nounwind
  br label %false

false:
  ret void
}

; CHECK-LABEL: une_neg_inf:
; CHECK: ucomisd
; CHECK: jae
define void @une_neg_inf(double %x) nounwind {
  %t0 = fcmp une double %x, 0xFFF0000000000000
  br i1 %t0, label %true, label %false

true:
  call void @f() nounwind
  br label %false

false:
  ret void
}
