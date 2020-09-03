; RUN: llc < %s -mtriple=thumbv7-none-eabi   -mcpu=cortex-m3 | FileCheck %s -check-prefix=CHECK -check-prefix=NONE
; RUN: llc < %s -mtriple=thumbv7-none-eabihf -mcpu=cortex-m4 | FileCheck %s -check-prefix=CHECK -check-prefix=HARD -check-prefix=SP
; RUN: llc < %s -mtriple=thumbv7-none-eabihf -mcpu=cortex-m7 | FileCheck %s -check-prefix=CHECK -check-prefix=HARD -check-prefix=DP
; RUN: llc < %s -mtriple=thumbv7-none-eabihf -mcpu=cortex-a8 | FileCheck %s -check-prefix=CHECK -check-prefix=HARD -check-prefix=DP



define i1 @cmp_f_false(float %a, float %b) {
; CHECK-LABEL: cmp_f_false:
; NONE: movs r0, #0
; HARD: movs r0, #0
  %1 = fcmp false float %a, %b
  ret i1 %1
}
define i1 @cmp_f_oeq(float %a, float %b) {
; CHECK-LABEL: cmp_f_oeq:
; NONE: bl __aeabi_fcmpeq
; HARD: vcmp.f32
; HARD: moveq r0, #1
  %1 = fcmp oeq float %a, %b
  ret i1 %1
}
define i1 @cmp_f_ogt(float %a, float %b) {
; CHECK-LABEL: cmp_f_ogt:
; NONE: bl __aeabi_fcmpgt
; HARD: vcmp.f32
; HARD: movgt r0, #1
  %1 = fcmp ogt float %a, %b
  ret i1 %1
}
define i1 @cmp_f_oge(float %a, float %b) {
; CHECK-LABEL: cmp_f_oge:
; NONE: bl __aeabi_fcmpge
; HARD: vcmp.f32
; HARD: movge r0, #1
  %1 = fcmp oge float %a, %b
  ret i1 %1
}
define i1 @cmp_f_olt(float %a, float %b) {
; CHECK-LABEL: cmp_f_olt:
; NONE: bl __aeabi_fcmplt
; HARD: vcmp.f32
; HARD: movmi r0, #1
  %1 = fcmp olt float %a, %b
  ret i1 %1
}
define i1 @cmp_f_ole(float %a, float %b) {
; CHECK-LABEL: cmp_f_ole:
; NONE: bl __aeabi_fcmple
; HARD: vcmp.f32
; HARD: movls r0, #1
  %1 = fcmp ole float %a, %b
  ret i1 %1
}
define i1 @cmp_f_one(float %a, float %b) {
; CHECK-LABEL: cmp_f_one:
; NONE: bl __aeabi_fcmpeq
; NONE: bl __aeabi_fcmpun
; HARD: vcmp.f32
; HARD: movmi r0, #1
; HARD: movgt r0, #1
  %1 = fcmp one float %a, %b
  ret i1 %1
}
define i1 @cmp_f_ord(float %a, float %b) {
; CHECK-LABEL: cmp_f_ord:
; NONE: bl __aeabi_fcmpun
; HARD: vcmp.f32
; HARD: movvc r0, #1
  %1 = fcmp ord float %a, %b
  ret i1 %1
}
define i1 @cmp_f_ueq(float %a, float %b) {
; CHECK-LABEL: cmp_f_ueq:
; NONE: bl __aeabi_fcmpeq
; NONE: bl __aeabi_fcmpun
; HARD: vcmp.f32
; HARD: moveq r0, #1
; HARD: movvs r0, #1
  %1 = fcmp ueq float %a, %b
  ret i1 %1
}
define i1 @cmp_f_ugt(float %a, float %b) {
; CHECK-LABEL: cmp_f_ugt:
; NONE: bl __aeabi_fcmple
; NONE-NEXT: clz r0, r0
; NONE-NEXT: lsrs r0, r0, #5
; HARD: vcmp.f32
; HARD: movhi r0, #1
  %1 = fcmp ugt float %a, %b
  ret i1 %1
}
define i1 @cmp_f_uge(float %a, float %b) {
; CHECK-LABEL: cmp_f_uge:
; NONE: bl __aeabi_fcmplt
; NONE-NEXT: clz r0, r0
; NONE-NEXT: lsrs r0, r0, #5
; HARD: vcmp.f32
; HARD: movpl r0, #1
  %1 = fcmp uge float %a, %b
  ret i1 %1
}
define i1 @cmp_f_ult(float %a, float %b) {
; CHECK-LABEL: cmp_f_ult:
; NONE: bl __aeabi_fcmpge
; NONE-NEXT: clz r0, r0
; NONE-NEXT: lsrs r0, r0, #5
; HARD: vcmp.f32
; HARD: movlt r0, #1
  %1 = fcmp ult float %a, %b
  ret i1 %1
}
define i1 @cmp_f_ule(float %a, float %b) {
; CHECK-LABEL: cmp_f_ule:
; NONE: bl __aeabi_fcmpgt
; NONE-NEXT: clz r0, r0
; NONE-NEXT: lsrs r0, r0, #5
; HARD: vcmp.f32
; HARD: movle r0, #1
  %1 = fcmp ule float %a, %b
  ret i1 %1
}
define i1 @cmp_f_une(float %a, float %b) {
; CHECK-LABEL: cmp_f_une:
; NONE: bl __aeabi_fcmpeq
; HARD: vcmp.f32
; HARD: movne r0, #1
  %1 = fcmp une float %a, %b
  ret i1 %1
}
define i1 @cmp_f_uno(float %a, float %b) {
; CHECK-LABEL: cmp_f_uno:
; NONE: bl __aeabi_fcmpun
; HARD: vcmp.f32
; HARD: movvs r0, #1
  %1 = fcmp uno float %a, %b
  ret i1 %1
}
define i1 @cmp_f_true(float %a, float %b) {
; CHECK-LABEL: cmp_f_true:
; NONE: movs r0, #1
; HARD: movs r0, #1
  %1 = fcmp true float %a, %b
  ret i1 %1
}

define i1 @cmp_d_false(double %a, double %b) {
; CHECK-LABEL: cmp_d_false:
; NONE: movs r0, #0
; HARD: movs r0, #0
  %1 = fcmp false double %a, %b
  ret i1 %1
}
define i1 @cmp_d_oeq(double %a, double %b) {
; CHECK-LABEL: cmp_d_oeq:
; NONE: bl __aeabi_dcmpeq
; SP: bl __aeabi_dcmpeq
; DP: vcmp.f64
; DP: moveq r0, #1
  %1 = fcmp oeq double %a, %b
  ret i1 %1
}
define i1 @cmp_d_ogt(double %a, double %b) {
; CHECK-LABEL: cmp_d_ogt:
; NONE: bl __aeabi_dcmpgt
; SP: bl __aeabi_dcmpgt
; DP: vcmp.f64
; DP: movgt r0, #1
  %1 = fcmp ogt double %a, %b
  ret i1 %1
}
define i1 @cmp_d_oge(double %a, double %b) {
; CHECK-LABEL: cmp_d_oge:
; NONE: bl __aeabi_dcmpge
; SP: bl __aeabi_dcmpge
; DP: vcmp.f64
; DP: movge r0, #1
  %1 = fcmp oge double %a, %b
  ret i1 %1
}
define i1 @cmp_d_olt(double %a, double %b) {
; CHECK-LABEL: cmp_d_olt:
; NONE: bl __aeabi_dcmplt
; SP: bl __aeabi_dcmplt
; DP: vcmp.f64
; DP: movmi r0, #1
  %1 = fcmp olt double %a, %b
  ret i1 %1
}
define i1 @cmp_d_ole(double %a, double %b) {
; CHECK-LABEL: cmp_d_ole:
; NONE: bl __aeabi_dcmple
; SP: bl __aeabi_dcmple
; DP: vcmp.f64
; DP: movls r0, #1
  %1 = fcmp ole double %a, %b
  ret i1 %1
}
define i1 @cmp_d_one(double %a, double %b) {
; CHECK-LABEL: cmp_d_one:
; NONE: bl __aeabi_dcmpeq
; NONE: bl __aeabi_dcmpun
; SP: bl __aeabi_dcmpeq
; SP: bl __aeabi_dcmpun
; DP: vcmp.f64
; DP: movmi r0, #1
; DP: movgt r0, #1
  %1 = fcmp one double %a, %b
  ret i1 %1
}
define i1 @cmp_d_ord(double %a, double %b) {
; CHECK-LABEL: cmp_d_ord:
; NONE: bl __aeabi_dcmpun
; SP: bl __aeabi_dcmpun
; DP: vcmp.f64
; DP: movvc r0, #1
  %1 = fcmp ord double %a, %b
  ret i1 %1
}
define i1 @cmp_d_ugt(double %a, double %b) {
; CHECK-LABEL: cmp_d_ugt:
; NONE: bl __aeabi_dcmple
; SP: bl __aeabi_dcmple
; DP: vcmp.f64
; DP: movhi r0, #1
  %1 = fcmp ugt double %a, %b
  ret i1 %1
}

define i1 @cmp_d_ult(double %a, double %b) {
; CHECK-LABEL: cmp_d_ult:
; NONE: bl __aeabi_dcmpge
; SP: bl __aeabi_dcmpge
; DP: vcmp.f64
; DP: movlt r0, #1
  %1 = fcmp ult double %a, %b
  ret i1 %1
}


define i1 @cmp_d_uno(double %a, double %b) {
; CHECK-LABEL: cmp_d_uno:
; NONE: bl __aeabi_dcmpun
; SP: bl __aeabi_dcmpun
; DP: vcmp.f64
; DP: movvs r0, #1
  %1 = fcmp uno double %a, %b
  ret i1 %1
}
define i1 @cmp_d_true(double %a, double %b) {
; CHECK-LABEL: cmp_d_true:
; NONE: movs r0, #1
; HARD: movs r0, #1
  %1 = fcmp true double %a, %b
  ret i1 %1
}
define i1 @cmp_d_ueq(double %a, double %b) {
; CHECK-LABEL: cmp_d_ueq:
; NONE: bl __aeabi_dcmpeq
; NONE: bl __aeabi_dcmpun
; SP: bl __aeabi_dcmpeq
; SP: bl __aeabi_dcmpun
; DP: vcmp.f64
; DP: moveq r0, #1
; DP: movvs r0, #1
  %1 = fcmp ueq double %a, %b
  ret i1 %1
}

define i1 @cmp_d_uge(double %a, double %b) {
; CHECK-LABEL: cmp_d_uge:
; NONE: bl __aeabi_dcmplt
; SP: bl __aeabi_dcmplt
; DP: vcmp.f64
; DP: movpl r0, #1
  %1 = fcmp uge double %a, %b
  ret i1 %1
}

define i1 @cmp_d_ule(double %a, double %b) {
; CHECK-LABEL: cmp_d_ule:
; NONE: bl __aeabi_dcmpgt
; SP: bl __aeabi_dcmpgt
; DP: vcmp.f64
; DP: movle r0, #1
  %1 = fcmp ule double %a, %b
  ret i1 %1
}

define i1 @cmp_d_une(double %a, double %b) {
; CHECK-LABEL: cmp_d_une:
; NONE: bl __aeabi_dcmpeq
; SP: bl __aeabi_dcmpeq
; DP: vcmp.f64
; DP: movne r0, #1
  %1 = fcmp une double %a, %b
  ret i1 %1
}
