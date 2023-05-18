; RUN: llc -mtriple=armv7k-apple-watchos2.0 < %s | FileCheck %s

define arm_aapcs_vfpcc float @t1(float %a, float %b) {
entry:
; CHECK: t1
; CHECK-NOT: vmov
; CHECK: vadd.f32 
  %a.addr = alloca float, align 4
  %b.addr = alloca float, align 4
  store float %a, ptr %a.addr, align 4
  store float %b, ptr %b.addr, align 4
  %0 = load float, ptr %a.addr, align 4
  %1 = load float, ptr %b.addr, align 4
  %add = fadd float %0, %1
  ret float %add
}

define arm_aapcs_vfpcc double @t2(double %a, double %b) {
entry:
; CHECK: t2
; CHECK-NOT: vmov
; CHECK: vadd.f64
  %a.addr = alloca double, align 8
  %b.addr = alloca double, align 8
  store double %a, ptr %a.addr, align 8
  store double %b, ptr %b.addr, align 8
  %0 = load double, ptr %a.addr, align 8
  %1 = load double, ptr %b.addr, align 8
  %add = fadd double %0, %1
  ret double %add
}

define arm_aapcs_vfpcc i64 @t3(double %ti) {
entry:
; CHECK-LABEL: t3:
; CHECK-NOT: vmov
; CHECK: bl ___fixunsdfdi
  %conv = fptoui double %ti to i64
  ret i64 %conv
}

define arm_aapcs_vfpcc i64 @t4(double %ti) {
entry:
; CHECK-LABEL: t4:
; CHECK-NOT: vmov
; CHECK: bl ___fixdfdi
  %conv = fptosi double %ti to i64
  ret i64 %conv
}

define arm_aapcs_vfpcc double @t5(i64 %ti) {
entry:
; CHECK-LABEL: t5:
; CHECK: bl ___floatundidf
; CHECK-NOT: vmov
; CHECK: pop
  %conv = uitofp i64 %ti to double
  ret double %conv
}

define arm_aapcs_vfpcc double @t6(i64 %ti) {
entry:
; CHECK-LABEL: t6:
; CHECK: bl ___floatdidf
; CHECK-NOT: vmov
; CHECK: pop
  %conv = sitofp i64 %ti to double
  ret double %conv
}

define arm_aapcs_vfpcc float @t7(i64 %ti) {
entry:
; CHECK-LABEL: t7:
; CHECK: bl ___floatundisf
; CHECK-NOT: vmov
; CHECK: pop
  %conv = uitofp i64 %ti to float
  ret float %conv
}

define arm_aapcs_vfpcc float @t8(i64 %ti) {
entry:
; CHECK-LABEL: t8:
; CHECK: bl ___floatdisf
; CHECK-NOT: vmov
; CHECK: pop
  %conv = sitofp i64 %ti to float
  ret float %conv
}

define arm_aapcs_vfpcc double @t9(double %d0, double %d1, double %d2, double %d3, double %d4, double %d5, double %d6, double %d7, float %a, float %b) {
entry:
; CHECK-LABEL: t9:
; CHECK-NOT: vmov
; CHECK: vldr
  %add = fadd float %a, %b
  %conv = fpext float %add to double
  ret double %conv
}

define arm_aapcs_vfpcc double @t10(double %d0, double %d1, double %d2, double %d3, double %d4, double %d5, double %a, float %b, double %c) {
entry:
; CHECK-LABEL: t10:
; CHECK-NOT: vmov
; CHECK: vldr
  %add = fadd double %a, %c
  ret double %add
}

define arm_aapcs_vfpcc float @t11(double %d0, double %d1, double %d2, double %d3, double %d4, double %d5, double %d6, float %a, double %b, float %c) {
entry:
; CHECK-LABEL: t11:
; CHECK: vldr
  %add = fadd float %a, %c
  ret float %add
}

define arm_aapcs_vfpcc double @t12(double %a, double %b) {
entry:
; CHECK-LABEL: t12:
; CHECK: vstr
  %add = fadd double %a, %b
  %sub = fsub double %a, %b
  %call = tail call arm_aapcs_vfpcc double @x(double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double %add, float 0.000000e+00, double %sub)
  ret double %call
}

define arm_aapcs_vfpcc double @t13(double %x) {
entry:
; CHECK-LABEL: t13:
; CHECK-NOT: vmov
; CHECK: bl ___sincos_stret
  %call = tail call arm_aapcs_vfpcc double @cos(double %x)
  %call1 = tail call arm_aapcs_vfpcc double @sin(double %x)
  %mul = fmul double %call, %call1
  ret double %mul
}

define arm_aapcs_vfpcc double @t14(double %x) {
; CHECK-LABEL: t14:
; CHECK-NOT: vmov
; CHECK: b ___exp10
  %__exp10 = tail call double @__exp10(double %x) #1
  ret double %__exp10
}

declare arm_aapcs_vfpcc double @x(double, double, double, double, double, double, double, float, double)
declare arm_aapcs_vfpcc double @cos(double) #0
declare arm_aapcs_vfpcc double @sin(double) #0
declare double @__exp10(double)

attributes #0 = { readnone }
attributes #1 = { readonly }
