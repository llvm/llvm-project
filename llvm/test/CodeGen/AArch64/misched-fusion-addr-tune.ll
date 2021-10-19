; RUN: llc %s -o - -mtriple=aarch64-unknown -mcpu=cortex-a65    | FileCheck %s

@var_float = dso_local global float 0.0
@var_double = dso_local global double 0.0
@var_double2 = dso_local global <2 x double> <double 0.0, double 0.0>

define dso_local void @ldst_double() {
  %valf = load volatile float, float* @var_float
  %vale = fpext float %valf to double
  %vald = load volatile double, double* @var_double
  %vald1 = insertelement <2 x double> undef, double %vald, i32 0
  %vald2 = insertelement <2 x double> %vald1, double %vale, i32 1
  store volatile <2 x double> %vald2, <2 x double>* @var_double2
  ret void

; CHECK-LABEL: ldst_double:
; CHECK: adrp [[RD:x[0-9]+]], var_double
; CHECK-NEXT: ldr {{d[0-9]+}}, {{\[}}[[RD]], {{#?}}:lo12:var_double{{\]}}
; CHECK: adrp [[RQ:x[0-9]+]], var_double2
; CHECK-NEXT: str {{q[0-9]+}}, {{\[}}[[RQ]], {{#?}}:lo12:var_double2{{\]}}
}

define dso_local void @ldst_double_tune_a53() #0 {
  %valf = load volatile float, float* @var_float
  %vale = fpext float %valf to double
  %vald = load volatile double, double* @var_double
  %vald1 = insertelement <2 x double> undef, double %vald, i32 0
  %vald2 = insertelement <2 x double> %vald1, double %vale, i32 1
  store volatile <2 x double> %vald2, <2 x double>* @var_double2
  ret void

; CHECK-LABEL: ldst_double_tune_a53:
; CHECK: adrp [[RD:x[0-9]+]], var_double
; CHECK-NEXT: ldr {{d[0-9]+}}, {{\[}}[[RD]], {{#?}}:lo12:var_double{{\]}}
; CHECK-NEXT: adrp [[RQ:x[0-9]+]], var_double2
; CHECK: fcvt
; CHECK: str {{q[0-9]+}}, {{\[}}[[RQ]], {{#?}}:lo12:var_double2{{\]}}
}

attributes #0 = { "tune-cpu"="cortex-a53" }
