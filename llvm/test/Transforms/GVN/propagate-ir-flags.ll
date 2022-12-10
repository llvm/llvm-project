
; RUN: opt < %s -passes=gvn -S | FileCheck %s

; CHECK-LABEL: func_fast
; CHECK:       fadd fast double
; CHECK-NEXT:  store
; CHECK-NEXT:  ret
define double @func_fast(double %a, double %b) {
entry:
  %a.addr = alloca double, align 8
  %add = fadd fast double %b, 3.000000e+00
  store double %add, double* %a.addr, align 8
  %load_add = load double, double* %a.addr, align 8
  ret double %load_add
}

; CHECK-LABEL: func_no_fast
; CHECK:       fadd double
; CHECK-NEXT:  store
; CHECK-NEXT:  ret
define double @func_no_fast(double %a, double %b) {
entry:
  %a.addr = alloca double, align 8
  %add = fadd fast double %b, 3.000000e+00
  store double %add, double* %a.addr, align 8
  %duplicated_add = fadd double %b, 3.000000e+00
  ret double %duplicated_add
}

