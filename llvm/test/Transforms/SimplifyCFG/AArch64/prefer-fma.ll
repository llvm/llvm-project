; RUN: opt < %s -mtriple=aarch64-linux-gnu -simplifycfg -enable-unsafe-fp-math -S >%t
; RUN: FileCheck %s < %t
; ModuleID = 't.cc'

; Function Attrs: nounwind
define double @_Z3fooRdS_S_S_(double* dereferenceable(8) %x, double* dereferenceable(8) %y, double* dereferenceable(8) %a) #0 {
entry:
  %0 = load double, double* %y, align 8
  %cmp = fcmp oeq double %0, 0.000000e+00
  %1 = load double, double* %x, align 8
  br i1 %cmp, label %if.then, label %if.else

; fadd (const, (fmul x, y))
if.then:                                          ; preds = %entry
; CHECK-LABEL: if.then:
; CHECK:   %3 = fmul fast double %1, %2
; CHECK-NEXT:   %mul = fadd fast double 1.000000e+00, %3
  %2 = load double, double* %a, align 8
  %3 = fmul fast double %1, %2
  %mul = fadd fast double 1.000000e+00, %3
  store double %mul, double* %y, align 8
  br label %if.end

; fsub ((fmul x, y), z)
if.else:                                          ; preds = %entry
; CHECK-LABEL: if.else:
; CHECK:   %mul1 = fmul fast double %1, %2
; CHECK-NEXT:   %sub1 = fsub fast double %mul1, %0
  %4 = load double, double* %a, align 8
  %mul1 = fmul fast double %1, %4
  %sub1 = fsub fast double %mul1, %0
  %gep1 = getelementptr double, double* %y, i32 1
  store double %sub1, double* %gep1, align 8
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %5 = load double, double* %y, align 8
  %cmp2 = fcmp oeq double %5, 2.000000e+00
  %6 = load double, double* %x, align 8
  br i1 %cmp2, label %if.then2, label %if.else2

; fsub (x, (fmul y, z))
if.then2:                                         ; preds = %entry
; CHECK-LABEL: if.then2:
; CHECK:   %7 = fmul fast double %5, 3.000000e+00
; CHECK-NEXT:   %mul2 = fsub fast double %6, %7
  %7 = load double, double* %a, align 8
  %8 = fmul fast double %6, 3.0000000e+00
  %mul2 = fsub fast double %7, %8
  store double %mul2, double* %y, align 8
  br label %if.end2

; fsub (fneg((fmul x, y)), const)
if.else2:                                         ; preds = %entry
; CHECK-LABEL: if.else2:
; CHECK:   %mul3 = fmul fast double %5, 3.000000e+00
; CHECK-NEXT:   %neg = fsub fast double 0.000000e+00, %mul3
; CHECK-NEXT:   %sub2 = fsub fast double %neg, 3.000000e+00
  %mul3 = fmul fast double %6, 3.0000000e+00
  %neg = fsub fast double 0.0000000e+00, %mul3
  %sub2 = fsub fast double %neg, 3.0000000e+00
  store double %sub2, double* %y, align 8
  br label %if.end2

if.end2:                                           ; preds = %if.else, %if.then
  %9 = load double, double* %x, align 8
  %10 = load double, double* %y, align 8
  %add = fadd fast double %9, %10
  %11 = load double, double* %a, align 8
  %add2 = fadd fast double %add, %11
  ret double %add2
}

