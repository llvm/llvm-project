; RUN: opt -S -instcombine < %s | FileCheck %s

; CHECK-LABEL: @t1
; CHECK-NEXT: fcmp oge float %a, 5.000000e+00
; CHECK-NEXT: select i1 %.inv, float 5.000000e+00, float %a
; CHECK-NEXT: fpext float %1 to double
define double @t1(float %a) {
  ; This is the canonical form for a type-changing min/max.
  %1 = fcmp ult float %a, 5.0
  %2 = select i1 %1, float %a, float 5.0
  %3 = fpext float %2 to double
  ret double %3
}

; CHECK-LABEL: @t2
; CHECK-NEXT: fcmp oge float %a, 5.000000e+00
; CHECK-NEXT: select i1 %.inv, float 5.000000e+00, float %a
; CHECK-NEXT: fpext float %1 to double
define double @t2(float %a) {
  ; Check this is converted into canonical form, as above.
  %1 = fcmp ult float %a, 5.0
  %2 = fpext float %a to double
  %3 = select i1 %1, double %2, double 5.0
  ret double %3
}

; CHECK-LABEL: @t4
; CHECK-NEXT: fcmp oge double %a, 5.000000e+00
; CHECK-NEXT: select i1 %.inv, double 5.000000e+00, double %a
; CHECK-NEXT: fptrunc double %1 to float
define float @t4(double %a) {
  ; Same again, with trunc.
  %1 = fcmp ult double %a, 5.0
  %2 = fptrunc double %a to float
  %3 = select i1 %1, float %2, float 5.0
  ret float %3
}

; CHECK-LABEL: @t5
; CHECK-NEXT: fcmp ult float %a, 5.000000e+00
; CHECK-NEXT: fpext float %a to double
; CHECK-NEXT: select i1 %1, double %2, double 5.001
define double @t5(float %a) {
  ; different values, should not be converted.
  %1 = fcmp ult float %a, 5.0
  %2 = fpext float %a to double
  %3 = select i1 %1, double %2, double 5.001
  ret double %3
}

; CHECK-LABEL: @t6
; CHECK-NEXT: fcmp ult float %a, -0.0
; CHECK-NEXT: fpext float %a to double
; CHECK-NEXT: select i1 %1, double %2, double 0.0
define double @t6(float %a) {
  ; Signed zero, should not be converted
  %1 = fcmp ult float %a, -0.0
  %2 = fpext float %a to double
  %3 = select i1 %1, double %2, double 0.0
  ret double %3
}

; CHECK-LABEL: @t7
; CHECK-NEXT: fcmp ult float %a, 0.0
; CHECK-NEXT: fpext float %a to double
; CHECK-NEXT: select i1 %1, double %2, double -0.0
define double @t7(float %a) {
  ; Signed zero, should not be converted
  %1 = fcmp ult float %a, 0.0
  %2 = fpext float %a to double
  %3 = select i1 %1, double %2, double -0.0
  ret double %3
}

; CHECK-LABEL: @t8
; CHECK-NEXT: fcmp oge float %a, 5.000000e+00
; CHECK-NEXT: select i1 %.inv, float 5.000000e+00, float %a
; CHECK-NEXT: fptoui float %1 to i64
define i64 @t8(float %a) {
  %1 = fcmp ult float %a, 5.0
  %2 = fptoui float %a to i64
  %3 = select i1 %1, i64 %2, i64 5
  ret i64 %3
}

; CHECK-LABEL: @t9
; CHECK-NEXT: fcmp oge float %a, 0.000000e+00
; CHECK-NEXT: select i1 %.inv, float 0.000000e+00, float %a
; CHECK-NEXT: fptosi float %1 to i8
define i8 @t9(float %a) {
  %1 = fcmp ult float %a, 0.0
  %2 = fptosi float %a to i8
  %3 = select i1 %1, i8 %2, i8 0
  ret i8 %3
}

; CHECK-LABEL: @t11
; CHECK-NEXT: fcmp fast oge float %b, %a
; CHECK-NEXT: select i1 %.inv, float %a, float %b
; CHECK-NEXT: fptosi
define i8 @t11(float %a, float %b) {
  ; Either operand could be NaN, but fast modifier applied.
  %1 = fcmp fast ult float %b, %a
  %2 = fptosi float %a to i8
  %3 = fptosi float %b to i8
  %4 = select i1 %1, i8 %3, i8 %2
  ret i8 %4
}

; CHECK-LABEL: @t12
; CHECK-NEXT: fcmp nnan oge float %b, %a
; CHECK-NEXT: select i1 %.inv, float %a, float %b
; CHECK-NEXT: fptosi float %.v to i8
define i8 @t12(float %a, float %b) {
  ; Either operand could be NaN, but nnan modifier applied.
  %1 = fcmp nnan ult float %b, %a
  %2 = fptosi float %a to i8
  %3 = fptosi float %b to i8
  %4 = select i1 %1, i8 %3, i8 %2
  ret i8 %4
}

; CHECK-LABEL: @t13
; CHECK-NEXT: fcmp ult float %a, 1.500000e+00
; CHECK-NEXT: fptosi float %a to i8
; CHECK-NEXT: select i1 %1, i8 %2, i8 1
define i8 @t13(float %a) {
  ; Float and int values do not match.
  %1 = fcmp ult float %a, 1.5
  %2 = fptosi float %a to i8
  %3 = select i1 %1, i8 %2, i8 1
  ret i8 %3
}

; CHECK-LABEL: @t14
; CHECK-NEXT: fcmp ule float %a, 0.000000e+00
; CHECK-NEXT: fptosi float %a to i8
; CHECK-NEXT: select i1 %1, i8 %2, i8 0
define i8 @t14(float %a) {
  ; <= comparison, where %a could be -0.0. Not safe.
  %1 = fcmp ule float %a, 0.0
  %2 = fptosi float %a to i8
  %3 = select i1 %1, i8 %2, i8 0
  ret i8 %3
}

; CHECK-LABEL: @t15
; CHECK-NEXT: fcmp nsz oge float %a, 0.000000e+00
; CHECK-NEXT: select i1 %.inv, float 0.000000e+00, float %a
; CHECK-NEXT: fptosi float %1 to i8
define i8 @t15(float %a) {
  %1 = fcmp nsz ule float %a, 0.0
  %2 = fptosi float %a to i8
  %3 = select i1 %1, i8 %2, i8 0
  ret i8 %3
}

; CHECK-LABEL: @t16
; CHECK:  %[[cmp:.*]] = icmp sgt i32 %x, 0
; CHECK:  %[[cst:.*]] = sitofp i32 %x to double
; CHECK:  %[[sel:.*]] = select i1 %[[cmp]], double %[[cst]], double 5.000000e-01
; CHECK:  ret double %[[sel]]
define double @t16(i32 %x) {
entry:
  %cmp = icmp sgt i32 %x, 0
  %cst = sitofp i32 %x to double
  %sel = select i1 %cmp, double %cst, double 5.000000e-01
  ret double %sel
}

; CHECK-LABEL: @t17
; CHECK:  %[[cmp:.*]] = icmp sgt i32 %x, 2
; CHECK:  %[[sel:.*]] = select i1 %[[cmp]], i32 %x, i32 2
; CHECK:  %[[cst:.*]] = sitofp i32 %[[sel]] to double
; CHECK:  ret double %[[cst]]
define double @t17(i32 %x) {
entry:
  %cmp = icmp sgt i32 %x, 2
  %cst = sitofp i32 %x to double
  %sel = select i1 %cmp, double %cst, double 2.0
  ret double %sel
}
