; RUN: llc < %s -march=nvptx | FileCheck %s --check-prefixes=CHECK,CHECK-NONAN
; RUN: llc < %s -march=nvptx -mcpu=sm_80 | FileCheck %s --check-prefixes=CHECK,CHECK-NAN

; ---- minimum ----

; CHECK-LABEL: minimum_half
define half @minimum_half(half %a) #0 {
  ; CHECK-NONAN: setp
  ; CHECK-NONAN: selp.b16
  ; CHECK-NAN: min.NaN.f16
  %p = fcmp ult half %a, 0.0
  %x = select i1 %p, half %a, half 0.0
  ret half %x
}

; CHECK-LABEL: minimum_float
define float @minimum_float(float %a) #0 {
  ; CHECK-NONAN: setp
  ; CHECK-NONAN: selp.f32
  ; CHECK-NAN: min.NaN.f32
  %p = fcmp ult float %a, 0.0
  %x = select i1 %p, float %a, float 0.0
  ret float %x
}

; CHECK-LABEL: minimum_double
define double @minimum_double(double %a) #0 {
  ; CHECK-NONAN: setp
  ; CHECK-NONAN: selp.f64
  ; CHECK-NAN: min.NaN.f64
  %p = fcmp ult double %a, 0.0
  %x = select i1 %p, double %a, double 0.0
  ret double %x
}

; CHECK-LABEL: minimum_v2half
define <2 x half> @minimum_v2half(<2 x half> %a) #0 {
  ; CHECK-NONAN-DAG: setp
  ; CHECK-NONAN-DAG: setp
  ; CHECK-NONAN-DAG: selp.b16
  ; CHECK-NONAN-DAG: selp.b16
  ; CHECK-NAN: min.NaN.f16x2
  %p = fcmp ult <2 x half> %a, zeroinitializer
  %x = select <2 x i1> %p, <2 x half> %a, <2 x half> zeroinitializer
  ret <2 x half> %x
}

; ---- maximum ----

; CHECK-LABEL: maximum_half
define half @maximum_half(half %a) #0 {
  ; CHECK-NONAN: setp
  ; CHECK-NONAN: selp.b16
  ; CHECK-NAN: max.NaN.f16
  %p = fcmp ugt half %a, 0.0
  %x = select i1 %p, half %a, half 0.0
  ret half %x
}

; CHECK-LABEL: maximum_float
define float @maximum_float(float %a) #0 {
  ; CHECK-NONAN: setp
  ; CHECK-NONAN: selp.f32
  ; CHECK-NAN: max.NaN.f32
  %p = fcmp ugt float %a, 0.0
  %x = select i1 %p, float %a, float 0.0
  ret float %x
}

; CHECK-LABEL: maximum_double
define double @maximum_double(double %a) #0 {
  ; CHECK-NONAN: setp
  ; CHECK-NONAN: selp.f64
  ; CHECK-NAN: max.NaN.f64
  %p = fcmp ugt double %a, 0.0
  %x = select i1 %p, double %a, double 0.0
  ret double %x
}

; CHECK-LABEL: maximum_v2half
define <2 x half> @maximum_v2half(<2 x half> %a) #0 {
  ; CHECK-NONAN-DAG: setp
  ; CHECK-NONAN-DAG: setp
  ; CHECK-NONAN-DAG: selp.b16
  ; CHECK-NONAN-DAG: selp.b16
  ; CHECK-NAN: max.NaN.f16x2
  %p = fcmp ugt <2 x half> %a, zeroinitializer
  %x = select <2 x i1> %p, <2 x half> %a, <2 x half> zeroinitializer
  ret <2 x half> %x
}
