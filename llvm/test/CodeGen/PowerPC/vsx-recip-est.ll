; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr8 | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr8 | FileCheck %s
@a = global float 3.000000e+00, align 4
@b = global float 4.000000e+00, align 4
@c = global double 3.000000e+00, align 8
@d = global double 4.000000e+00, align 8

; Function Attrs: nounwind
define float @emit_xsresp() {
entry:
  %0 = load float, ptr @a, align 4
  %1 = load float, ptr @b, align 4
  %div = fdiv arcp ninf float %0, %1
  ret float %div
; CHECK-LABEL: @emit_xsresp
; CHECK: xsresp {{[0-9]+}}
}

; Function Attrs: nounwind
define float @emit_xsrsqrtesp(float %f) {
entry:
  %f.addr = alloca float, align 4
  store float %f, ptr %f.addr, align 4
  %0 = load float, ptr %f.addr, align 4
  %1 = load float, ptr @b, align 4
  %2 = call float @llvm.sqrt.f32(float %1)
  %div = fdiv arcp float %0, %2
  ret float %div
; CHECK-LABEL: @emit_xsrsqrtesp
; CHECK: xsrsqrtesp {{[0-9]+}}
}

; Function Attrs: nounwind readnone
declare float @llvm.sqrt.f32(float)

; Function Attrs: nounwind
define double @emit_xsredp() {
entry:
  %0 = load double, ptr @c, align 8
  %1 = load double, ptr @d, align 8
  %div = fdiv arcp ninf double %0, %1
  ret double %div
; CHECK-LABEL: @emit_xsredp
; CHECK: xsredp {{[0-9]+}}
}

; Function Attrs: nounwind
define double @emit_xsrsqrtedp(double %f) {
entry:
  %f.addr = alloca double, align 8
  store double %f, ptr %f.addr, align 8
  %0 = load double, ptr %f.addr, align 8
  %1 = load double, ptr @d, align 8
  %2 = call double @llvm.sqrt.f64(double %1)
  %div = fdiv arcp double %0, %2
  ret double %div
; CHECK-LABEL: @emit_xsrsqrtedp
; CHECK: xsrsqrtedp {{[0-9]+}}
}

; Function Attrs: nounwind readnone
declare double @llvm.sqrt.f64(double) #1
