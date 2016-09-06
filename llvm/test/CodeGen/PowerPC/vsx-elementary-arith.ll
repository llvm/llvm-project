; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr8 | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr8 | FileCheck %s
@a = global float 3.000000e+00, align 4
@b = global float 4.000000e+00, align 4
@c = global double 3.000000e+00, align 8
@d = global double 4.000000e+00, align 8

; Function Attrs: nounwind
define float @emit_xsaddsp() {
entry:
  %0 = load float, float* @a, align 4
  %1 = load float, float* @b, align 4
  %add = fadd float %0, %1
  ret float %add
; CHECK-LABEL: @emit_xsaddsp
; CHECK: xsaddsp {{[0-9]+}}
}

; Function Attrs: nounwind
define float @emit_xssubsp() {
entry:
  %0 = load float, float* @a, align 4
  %1 = load float, float* @b, align 4
  %sub = fsub float %0, %1
  ret float %sub
; CHECK-LABEL: @emit_xssubsp
; CHECK: xssubsp {{[0-9]+}}
}

; Function Attrs: nounwind
define float @emit_xsdivsp() {
entry:
  %0 = load float, float* @a, align 4
  %1 = load float, float* @b, align 4
  %div = fdiv float %0, %1
  ret float %div
; CHECK-LABEL: @emit_xsdivsp
; CHECK: xsdivsp {{[0-9]+}}
}

; Function Attrs: nounwind
define float @emit_xsmulsp() {
entry:
  %0 = load float, float* @a, align 4
  %1 = load float, float* @b, align 4
  %mul = fmul float %0, %1
  ret float %mul
; CHECK-LABEL: @emit_xsmulsp
; CHECK: xsmulsp {{[0-9]+}}
}

; Function Attrs: nounwind
define float @emit_xssqrtsp() {
entry:
  %0 = load float, float* @b, align 4
  %call = call float @sqrtf(float %0)
  ret float %call
; CHECK-LABEL: @emit_xssqrtsp
; CHECK: xssqrtsp {{[0-9]+}}
}

; Function Attrs: nounwind
declare float @sqrtf(float)

; Function Attrs: nounwind
define double @emit_xsadddp() {
entry:
  %0 = load double, double* @c, align 8
  %1 = load double, double* @d, align 8
  %add = fadd double %0, %1
  ret double %add
; CHECK-LABEL: @emit_xsadddp
; CHECK: xsadddp {{[0-9]+}}
}

; Function Attrs: nounwind
define double @emit_xssubdp() {
entry:
  %0 = load double, double* @c, align 8
  %1 = load double, double* @d, align 8
  %sub = fsub double %0, %1
  ret double %sub
; CHECK-LABEL: @emit_xssubdp
; CHECK: xssubdp {{[0-9]+}}
}

; Function Attrs: nounwind
define double @emit_xsdivdp() {
entry:
  %0 = load double, double* @c, align 8
  %1 = load double, double* @d, align 8
  %div = fdiv double %0, %1
  ret double %div
; CHECK-LABEL: @emit_xsdivdp
; CHECK: xsdivdp {{[0-9]+}}
}

; Function Attrs: nounwind
define double @emit_xsmuldp() {
entry:
  %0 = load double, double* @c, align 8
  %1 = load double, double* @d, align 8
  %mul = fmul double %0, %1
  ret double %mul
; CHECK-LABEL: @emit_xsmuldp
; CHECK: xsmuldp {{[0-9]+}}
}

; Function Attrs: nounwind
define double @emit_xssqrtdp() {
entry:
  %0 = load double, double* @d, align 8
  %call = call double @sqrt(double %0)
  ret double %call
; CHECK-LABEL: @emit_xssqrtdp
; CHECK: xssqrtdp {{[0-9]+}}
}

; Vector forms
; Function Attrs: nounwind
define <4 x float> @emit_xvrsqrtesp() {
entry:
; CHECK-LABEL: @emit_xvrsqrtesp
  %vf = alloca <4 x float>, align 16
  %vfr = alloca <4 x float>, align 16
  %0 = load <4 x float>, <4 x float>* %vf, align 16
  %call = call <4 x float> @llvm.ppc.vsx.xvrsqrtesp(<4 x float> %0)
; CHECK: xvrsqrtesp {{[0-9]+}}, {{[0-9]+}}
  ret <4 x float> %call
}

; Function Attrs: nounwind
define <2 x double> @emit_xvrsqrtedp() {
entry:
; CHECK-LABEL: @emit_xvrsqrtedp
  %vd = alloca <2 x double>, align 16
  %vdr = alloca <2 x double>, align 16
  %0 = load <2 x double>, <2 x double>* %vd, align 16
  %call = call <2 x double> @llvm.ppc.vsx.xvrsqrtedp(<2 x double> %0)
  ret <2 x double> %call
; CHECK: xvrsqrtedp {{[0-9]+}}, {{[0-9]+}}
}

; Function Attrs: nounwind
declare double @sqrt(double)

; Function Attrs: nounwind readnone
declare <4 x float> @llvm.ppc.vsx.xvrsqrtesp(<4 x float>)

; Function Attrs: nounwind readnone
declare <2 x double> @llvm.ppc.vsx.xvrsqrtedp(<2 x double>)
