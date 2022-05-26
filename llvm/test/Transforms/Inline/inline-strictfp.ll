; RUN: opt -inline %s -S | FileCheck %s


; Ordinary function is inlined into strictfp function.

define float @inlined_01(float %a) {
entry:
  %add = fadd float %a, %a
  ret float %add
}

define float @host_02(float %a) #0 {
entry:
  %0 = call float @inlined_01(float %a) #0
  %add = call float @llvm.experimental.constrained.fadd.f32(float %0, float 2.000000e+00, metadata !"round.dynamic", metadata !"fpexcept.strict") #0
  ret float %add
; CHECK-LABEL: @host_02
; CHECK: call float @llvm.experimental.constrained.fadd.f32(float {{.*}}, float {{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore") #0
; CHECK: call float @llvm.experimental.constrained.fadd.f32(float {{.*}}, float 2.000000e+00, metadata !"round.dynamic", metadata !"fpexcept.strict") #0
}


; strictfp function is inlined into another strictfp function.

define float @inlined_03(float %a) #0 {
entry:
  %add = call float @llvm.experimental.constrained.fadd.f32(float %a, float %a, metadata !"round.downward", metadata !"fpexcept.maytrap") #0
  ret float %add
}

define float @host_04(float %a) #0 {
entry:
  %0 = call float @inlined_03(float %a) #0
  %add = call float @llvm.experimental.constrained.fadd.f32(float %0, float 2.000000e+00, metadata !"round.dynamic", metadata !"fpexcept.strict") #0
  ret float %add
; CHECK-LABEL: @host_04
; CHECK: call float @llvm.experimental.constrained.fadd.f32(float {{.*}}, float {{.*}}, metadata !"round.downward", metadata !"fpexcept.maytrap") #0
; CHECK: call float @llvm.experimental.constrained.fadd.f32(float {{.*}}, float 2.000000e+00, metadata !"round.dynamic", metadata !"fpexcept.strict") #0
}


; strictfp function is NOT inlined into ordinary function.

define float @inlined_05(float %a) strictfp {
entry:
  %add = call float @llvm.experimental.constrained.fadd.f32(float %a, float %a, metadata !"round.downward", metadata !"fpexcept.maytrap") #0
  ret float %add
}

define float @host_06(float %a) {
entry:
  %0 = call float @inlined_05(float %a)
  %add = fadd float %0, 2.000000e+00
  ret float %add
; CHECK-LABEL: @host_06
; CHECK: call float @inlined_05(float %a)
; CHECK: fadd float %0, 2.000000e+00
}


; Calls in inlined function must get strictfp attribute.

declare float @func_ext(float);

define float @inlined_07(float %a) {
entry:
  %0 = call float @func_ext(float %a)
  %add = fadd float %0, %a

  ret float %add
}

define float @host_08(float %a) #0 {
entry:
  %0 = call float @inlined_07(float %a) #0
  %add = call float @llvm.experimental.constrained.fadd.f32(float %0, float 2.000000e+00, metadata !"round.dynamic", metadata !"fpexcept.strict") #0
  ret float %add
; CHECK-LABEL: @host_08
; CHECK: call float @func_ext(float {{.*}}) #0
; CHECK: call float @llvm.experimental.constrained.fadd.f32(float {{.*}}, float {{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore") #0
; CHECK: call float @llvm.experimental.constrained.fadd.f32(float {{.*}}, float 2.000000e+00, metadata !"round.dynamic", metadata !"fpexcept.strict") #0
}


; Cloning particular instructions.

; fpext has two overloaded types.
define double @inlined_09(float %a) {
entry:
  %t = fpext float %a to double
  ret double %t
}

define double @host_10(float %a) #0 {
entry:
  %0 = call double @inlined_09(float %a) #0
  %add = call double @llvm.experimental.constrained.fadd.f64(double %0, double 2.000000e+00, metadata !"round.dynamic", metadata !"fpexcept.strict") #0
  ret double %add
; CHECK-LABEL: @host_10
; CHECK: call double @llvm.experimental.constrained.fpext.f64.f32(float {{.*}}, metadata !"fpexcept.ignore") #0
; CHECK: call double @llvm.experimental.constrained.fadd.f64(double {{.*}}, double 2.000000e+00, metadata !"round.dynamic", metadata !"fpexcept.strict") #0
}

; fcmp does not depend on rounding mode and has metadata argument.
define i1 @inlined_11(float %a, float %b) {
entry:
  %t = fcmp oeq float %a, %b
  ret i1 %t
}

define i1 @host_12(float %a, float %b) #0 {
entry:
  %add = call float @llvm.experimental.constrained.fadd.f32(float %a, float %b, metadata !"round.dynamic", metadata !"fpexcept.strict") #0
  %cmp = call i1 @inlined_11(float %a, float %b) #0
  ret i1 %cmp
; CHECK-LABEL: @host_12
; CHECK: call float @llvm.experimental.constrained.fadd.f32(float %a, float %b, metadata !"round.dynamic", metadata !"fpexcept.strict") #0
; CHECK: call i1 @llvm.experimental.constrained.fcmp.f32(float {{.*}}, metadata !"oeq", metadata !"fpexcept.ignore") #0
}

; Intrinsic 'ceil' has constrained variant.
define float @inlined_13(float %a) {
entry:
  %t = call float @llvm.ceil.f32(float %a)
  ret float %t
}

define float @host_14(float %a) #0 {
entry:
  %0 = call float @inlined_13(float %a) #0
  %add = call float @llvm.experimental.constrained.fadd.f32(float %0, float 2.000000e+00, metadata !"round.dynamic", metadata !"fpexcept.strict") #0
  ret float %add
; CHECK-LABEL: @host_14
; CHECK: call float @llvm.experimental.constrained.ceil.f32(float %a, metadata !"fpexcept.ignore") #0
; CHECK: call float @llvm.experimental.constrained.fadd.f32(float {{.*}}, float 2.000000e+00, metadata !"round.dynamic", metadata !"fpexcept.strict") #0
}

attributes #0 = { strictfp }

declare float  @llvm.experimental.constrained.fadd.f32(float, float, metadata, metadata)
declare double @llvm.experimental.constrained.fadd.f64(double, double, metadata, metadata)
declare double @llvm.experimental.constrained.fpext.f64.f32(float, metadata)
declare i1     @llvm.experimental.constrained.fcmp.f32(float, float, metadata, metadata)
declare float  @llvm.experimental.constrained.ceil.f32(float, metadata)
declare float  @llvm.ceil.f32(float)
