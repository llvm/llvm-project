; RUN: llc -mtriple=mipsel -mcpu=mips32 -mattr=+mips16 < %s | FileCheck %s
; RUN: llc -mtriple=mipsel -mcpu=mips32 -mattr=+mips16 -relocation-model=pic < %s | FileCheck %s

; Constrained FP intrinsics are lowered like the unconstrained operations and
; do not get FP call stubs.

; CHECK-NOT: __call_stub_fp_llvm.experimental.constrained

define float @fadd(float %a, float %b) #0 {
; CHECK-LABEL: fadd:
; CHECK: __mips16_addsf3
  %r = call float @llvm.experimental.constrained.fadd.f32(float %a, float %b, metadata !"round.dynamic", metadata !"fpexcept.strict") #0
  ret float %r
}

define double @fdiv(double %a, double %b) #0 {
; CHECK-LABEL: fdiv:
; CHECK: __mips16_divdf3
  %r = call double @llvm.experimental.constrained.fdiv.f64(double %a, double %b, metadata !"round.dynamic", metadata !"fpexcept.strict") #0
  ret double %r
}

define i32 @fcmps(float %a, float %b) #0 {
; CHECK-LABEL: fcmps:
; CHECK: __mips16_ltsf2
  %c = call i1 @llvm.experimental.constrained.fcmps.f32(float %a, float %b, metadata !"olt", metadata !"fpexcept.strict") #0
  %z = zext i1 %c to i32
  ret i32 %z
}

define i32 @fptosi(double %a) #0 {
; CHECK-LABEL: fptosi:
; CHECK: __mips16_fix_truncdfsi
  %r = call i32 @llvm.experimental.constrained.fptosi.i32.f64(double %a, metadata !"fpexcept.strict") #0
  ret i32 %r
}

define float @sitofp(i32 %a) #0 {
; CHECK-LABEL: sitofp:
; CHECK: __mips16_floatsisf
  %r = call float @llvm.experimental.constrained.sitofp.f32.i32(i32 %a, metadata !"round.dynamic", metadata !"fpexcept.strict") #0
  ret float %r
}

attributes #0 = { strictfp }
