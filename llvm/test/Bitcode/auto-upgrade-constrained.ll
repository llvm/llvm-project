; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: llvm-dis %s.bc -o - | FileCheck %s

define float @test_fadd(float %a, float %b) strictfp {
  %res = call float @llvm.experimental.constrained.fadd.f32(float %a, float %b, metadata !"round.dynamic", metadata !"fpexcept.ignore")
  ret float %res
}
; CHECK-LABEL: define float @test_fadd(
; CHECK: call float @llvm.experimental.constrained.fadd.f32(float {{.*}}, float {{.*}}, metadata !"round.dynamic", metadata !"fpexcept.ignore") #[[ATTR0:[0-9]+]] [ "fpe.control"(metadata !"dyn"), "fpe.except"(metadata !"ignore") ]


define float @test_fsub(float %a, float %b) strictfp {
  %res = call float @llvm.experimental.constrained.fsub.f32(float %a, float %b, metadata !"round.tonearest", metadata !"fpexcept.ignore")
  ret float %res
}
; CHECK-LABEL: define float @test_fsub(
; CHECK: call float @llvm.experimental.constrained.fsub.f32(float {{.*}}, float {{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore") #[[ATTR0]] [ "fpe.control"(metadata !"rte"), "fpe.except"(metadata !"ignore") ]

define float @test_fmul(float %a, float %b) strictfp {
  %res = call float @llvm.experimental.constrained.fmul.f32(float %a, float %b, metadata !"round.downward", metadata !"fpexcept.ignore")
  ret float %res
}
; CHECK-LABEL: define float @test_fmul(
; CHECK: call float @llvm.experimental.constrained.fmul.f32(float {{.*}}, float {{.*}}, metadata !"round.downward", metadata !"fpexcept.ignore") #[[ATTR0]] [ "fpe.control"(metadata !"rtn"), "fpe.except"(metadata !"ignore") ]

define float @test_fdiv(float %a, float %b) strictfp {
  %res = call float @llvm.experimental.constrained.fdiv.f32(float %a, float %b, metadata !"round.upward", metadata !"fpexcept.ignore")
  ret float %res
}
; CHECK-LABEL: define float @test_fdiv(
; CHECK: call float @llvm.experimental.constrained.fdiv.f32(float {{.*}}, float {{.*}}, metadata !"round.upward", metadata !"fpexcept.ignore") #[[ATTR0]] [ "fpe.control"(metadata !"rtp"), "fpe.except"(metadata !"ignore") ]

define float @test_frem(float %a, float %b) strictfp {
  %res = call float @llvm.experimental.constrained.frem.f32(float %a, float %b, metadata !"round.dynamic", metadata !"fpexcept.ignore")
  ret float %res
}
; CHECK-LABEL: define float @test_frem(
; CHECK: call float @llvm.experimental.constrained.frem.f32(float {{.*}}, float {{.*}}, metadata !"round.dynamic", metadata !"fpexcept.ignore") #[[ATTR0]] [ "fpe.control"(metadata !"dyn"), "fpe.except"(metadata !"ignore") ]

define float @test_fma(float %a, float %b, float %c) strictfp {
  %res = call float @llvm.experimental.constrained.fma.f32(float %a, float %b, float %c, metadata !"round.towardzero", metadata !"fpexcept.ignore")
  ret float %res
}
; CHECK-LABEL: define float @test_fma(
; CHECK: call float @llvm.experimental.constrained.fma.f32(float {{.*}}, float {{.*}}, float {{.*}}, metadata !"round.towardzero", metadata !"fpexcept.ignore") #[[ATTR0]] [ "fpe.control"(metadata !"rtz"), "fpe.except"(metadata !"ignore") ]

define float @test_fmuladd(float %a, float %b, float %c) strictfp {
  %res = call float @llvm.experimental.constrained.fmuladd.f32(float %a, float %b, float %c, metadata !"round.tonearestaway", metadata !"fpexcept.ignore")
  ret float %res
}
; CHECK-LABEL: define float @test_fmuladd(
; CHECK: call float @llvm.experimental.constrained.fmuladd.f32(float {{.*}}, float {{.*}}, float {{.*}}, metadata !"round.tonearestaway", metadata !"fpexcept.ignore") #[[ATTR0]] [ "fpe.control"(metadata !"rmm"), "fpe.except"(metadata !"ignore") ]

define i32 @test_fptosi(float %a) strictfp {
  %res = call i32 @llvm.experimental.constrained.fptosi.i32.f32(float %a, metadata !"fpexcept.ignore")
  ret i32 %res
}
; CHECK-LABEL: define i32 @test_fptosi(
; CHECK: call i32 @llvm.experimental.constrained.fptosi.i32.f32(float {{.*}}, metadata !"fpexcept.ignore") #[[ATTR0]] [ "fpe.except"(metadata !"ignore") ]

define i32 @test_fptoui(float %a) strictfp {
  %res = call i32 @llvm.experimental.constrained.fptoui.f32.i32(float %a, metadata !"fpexcept.strict")
  ret i32 %res
}
; CHECK-LABEL: define i32 @test_fptoui(
; CHECK: call i32 @llvm.experimental.constrained.fptoui.i32.f32(float {{.*}}, metadata !"fpexcept.strict") #[[ATTR0]] [ "fpe.except"(metadata !"strict") ]

define float @test_sitofp(i32 %a) strictfp {
  %res = call float @llvm.experimental.constrained.sitofp.i32.f32(i32 %a, metadata !"round.dynamic", metadata !"fpexcept.ignore")
  ret float %res
}
; CHECK-LABEL: define float @test_sitofp(
; CHECK: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 {{.*}}, metadata !"round.dynamic", metadata !"fpexcept.ignore") #[[ATTR0]] [ "fpe.control"(metadata !"dyn"), "fpe.except"(metadata !"ignore") ]

define float @test_uitofp(i32 %a) strictfp {
  %res = call float @llvm.experimental.constrained.uitofp.i32.f32(i32 %a, metadata !"round.dynamic", metadata !"fpexcept.ignore")
  ret float %res
}
; CHECK-LABEL: define float @test_uitofp(
; CHECK: call float @llvm.experimental.constrained.uitofp.f32.i32(i32 {{.*}}, metadata !"round.dynamic", metadata !"fpexcept.ignore") #[[ATTR0]] [ "fpe.control"(metadata !"dyn"), "fpe.except"(metadata !"ignore") ]

define float @test_fptrunc(double %a) strictfp {
  %res = call float @llvm.experimental.constrained.fptrunc.f32.f64(double %a, metadata !"round.dynamic", metadata !"fpexcept.ignore")
  ret float %res
}
; CHECK-LABEL: define float @test_fptrunc(
; CHECK: call float @llvm.experimental.constrained.fptrunc.f32.f64(double {{.*}}, metadata !"round.dynamic", metadata !"fpexcept.ignore") #[[ATTR0]] [ "fpe.control"(metadata !"dyn"), "fpe.except"(metadata !"ignore") ]

define double @test_fpext(float %a) strictfp {
  %res = call double @llvm.experimental.constrained.fpext.f64.f32(float %a, metadata !"fpexcept.ignore")
  ret double %res
}
; CHECK-LABEL: define double @test_fpext(
; CHECK: call double @llvm.experimental.constrained.fpext.f64.f32(float {{.*}}, metadata !"fpexcept.ignore") #[[ATTR0]] [ "fpe.except"(metadata !"ignore") ]

define float @test_sqrt(float %a) strictfp {
  %res = call float @llvm.experimental.constrained.sqrt.f32(float %a, metadata !"round.dynamic", metadata !"fpexcept.strict")
  ret float %res
}
; CHECK-LABEL: define float @test_sqrt(
; CHECK: call float @llvm.experimental.constrained.sqrt.f32(float {{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict") #[[ATTR0]] [ "fpe.control"(metadata !"dyn"), "fpe.except"(metadata !"strict") ]

define float @test_powi(float %a, i32 %b) strictfp {
  %res = call float @llvm.experimental.constrained.powi.f32.i32(float %a, i32 %b, metadata !"round.dynamic", metadata !"fpexcept.strict")
  ret float %res
}
; CHECK-LABEL: define float @test_powi(
; CHECK: call float @llvm.experimental.constrained.powi.f32(float {{.*}}, i32 %b, metadata !"round.dynamic", metadata !"fpexcept.strict") #[[ATTR0]] [ "fpe.control"(metadata !"dyn"), "fpe.except"(metadata !"strict") ]

define float @test_ldexp(float %a, i32 %b) strictfp {
  %res = call float @llvm.experimental.constrained.ldexp.f32.i32(float %a, i32 %b, metadata !"round.dynamic", metadata !"fpexcept.strict")
  ret float %res
}
; CHECK-LABEL: define float @test_ldexp(
; CHECK: call float @llvm.experimental.constrained.ldexp.f32.i32(float {{.*}}, i32 {{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict") #[[ATTR0]] [ "fpe.control"(metadata !"dyn"), "fpe.except"(metadata !"strict") ]

define float @test_asin(float %a) strictfp {
  %res = call float @llvm.experimental.constrained.asin.f32(float %a, metadata !"round.dynamic", metadata !"fpexcept.strict")
  ret float %res
}
; CHECK-LABEL: define float @test_asin(
; CHECK: call float @llvm.experimental.constrained.asin.f32(float {{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict") #[[ATTR0]] [ "fpe.control"(metadata !"dyn"), "fpe.except"(metadata !"strict") ]

define float @test_acos(float %a) strictfp {
  %res = call float @llvm.experimental.constrained.acos.f32(float %a, metadata !"round.dynamic", metadata !"fpexcept.strict")
  ret float %res
}
; CHECK-LABEL: define float @test_acos(
; CHECK: call float @llvm.experimental.constrained.acos.f32(float {{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict") #[[ATTR0]] [ "fpe.control"(metadata !"dyn"), "fpe.except"(metadata !"strict") ]

define float @test_atan(float %a) strictfp {
  %res = call float @llvm.experimental.constrained.atan.f32(float %a, metadata !"round.dynamic", metadata !"fpexcept.strict")
  ret float %res
}
; CHECK-LABEL: define float @test_atan(
; CHECK: call float @llvm.experimental.constrained.atan.f32(float {{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict") #[[ATTR0]] [ "fpe.control"(metadata !"dyn"), "fpe.except"(metadata !"strict") ]

define float @test_sin(float %a) strictfp {
  %res = call float @llvm.experimental.constrained.sin.f32(float %a, metadata !"round.dynamic", metadata !"fpexcept.strict")
  ret float %res
}
; CHECK-LABEL: define float @test_sin(
; CHECK: call float @llvm.experimental.constrained.sin.f32(float {{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict") #[[ATTR0]] [ "fpe.control"(metadata !"dyn"), "fpe.except"(metadata !"strict") ]

define float @test_cos(float %a) strictfp {
  %res = call float @llvm.experimental.constrained.cos.f32(float %a, metadata !"round.dynamic", metadata !"fpexcept.strict")
  ret float %res
}
; CHECK-LABEL: define float @test_cos(
; CHECK: call float @llvm.experimental.constrained.cos.f32(float {{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict") #[[ATTR0]] [ "fpe.control"(metadata !"dyn"), "fpe.except"(metadata !"strict") ]

define float @test_tan(float %a) strictfp {
  %res = call float @llvm.experimental.constrained.tan.f32(float %a, metadata !"round.dynamic", metadata !"fpexcept.strict")
  ret float %res
}
; CHECK-LABEL: define float @test_tan(
; CHECK: call float @llvm.experimental.constrained.tan.f32(float {{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict") #[[ATTR0]] [ "fpe.control"(metadata !"dyn"), "fpe.except"(metadata !"strict") ]

define float @test_sinh(float %a) strictfp {
  %res = call float @llvm.experimental.constrained.sinh.f32(float %a, metadata !"round.dynamic", metadata !"fpexcept.strict")
  ret float %res
}
; CHECK-LABEL: define float @test_sinh(
; CHECK: call float @llvm.experimental.constrained.sinh.f32(float {{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict") #[[ATTR0]] [ "fpe.control"(metadata !"dyn"), "fpe.except"(metadata !"strict") ]

define float @test_cosh(float %a) strictfp {
  %res = call float @llvm.experimental.constrained.cosh.f32(float %a, metadata !"round.dynamic", metadata !"fpexcept.strict")
  ret float %res
}
; CHECK-LABEL: define float @test_cosh(
; CHECK: call float @llvm.experimental.constrained.cosh.f32(float {{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict") #[[ATTR0]] [ "fpe.control"(metadata !"dyn"), "fpe.except"(metadata !"strict") ]

define float @test_tanh(float %a) strictfp {
  %res = call float @llvm.experimental.constrained.tanh.f32(float %a, metadata !"round.dynamic", metadata !"fpexcept.strict")
  ret float %res
}
; CHECK-LABEL: define float @test_tanh(
; CHECK: call float @llvm.experimental.constrained.tanh.f32(float {{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict") #[[ATTR0]] [ "fpe.control"(metadata !"dyn"), "fpe.except"(metadata !"strict") ]

define float @test_pow(float %a, float %b) strictfp {
  %res = call float @llvm.experimental.constrained.pow.f32(float %a, float %b, metadata !"round.dynamic", metadata !"fpexcept.strict")
  ret float %res
}
; CHECK-LABEL: define float @test_pow(
; CHECK: call float @llvm.experimental.constrained.pow.f32(float {{.*}}, float {{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict") #[[ATTR0]] [ "fpe.control"(metadata !"dyn"), "fpe.except"(metadata !"strict") ]

define float @test_log(float %a) strictfp {
  %res = call float @llvm.experimental.constrained.log.f32(float %a, metadata !"round.dynamic", metadata !"fpexcept.strict")
  ret float %res
}
; CHECK-LABEL: define float @test_log(
; CHECK: call float @llvm.experimental.constrained.log.f32(float {{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict") #[[ATTR0]] [ "fpe.control"(metadata !"dyn"), "fpe.except"(metadata !"strict") ]

define float @test_log10(float %a) strictfp {
  %res = call float @llvm.experimental.constrained.log10.f32(float %a, metadata !"round.dynamic", metadata !"fpexcept.strict")
  ret float %res
}
; CHECK-LABEL: define float @test_log10(
; CHECK: call float @llvm.experimental.constrained.log10.f32(float {{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict") #[[ATTR0]] [ "fpe.control"(metadata !"dyn"), "fpe.except"(metadata !"strict") ]

define float @test_log2(float %a) strictfp {
  %res = call float @llvm.experimental.constrained.log2.f32(float %a, metadata !"round.dynamic", metadata !"fpexcept.strict")
  ret float %res
}
; CHECK-LABEL: define float @test_log2(
; CHECK: call float @llvm.experimental.constrained.log2.f32(float {{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict") #[[ATTR0]] [ "fpe.control"(metadata !"dyn"), "fpe.except"(metadata !"strict") ]

define float @test_exp(float %a) strictfp {
  %res = call float @llvm.experimental.constrained.exp.f32(float %a, metadata !"round.dynamic", metadata !"fpexcept.strict")
  ret float %res
}
; CHECK-LABEL: define float @test_exp(
; CHECK: call float @llvm.experimental.constrained.exp.f32(float {{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict") #[[ATTR0]] [ "fpe.control"(metadata !"dyn"), "fpe.except"(metadata !"strict") ]

define float @test_exp2(float %a) strictfp {
  %res = call float @llvm.experimental.constrained.exp2.f32(float %a, metadata !"round.dynamic", metadata !"fpexcept.strict")
  ret float %res
}
; CHECK-LABEL: define float @test_exp2(
; CHECK: call float @llvm.experimental.constrained.exp2.f32(float {{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict") #[[ATTR0]] [ "fpe.control"(metadata !"dyn"), "fpe.except"(metadata !"strict") ]

define float @test_rint(float %a) strictfp {
  %res = call float @llvm.experimental.constrained.rint.f32(float %a, metadata !"round.dynamic", metadata !"fpexcept.strict")
  ret float %res
}
; CHECK-LABEL: define float @test_rint(
; CHECK: call float @llvm.experimental.constrained.rint.f32(float {{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict") #[[ATTR0]] [ "fpe.control"(metadata !"dyn"), "fpe.except"(metadata !"strict") ]

define float @test_nearbyint(float %a) strictfp {
  %res = call float @llvm.experimental.constrained.nearbyint.f32(float %a, metadata !"round.dynamic", metadata !"fpexcept.strict")
  ret float %res
}
; CHECK-LABEL: define float @test_nearbyint(
; CHECK: call float @llvm.experimental.constrained.nearbyint.f32(float {{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict") #[[ATTR0]] [ "fpe.control"(metadata !"dyn"), "fpe.except"(metadata !"strict") ]

define i32 @test_lrint(float %a) strictfp {
  %res = call i32 @llvm.experimental.constrained.lrint.i32.f32(float %a, metadata !"round.dynamic", metadata !"fpexcept.strict")
  ret i32 %res
}
; CHECK-LABEL: define i32 @test_lrint(
; CHECK: call i32 @llvm.experimental.constrained.lrint.i32.f32(float {{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict") #[[ATTR0]] [ "fpe.control"(metadata !"dyn"), "fpe.except"(metadata !"strict") ]

define i32 @test_llrint(float %a) strictfp {
  %res = call i32 @llvm.experimental.constrained.llrint.i32.f32(float %a, metadata !"round.dynamic", metadata !"fpexcept.strict")
  ret i32 %res
}
; CHECK-LABEL: define i32 @test_llrint(
; CHECK: call i32 @llvm.experimental.constrained.llrint.i32.f32(float {{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict") #[[ATTR0]] [ "fpe.control"(metadata !"dyn"), "fpe.except"(metadata !"strict") ]

define float @test_maxnum(float %a, float %b) strictfp {
  %res = call float @llvm.experimental.constrained.maxnum.f32(float %a, float %b, metadata !"fpexcept.strict")
  ret float %res
}
; CHECK-LABEL: define float @test_maxnum(
; CHECK: call float @llvm.experimental.constrained.maxnum.f32(float {{.*}}, float {{.*}}, metadata !"fpexcept.strict") #[[ATTR0]] [ "fpe.except"(metadata !"strict") ]

define float @test_minnum(float %a, float %b) strictfp {
  %res = call float @llvm.experimental.constrained.minnum.f32(float %a, float %b, metadata !"fpexcept.strict")
  ret float %res
}
; CHECK-LABEL: define float @test_minnum(
; CHECK: call float @llvm.experimental.constrained.minnum.f32(float {{.*}}, float {{.*}}, metadata !"fpexcept.strict") #[[ATTR0]] [ "fpe.except"(metadata !"strict") ]

define float @test_maximum(float %a, float %b) strictfp {
  %res = call float @llvm.experimental.constrained.maximum.f32(float %a, float %b, metadata !"fpexcept.strict")
  ret float %res
}
; CHECK-LABEL: define float @test_maximum(
; CHECK: call float @llvm.experimental.constrained.maximum.f32(float {{.*}}, float {{.*}}, metadata !"fpexcept.strict") #[[ATTR0]] [ "fpe.except"(metadata !"strict") ]

define float @test_minimum(float %a, float %b) strictfp {
  %res = call float @llvm.experimental.constrained.minimum.f32(float %a, float %b, metadata !"fpexcept.strict")
  ret float %res
}
; CHECK-LABEL: define float @test_minimum(
; CHECK: call float @llvm.experimental.constrained.minimum.f32(float {{.*}}, float {{.*}}, metadata !"fpexcept.strict") #[[ATTR0]] [ "fpe.except"(metadata !"strict") ]

define float @test_ceil(float %a) strictfp {
  %res = call float @llvm.experimental.constrained.ceil.f32(float %a, metadata !"fpexcept.strict")
  ret float %res
}
; CHECK-LABEL: define float @test_ceil(
; call float @llvm.experimental.constrained.ceil.f32(float {{.*}}, metadata !"fpexcept.strict") #[[ATTR0]] [ "fpe.except"(metadata !"strict") ]

define float @test_floor(float %a) strictfp {
  %res = call float @llvm.experimental.constrained.floor.f32(float %a, metadata !"fpexcept.strict")
  ret float %res
}
; CHECK-LABEL: define float @test_floor(
; call float @llvm.experimental.constrained.floor.f32(float {{.*}}, metadata !"fpexcept.strict") #[[ATTR0]] [ "fpe.except"(metadata !"strict") ]

define i32 @test_lround(float %a) strictfp {
  %res = call i32 @llvm.experimental.constrained.lround.i32.f32(float %a, metadata !"fpexcept.strict")
  ret i32 %res
}
; CHECK-LABEL: define i32 @test_lround(
; CHECK: call i32 @llvm.experimental.constrained.lround.i32.f32(float {{.*}}, metadata !"fpexcept.strict") #[[ATTR0]] [ "fpe.except"(metadata !"strict") ]

define i32 @test_llround(float %a) strictfp {
  %res = call i32 @llvm.experimental.constrained.llround.i32.f32(float %a, metadata !"fpexcept.strict")
  ret i32 %res
}
; CHECK-LABEL: define i32 @test_llround(
; CHECK: call i32 @llvm.experimental.constrained.llround.i32.f32(float {{.*}}, metadata !"fpexcept.strict") #[[ATTR0]] [ "fpe.except"(metadata !"strict") ]

define float @test_round(float %a) strictfp {
  %res = call float @llvm.experimental.constrained.round.f32(float %a, metadata !"fpexcept.strict")
  ret float %res
}
; CHECK-LABEL: define float @test_round(
; CHECK: call float @llvm.experimental.constrained.round.f32(float {{.*}}, metadata !"fpexcept.strict") #[[ATTR0]] [ "fpe.except"(metadata !"strict") ]

define float @test_roundeven(float %a) strictfp {
  %res = call float @llvm.experimental.constrained.roundeven.f32(float %a, metadata !"fpexcept.strict")
  ret float %res
}
; CHECK-LABEL: define float @test_roundeven(
; CHECK: call float @llvm.experimental.constrained.roundeven.f32(float {{.*}}, metadata !"fpexcept.strict") #[[ATTR0]] [ "fpe.except"(metadata !"strict") ]

define float @test_trunc(float %a) strictfp {
  %res = call float @llvm.experimental.constrained.trunc.f32(float %a, metadata !"fpexcept.strict")
  ret float %res
}
; CHECK-LABEL: define float @test_trunc(
; CHECK: call float @llvm.trunc.f32(float {{.*}}) #[[ATTR0]] [ "fpe.except"(metadata !"strict") ]

; CHECK: attributes #[[ATTR0]] = { strictfp memory(inaccessiblemem: readwrite) }
