; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; Test to verify that constrained intrinsics all have the strictfp attribute.
; Ordering is from Intrinsics.td.

define void @func(double %a, double %b, double %c, i32 %i) strictfp {
; CHECK-LABEL: define void @func
; CHECK-SAME: (double [[A:%.*]], double [[B:%.*]], double [[C:%.*]], i32 [[I:%.*]]) #[[ATTR0:[0-9]+]] {

  %add = call double @llvm.experimental.constrained.fadd.f64(
                                               double %a, double %b,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")

  %sub = call double @llvm.experimental.constrained.fsub.f64(
                                               double %a, double %b,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")

  %mul = call double @llvm.experimental.constrained.fmul.f64(
                                               double %a, double %b,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")

  %div = call double @llvm.experimental.constrained.fdiv.f64(
                                               double %a, double %b,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")

  %rem = call double @llvm.experimental.constrained.frem.f64(
                                               double %a, double %b,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")

  %fma = call double @llvm.experimental.constrained.fma.f64(
                                               double %a, double %b, double %c,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")

  %fmuladd = call double @llvm.experimental.constrained.fmuladd.f64(
                                               double %a, double %b, double %c,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")

  %si = call i32 @llvm.experimental.constrained.fptosi.i32.f64(double %a,
                                               metadata !"fpexcept.strict")

  %ui = call i32 @llvm.experimental.constrained.fptoui.i32.f64(double %a,
                                               metadata !"fpexcept.strict")

  %sfp = call double @llvm.experimental.constrained.sitofp.f64.i32(i32 %i,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")

  %ufp = call double @llvm.experimental.constrained.uitofp.f64.i32(i32 %i,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")

  %fptrunc = call float @llvm.experimental.constrained.fptrunc.f32.f64(
                                               double %a,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")

  %ext = call double @llvm.experimental.constrained.fpext.f64.f32(
                                               float %fptrunc,
                                               metadata !"fpexcept.strict")

  %sqrt = call double @llvm.experimental.constrained.sqrt.f64(
                                               double %a,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")

  %powi = call double @llvm.experimental.constrained.powi.f64(
                                               double %a, i32 %i,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")

  %sin = call double @llvm.experimental.constrained.sin.f64(
                                               double %a,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")

  %cos = call double @llvm.experimental.constrained.cos.f64(
                                               double %a,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")

  %tan = call double @llvm.experimental.constrained.tan.f64(
                                               double %a,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")

  %acos = call double @llvm.experimental.constrained.acos.f64(
                                               double %a,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")

  %asin = call double @llvm.experimental.constrained.asin.f64(
                                               double %a,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")

  %atan = call double @llvm.experimental.constrained.atan.f64(
                                               double %a,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")

  %atan2 = call double @llvm.experimental.constrained.atan2.f64(
                                               double %a, double %b,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")

  %cosh = call double @llvm.experimental.constrained.cosh.f64(
                                               double %a,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")

  %sinh = call double @llvm.experimental.constrained.sinh.f64(
                                               double %a,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")

  %tanh = call double @llvm.experimental.constrained.tanh.f64(
                                               double %a,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")

  %pow = call double @llvm.experimental.constrained.pow.f64(
                                               double %a, double %b,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")

  %log = call double @llvm.experimental.constrained.log.f64(
                                               double %a,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")

  %log10 = call double @llvm.experimental.constrained.log10.f64(
                                               double %a,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")

  %log2 = call double @llvm.experimental.constrained.log2.f64(
                                               double %a,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")

  %exp = call double @llvm.experimental.constrained.exp.f64(
                                               double %a,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")

  %exp2 = call double @llvm.experimental.constrained.exp2.f64(
                                               double %a,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")

  %rint = call double @llvm.experimental.constrained.rint.f64(
                                               double %a,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")

  %neari = call double @llvm.experimental.constrained.nearbyint.f64(
                                               double %a,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")

  %x32 = call i32 @llvm.experimental.constrained.lrint.i32.f64(
                                               double %a,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")

  %x64 = call i64 @llvm.experimental.constrained.llrint.i64.f64(
                                               double %a,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")

  %maxnum = call double @llvm.experimental.constrained.maxnum.f64(
                                               double %a, double %b,
                                               metadata !"fpexcept.strict")

  %minnum = call double @llvm.experimental.constrained.minnum.f64(
                                               double %a, double %b,
                                               metadata !"fpexcept.strict")

  %maxmum = call double @llvm.experimental.constrained.maximum.f64(
                                               double %a, double %b,
                                               metadata !"fpexcept.strict")

  %minmum = call double @llvm.experimental.constrained.minimum.f64(
                                               double %a, double %b,
                                               metadata !"fpexcept.strict")

  %ceil = call double @llvm.experimental.constrained.ceil.f64(
                                               double %a,
                                               metadata !"fpexcept.strict")

  %floor = call double @llvm.experimental.constrained.floor.f64(
                                               double %a,
                                               metadata !"fpexcept.strict")

  %y32 = call i32 @llvm.experimental.constrained.lround.i32.f64(
                                               double %a,
                                               metadata !"fpexcept.strict")

  %y64 = call i64 @llvm.experimental.constrained.llround.i64.f64(
                                               double %a,
                                               metadata !"fpexcept.strict")

  %round = call double @llvm.experimental.constrained.round.f64(
                                               double %a,
                                               metadata !"fpexcept.strict")

  %roundev = call double @llvm.experimental.constrained.roundeven.f64(
                                               double %a,
                                               metadata !"fpexcept.strict")

  %trunc = call double @llvm.experimental.constrained.trunc.f64(
                                               double %a,
                                               metadata !"fpexcept.strict")

  %q1 = call i1 @llvm.experimental.constrained.fcmp.f64(
                                               double %a, double %b,
                                               metadata !"oeq",
                                               metadata !"fpexcept.strict")

  %s1 = call i1 @llvm.experimental.constrained.fcmps.f64(
                                               double %a, double %b,
                                               metadata !"oeq",
                                               metadata !"fpexcept.strict")

; CHECK: ret void
  ret void
}

declare double @llvm.experimental.constrained.fadd.f64(double, double, metadata, metadata)
; CHECK: @llvm.experimental.constrained.fadd.f64({{.*}}) #[[ATTR1:[0-9]+]]

declare double @llvm.experimental.constrained.fsub.f64(double, double, metadata, metadata)
; CHECK: @llvm.experimental.constrained.fsub.f64({{.*}}) #[[ATTR1]]

declare double @llvm.experimental.constrained.fmul.f64(double, double, metadata, metadata)
; CHECK: @llvm.experimental.constrained.fmul.f64({{.*}}) #[[ATTR1]]

declare double @llvm.experimental.constrained.fdiv.f64(double, double, metadata, metadata)
; CHECK: @llvm.experimental.constrained.fdiv.f64({{.*}}) #[[ATTR1]]

declare double @llvm.experimental.constrained.frem.f64(double, double, metadata, metadata)
; CHECK: @llvm.experimental.constrained.frem.f64({{.*}}) #[[ATTR1]]

declare double @llvm.experimental.constrained.fma.f64(double, double, double, metadata, metadata)
; CHECK: @llvm.experimental.constrained.fma.f64({{.*}}) #[[ATTR1]]

declare double @llvm.experimental.constrained.fmuladd.f64(double, double, double, metadata, metadata)
; CHECK: @llvm.experimental.constrained.fmuladd.f64({{.*}}) #[[ATTR1]]

declare i32 @llvm.experimental.constrained.fptosi.i32.f64(double, metadata)
; CHECK: @llvm.experimental.constrained.fptosi.i32.f64({{.*}}) #[[ATTR1]]

declare i32 @llvm.experimental.constrained.fptoui.i32.f64(double, metadata)
; CHECK: @llvm.experimental.constrained.fptoui.i32.f64({{.*}}) #[[ATTR1]]

declare double @llvm.experimental.constrained.sitofp.f64.i32(i32, metadata, metadata)
; CHECK: @llvm.experimental.constrained.sitofp.f64.i32({{.*}}) #[[ATTR1]]

declare double @llvm.experimental.constrained.uitofp.f64.i32(i32, metadata, metadata)
; CHECK: @llvm.experimental.constrained.uitofp.f64.i32({{.*}}) #[[ATTR1]]

declare float @llvm.experimental.constrained.fptrunc.f32.f64(double, metadata, metadata)
; CHECK: @llvm.experimental.constrained.fptrunc.f32.f64({{.*}}) #[[ATTR1]]

declare double @llvm.experimental.constrained.fpext.f64.f32(float, metadata)
; CHECK: @llvm.experimental.constrained.fpext.f64.f32({{.*}}) #[[ATTR1]]

declare double @llvm.experimental.constrained.sqrt.f64(double, metadata, metadata)
; CHECK: @llvm.experimental.constrained.sqrt.f64({{.*}}) #[[ATTR1]]

declare double @llvm.experimental.constrained.powi.f64(double, i32, metadata, metadata)
; CHECK: @llvm.experimental.constrained.powi.f64({{.*}}) #[[ATTR1]]

declare double @llvm.experimental.constrained.sin.f64(double, metadata, metadata)
; CHECK: @llvm.experimental.constrained.sin.f64({{.*}}) #[[ATTR1]]

declare double @llvm.experimental.constrained.cos.f64(double, metadata, metadata)
; CHECK: @llvm.experimental.constrained.cos.f64({{.*}}) #[[ATTR1]]

declare double @llvm.experimental.constrained.tan.f64(double, metadata, metadata)
; CHECK: @llvm.experimental.constrained.tan.f64({{.*}}) #[[ATTR1]]

declare double @llvm.experimental.constrained.asin.f64(double, metadata, metadata)
; CHECK: @llvm.experimental.constrained.asin.f64({{.*}}) #[[ATTR1]]

declare double @llvm.experimental.constrained.acos.f64(double, metadata, metadata)
; CHECK: @llvm.experimental.constrained.acos.f64({{.*}}) #[[ATTR1]]

declare double @llvm.experimental.constrained.atan.f64(double, metadata, metadata)
; CHECK: @llvm.experimental.constrained.atan.f64({{.*}}) #[[ATTR1]]

declare double @llvm.experimental.constrained.atan2.f64(double, double, metadata, metadata)
; CHECK: @llvm.experimental.constrained.atan2.f64({{.*}}) #[[ATTR1]]

declare double @llvm.experimental.constrained.sinh.f64(double, metadata, metadata)
; CHECK: @llvm.experimental.constrained.sinh.f64({{.*}}) #[[ATTR1]]

declare double @llvm.experimental.constrained.cosh.f64(double, metadata, metadata)
; CHECK: @llvm.experimental.constrained.cosh.f64({{.*}}) #[[ATTR1]]

declare double @llvm.experimental.constrained.tanh.f64(double, metadata, metadata)
; CHECK: @llvm.experimental.constrained.tanh.f64({{.*}}) #[[ATTR1]]

declare double @llvm.experimental.constrained.pow.f64(double, double, metadata, metadata)
; CHECK: @llvm.experimental.constrained.pow.f64({{.*}}) #[[ATTR1]]

declare double @llvm.experimental.constrained.log.f64(double, metadata, metadata)
; CHECK: @llvm.experimental.constrained.log.f64({{.*}}) #[[ATTR1]]

declare double @llvm.experimental.constrained.log10.f64(double, metadata, metadata)
; CHECK: @llvm.experimental.constrained.log10.f64({{.*}}) #[[ATTR1]]

declare double @llvm.experimental.constrained.log2.f64(double, metadata, metadata)
; CHECK: @llvm.experimental.constrained.log2.f64({{.*}}) #[[ATTR1]]

declare double @llvm.experimental.constrained.exp.f64(double, metadata, metadata)
; CHECK: @llvm.experimental.constrained.exp.f64({{.*}}) #[[ATTR1]]

declare double @llvm.experimental.constrained.exp2.f64(double, metadata, metadata)
; CHECK: @llvm.experimental.constrained.exp2.f64({{.*}}) #[[ATTR1]]

declare double @llvm.experimental.constrained.rint.f64(double, metadata, metadata)
; CHECK: @llvm.experimental.constrained.rint.f64({{.*}}) #[[ATTR1]]

declare double @llvm.experimental.constrained.nearbyint.f64(double, metadata, metadata)
; CHECK: @llvm.experimental.constrained.nearbyint.f64({{.*}}) #[[ATTR1]]

declare i32 @llvm.experimental.constrained.lrint.i32.f64(double, metadata, metadata)
; CHECK: @llvm.experimental.constrained.lrint.i32.f64({{.*}}) #[[ATTR1]]

declare i64 @llvm.experimental.constrained.llrint.i64.f64(double, metadata, metadata)
; CHECK: @llvm.experimental.constrained.llrint.i64.f64({{.*}}) #[[ATTR1]]

declare double @llvm.experimental.constrained.maxnum.f64(double, double, metadata)
; CHECK: @llvm.experimental.constrained.maxnum.f64({{.*}}) #[[ATTR1]]

declare double @llvm.experimental.constrained.minnum.f64(double, double, metadata)
; CHECK: @llvm.experimental.constrained.minnum.f64({{.*}}) #[[ATTR1]]

declare double @llvm.experimental.constrained.maximum.f64(double, double, metadata)
; CHECK: @llvm.experimental.constrained.maximum.f64({{.*}}) #[[ATTR1]]

declare double @llvm.experimental.constrained.minimum.f64(double, double, metadata)
; CHECK: @llvm.experimental.constrained.minimum.f64({{.*}}) #[[ATTR1]]

declare double @llvm.experimental.constrained.ceil.f64(double, metadata)
; CHECK: @llvm.experimental.constrained.ceil.f64({{.*}}) #[[ATTR1]]

declare double @llvm.experimental.constrained.floor.f64(double, metadata)
; CHECK: @llvm.experimental.constrained.floor.f64({{.*}}) #[[ATTR1]]

declare i32 @llvm.experimental.constrained.lround.i32.f64(double, metadata)
; CHECK: @llvm.experimental.constrained.lround.i32.f64({{.*}}) #[[ATTR1]]

declare i64 @llvm.experimental.constrained.llround.i64.f64(double, metadata)
; CHECK: @llvm.experimental.constrained.llround.i64.f64({{.*}}) #[[ATTR1]]

declare double @llvm.experimental.constrained.round.f64(double, metadata)
; CHECK: @llvm.experimental.constrained.round.f64({{.*}}) #[[ATTR1]]

declare double @llvm.experimental.constrained.roundeven.f64(double, metadata)
; CHECK: @llvm.experimental.constrained.roundeven.f64({{.*}}) #[[ATTR1]]

declare double @llvm.experimental.constrained.trunc.f64(double, metadata)
; CHECK: @llvm.experimental.constrained.trunc.f64({{.*}}) #[[ATTR1]]

declare i1 @llvm.experimental.constrained.fcmp.f64(double, double, metadata, metadata)
; CHECK: @llvm.experimental.constrained.fcmp.f64({{.*}}) #[[ATTR1]]

declare i1 @llvm.experimental.constrained.fcmps.f64(double, double, metadata, metadata)
; CHECK: @llvm.experimental.constrained.fcmps.f64({{.*}}) #[[ATTR1]]

; CHECK: attributes #[[ATTR0]] = {{{.*}} strictfp {{.*}}}
; CHECK: attributes #[[ATTR1]] = { {{.*}} strictfp {{.*}} }

