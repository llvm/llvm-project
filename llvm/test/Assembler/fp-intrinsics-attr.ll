; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; Test to verify that constrained intrinsics are auto-upgraded on bitcode load.
; With default rounding mode (dynamic) and exception behavior (strict), arithmetic
; ops become plain instructions and math intrinsics become non-constrained calls.
; fcmps (signaling compare) is upgraded to llvm.fcmps with no fp.except bundle
; (strict is the default in a strictfp function).

define void @func(double %a, double %b, double %c, i32 %i) strictfp {
; CHECK-LABEL: define void @func
; CHECK-SAME: (double [[A:%.*]], double [[B:%.*]], double [[C:%.*]], i32 [[I:%.*]]) #[[ATTR0:[0-9]+]] {
; CHECK:      [[ADD:%.*]] = fadd double [[A]], [[B]]
; CHECK-NEXT: [[SUB:%.*]] = fsub double [[A]], [[B]]
; CHECK-NEXT: [[MUL:%.*]] = fmul double [[A]], [[B]]
; CHECK-NEXT: [[DIV:%.*]] = fdiv double [[A]], [[B]]
; CHECK-NEXT: [[REM:%.*]] = frem double [[A]], [[B]]
; CHECK-NEXT: {{.*}} = call double @llvm.fma.f64(double [[A]], double [[B]], double [[C]])
; CHECK-NEXT: {{.*}} = call double @llvm.fmuladd.f64(double [[A]], double [[B]], double [[C]])
; CHECK-NEXT: {{.*}} = fptosi double [[A]] to i32
; CHECK-NEXT: {{.*}} = fptoui double [[A]] to i32
; CHECK-NEXT: {{.*}} = sitofp i32 [[I]] to double
; CHECK-NEXT: {{.*}} = uitofp i32 [[I]] to double
; CHECK-NEXT: [[FPTRUNC:%.*]] = fptrunc double [[A]] to float
; CHECK-NEXT: {{.*}} = fpext float [[FPTRUNC]] to double
; CHECK-NEXT: {{.*}} = call double @llvm.sqrt.f64(double [[A]])
; CHECK-NEXT: {{.*}} = call double @llvm.powi.f64.i32(double [[A]], i32 [[I]])
; CHECK-NEXT: {{.*}} = call double @llvm.sin.f64(double [[A]])
; CHECK-NEXT: {{.*}} = call double @llvm.cos.f64(double [[A]])
; CHECK-NEXT: {{.*}} = call double @llvm.tan.f64(double [[A]])
; CHECK-NEXT: {{.*}} = call double @llvm.acos.f64(double [[A]])
; CHECK-NEXT: {{.*}} = call double @llvm.asin.f64(double [[A]])
; CHECK-NEXT: {{.*}} = call double @llvm.atan.f64(double [[A]])
; CHECK-NEXT: {{.*}} = call double @llvm.atan2.f64(double [[A]], double [[B]])
; CHECK-NEXT: {{.*}} = call double @llvm.cosh.f64(double [[A]])
; CHECK-NEXT: {{.*}} = call double @llvm.sinh.f64(double [[A]])
; CHECK-NEXT: {{.*}} = call double @llvm.tanh.f64(double [[A]])
; CHECK-NEXT: {{.*}} = call double @llvm.pow.f64(double [[A]], double [[B]])
; CHECK-NEXT: {{.*}} = call double @llvm.log.f64(double [[A]])
; CHECK-NEXT: {{.*}} = call double @llvm.log10.f64(double [[A]])
; CHECK-NEXT: {{.*}} = call double @llvm.log2.f64(double [[A]])
; CHECK-NEXT: {{.*}} = call double @llvm.exp.f64(double [[A]])
; CHECK-NEXT: {{.*}} = call double @llvm.exp2.f64(double [[A]])
; CHECK-NEXT: {{.*}} = call double @llvm.rint.f64(double [[A]])
; CHECK-NEXT: {{.*}} = call double @llvm.nearbyint.f64(double [[A]])
; CHECK-NEXT: {{.*}} = call i32 @llvm.lrint.i32.f64(double [[A]])
; CHECK-NEXT: {{.*}} = call i64 @llvm.llrint.i64.f64(double [[A]])
; CHECK-NEXT: {{.*}} = call double @llvm.maxnum.f64(double [[A]], double [[B]])
; CHECK-NEXT: {{.*}} = call double @llvm.minnum.f64(double [[A]], double [[B]])
; CHECK-NEXT: {{.*}} = call double @llvm.maximum.f64(double [[A]], double [[B]])
; CHECK-NEXT: {{.*}} = call double @llvm.minimum.f64(double [[A]], double [[B]])
; CHECK-NEXT: {{.*}} = call double @llvm.ceil.f64(double [[A]])
; CHECK-NEXT: {{.*}} = call double @llvm.floor.f64(double [[A]])
; CHECK-NEXT: {{.*}} = call i32 @llvm.lround.i32.f64(double [[A]])
; CHECK-NEXT: {{.*}} = call i64 @llvm.llround.i64.f64(double [[A]])
; CHECK-NEXT: {{.*}} = call double @llvm.round.f64(double [[A]])
; CHECK-NEXT: {{.*}} = call double @llvm.roundeven.f64(double [[A]])
; CHECK-NEXT: {{.*}} = call double @llvm.trunc.f64(double [[A]])
; CHECK-NEXT: {{.*}} = fcmp oeq double [[A]], [[B]]
; CHECK-NEXT: {{.*}} = call i1 @llvm.fcmps.f64(double [[A]], double [[B]], metadata !"oeq")
; CHECK-NEXT: ret void

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

  ret void
}

; fcmps is auto-upgraded to the new llvm.fcmps intrinsic (3 args: float, float, metadata pred).
; Plain intrinsic declarations are emitted for the upgraded calls.
; CHECK-DAG: declare i1 @llvm.fcmps.f64(double, double, metadata) #[[ATTR1:[0-9]+]]
; CHECK-DAG: declare double @llvm.fma.f64(double, double, double)
; CHECK-DAG: declare double @llvm.fmuladd.f64(double, double, double)
; CHECK-DAG: declare double @llvm.sqrt.f64(double)
; CHECK-DAG: declare double @llvm.powi.f64.i32(double, i32)
; CHECK-DAG: declare double @llvm.sin.f64(double)
; CHECK-DAG: declare double @llvm.cos.f64(double)
; CHECK-DAG: declare double @llvm.tan.f64(double)
; CHECK-DAG: declare double @llvm.asin.f64(double)
; CHECK-DAG: declare double @llvm.acos.f64(double)
; CHECK-DAG: declare double @llvm.atan.f64(double)
; CHECK-DAG: declare double @llvm.atan2.f64(double, double)
; CHECK-DAG: declare double @llvm.sinh.f64(double)
; CHECK-DAG: declare double @llvm.cosh.f64(double)
; CHECK-DAG: declare double @llvm.tanh.f64(double)
; CHECK-DAG: declare double @llvm.pow.f64(double, double)
; CHECK-DAG: declare double @llvm.log.f64(double)
; CHECK-DAG: declare double @llvm.log10.f64(double)
; CHECK-DAG: declare double @llvm.log2.f64(double)
; CHECK-DAG: declare double @llvm.exp.f64(double)
; CHECK-DAG: declare double @llvm.exp2.f64(double)
; CHECK-DAG: declare double @llvm.rint.f64(double)
; CHECK-DAG: declare double @llvm.nearbyint.f64(double)
; CHECK-DAG: declare i32 @llvm.lrint.i32.f64(double)
; CHECK-DAG: declare i64 @llvm.llrint.i64.f64(double)
; CHECK-DAG: declare double @llvm.maxnum.f64(double, double)
; CHECK-DAG: declare double @llvm.minnum.f64(double, double)
; CHECK-DAG: declare double @llvm.maximum.f64(double, double)
; CHECK-DAG: declare double @llvm.minimum.f64(double, double)
; CHECK-DAG: declare double @llvm.ceil.f64(double)
; CHECK-DAG: declare double @llvm.floor.f64(double)
; CHECK-DAG: declare i32 @llvm.lround.i32.f64(double)
; CHECK-DAG: declare i64 @llvm.llround.i64.f64(double)
; CHECK-DAG: declare double @llvm.round.f64(double)
; CHECK-DAG: declare double @llvm.roundeven.f64(double)
; CHECK-DAG: declare double @llvm.trunc.f64(double)

declare double @llvm.experimental.constrained.fadd.f64(double, double, metadata, metadata)
declare double @llvm.experimental.constrained.fsub.f64(double, double, metadata, metadata)
declare double @llvm.experimental.constrained.fmul.f64(double, double, metadata, metadata)
declare double @llvm.experimental.constrained.fdiv.f64(double, double, metadata, metadata)
declare double @llvm.experimental.constrained.frem.f64(double, double, metadata, metadata)
declare double @llvm.experimental.constrained.fma.f64(double, double, double, metadata, metadata)
declare double @llvm.experimental.constrained.fmuladd.f64(double, double, double, metadata, metadata)
declare i32 @llvm.experimental.constrained.fptosi.i32.f64(double, metadata)
declare i32 @llvm.experimental.constrained.fptoui.i32.f64(double, metadata)
declare double @llvm.experimental.constrained.sitofp.f64.i32(i32, metadata, metadata)
declare double @llvm.experimental.constrained.uitofp.f64.i32(i32, metadata, metadata)
declare float @llvm.experimental.constrained.fptrunc.f32.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.fpext.f64.f32(float, metadata)
declare double @llvm.experimental.constrained.sqrt.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.powi.f64(double, i32, metadata, metadata)
declare double @llvm.experimental.constrained.sin.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.cos.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.tan.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.asin.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.acos.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.atan.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.atan2.f64(double, double, metadata, metadata)
declare double @llvm.experimental.constrained.sinh.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.cosh.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.tanh.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.pow.f64(double, double, metadata, metadata)
declare double @llvm.experimental.constrained.log.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.log10.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.log2.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.exp.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.exp2.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.rint.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.nearbyint.f64(double, metadata, metadata)
declare i32 @llvm.experimental.constrained.lrint.i32.f64(double, metadata, metadata)
declare i64 @llvm.experimental.constrained.llrint.i64.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.maxnum.f64(double, double, metadata)
declare double @llvm.experimental.constrained.minnum.f64(double, double, metadata)
declare double @llvm.experimental.constrained.maximum.f64(double, double, metadata)
declare double @llvm.experimental.constrained.minimum.f64(double, double, metadata)
declare double @llvm.experimental.constrained.ceil.f64(double, metadata)
declare double @llvm.experimental.constrained.floor.f64(double, metadata)
declare i32 @llvm.experimental.constrained.lround.i32.f64(double, metadata)
declare i64 @llvm.experimental.constrained.llround.i64.f64(double, metadata)
declare double @llvm.experimental.constrained.round.f64(double, metadata)
declare double @llvm.experimental.constrained.roundeven.f64(double, metadata)
declare double @llvm.experimental.constrained.trunc.f64(double, metadata)
declare i1 @llvm.experimental.constrained.fcmp.f64(double, double, metadata, metadata)
declare i1 @llvm.experimental.constrained.fcmps.f64(double, double, metadata, metadata)

; The function retains its strictfp attribute after upgrade.
; CHECK: attributes #[[ATTR0]] = {{{.*}} strictfp {{.*}}}
; The fcmps declaration gets willreturn + inaccessiblemem attributes.
; CHECK: attributes #[[ATTR1]] = { nocreateundeforpoison nounwind willreturn memory(inaccessiblemem: readwrite) }
