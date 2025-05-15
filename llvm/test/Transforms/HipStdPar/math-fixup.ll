; RUN: opt -S -passes=hipstdpar-math-fixup %s | FileCheck %s

define void @foo(double noundef %dbl, float noundef %flt, i32 noundef %quo) #0 {
; CHECK-LABEL: define void @foo(
; CHECK-SAME: double noundef [[DBL:%.*]], float noundef [[FLT:%.*]], i32 noundef [[QUO:%.*]]) #[[ATTR0:[0-9]+]] {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[QUO_ADDR:%.*]] = alloca i32, align 4
; CHECK-NEXT:    store i32 [[QUO]], ptr [[QUO_ADDR]], align 4
entry:
  %quo.addr = alloca i32, align 4
  store i32 %quo, ptr %quo.addr, align 4
  ; CHECK-NEXT:    [[TMP0:%.*]] = tail call contract double @llvm.fabs.f64(double [[DBL]])
  %0 = tail call contract double @llvm.fabs.f64(double %dbl)
  ; CHECK-NEXT:    [[TMP1:%.*]] = tail call contract float @llvm.fabs.f32(float [[FLT]])
  %1 = tail call contract float @llvm.fabs.f32(float %flt)
  ; CHECK-NEXT:    [[CALL:%.*]] = tail call contract double @__hipstdpar_remainder_f64(double noundef [[TMP0]], double noundef [[TMP0]]) #[[ATTR4:[0-9]+]]
  %call = tail call contract double @remainder(double noundef %0, double noundef %0) #4
  ; CHECK-NEXT:    [[CALL1:%.*]] = tail call contract float @__hipstdpar_remainder_f32(float noundef [[TMP1]], float noundef [[TMP1]]) #[[ATTR4]]
  %call1 = tail call contract float @remainderf(float noundef %1, float noundef %1) #4
  ; CHECK-NEXT:    [[CALL2:%.*]] = call contract double @__hipstdpar_remquo_f64(double noundef [[CALL]], double noundef [[CALL]], ptr noundef nonnull [[QUO_ADDR]]) #[[ATTR3:[0-9]+]]
  %call2 = call contract double @remquo(double noundef %call, double noundef %call, ptr noundef nonnull %quo.addr) #5
  ; CHECK-NEXT:    [[CALL3:%.*]] = call contract float @__hipstdpar_remquo_f32(float noundef [[CALL1]], float noundef [[CALL1]], ptr noundef nonnull [[QUO_ADDR]]) #[[ATTR3]]
  %call3 = call contract float @remquof(float noundef %call1, float noundef %call1, ptr noundef nonnull %quo.addr) #5
  ; CHECK-NEXT:    [[TMP2:%.*]] = call contract double @llvm.fma.f64(double [[CALL2]], double [[CALL2]], double [[CALL2]])
  %2 = call contract double @llvm.fma.f64(double %call2, double %call2, double %call2)
  ; CHECK-NEXT:    [[TMP3:%.*]] = call contract float @llvm.fma.f32(float [[CALL3]], float [[CALL3]], float [[CALL3]])
  %3 = call contract float @llvm.fma.f32(float %call3, float %call3, float %call3)
  ; CHECK-NEXT:    [[CALL4:%.*]] = call contract double @__hipstdpar_fdim_f64(double noundef [[TMP2]], double noundef [[TMP2]]) #[[ATTR4]]
  %call4 = call contract double @fdim(double noundef %2, double noundef %2) #4
  ; CHECK-NEXT:    [[CALL5:%.*]] = call contract float @__hipstdpar_fdim_f32(float noundef [[TMP3]], float noundef [[TMP3]]) #[[ATTR4]]
  %call5 = call contract float @fdimf(float noundef %3, float noundef %3) #4
  ; CHECK-NEXT:    [[TMP4:%.*]] = call contract double @__hipstdpar_exp_f64(double [[CALL4]])
  %4 = call contract double @llvm.exp.f64(double %call4)
  ; CHECK-NEXT:    [[TMP5:%.*]] = call contract float @llvm.exp.f32(float [[CALL5]])
  %5 = call contract float @llvm.exp.f32(float %call5)
  ; CHECK-NEXT:    [[TMP6:%.*]] = call contract double @__hipstdpar_exp2_f64(double [[TMP4]])
  %6 = call contract double @llvm.exp2.f64(double %4)
  ; CHECK-NEXT:    [[TMP7:%.*]] = call contract float @llvm.exp2.f32(float [[TMP5]])
  %7 = call contract float @llvm.exp2.f32(float %5)
  ; CHECK-NEXT:    [[CALL6:%.*]] = call contract double @__hipstdpar_expm1_f64(double noundef [[TMP6]]) #[[ATTR4]]
  %call6 = call contract double @expm1(double noundef %6) #4
  ; CHECK-NEXT:    [[CALL7:%.*]] = call contract float @__hipstdpar_expm1_f32(float noundef [[TMP7]]) #[[ATTR4]]
  %call7 = call contract float @expm1f(float noundef %7) #4
  ; CHECK-NEXT:    [[TMP8:%.*]] = call contract double @__hipstdpar_log_f64(double [[CALL6]])
  %8 = call contract double @llvm.log.f64(double %call6)
  ; CHECK-NEXT:    [[TMP9:%.*]] = call contract float @llvm.log.f32(float [[CALL7]])
  %9 = call contract float @llvm.log.f32(float %call7)
  ; CHECK-NEXT:    [[TMP10:%.*]] = call contract double @__hipstdpar_log10_f64(double [[TMP8]])
  %10 = call contract double @llvm.log10.f64(double %8)
  ; CHECK-NEXT:    [[TMP11:%.*]] = call contract float @llvm.log10.f32(float [[TMP9]])
  %11 = call contract float @llvm.log10.f32(float %9)
  ; CHECK-NEXT:    [[TMP12:%.*]] = call contract double @__hipstdpar_log2_f64(double [[TMP10]])
  %12 = call contract double @llvm.log2.f64(double %10)
  ; CHECK-NEXT:    [[TMP13:%.*]] = call contract float @llvm.log2.f32(float [[TMP11]])
  %13 = call contract float @llvm.log2.f32(float %11)
  ; CHECK-NEXT:    [[CALL8:%.*]] = call contract double @__hipstdpar_log1p_f64(double noundef [[TMP12]]) #[[ATTR4]]
  %call8 = call contract double @log1p(double noundef %12) #4
  ; CHECK-NEXT:    [[CALL9:%.*]] = call contract float @__hipstdpar_log1p_f32(float noundef [[TMP13]]) #[[ATTR4]]
  %call9 = call contract float @log1pf(float noundef %13) #4
  ; CHECK-NEXT:    [[TMP14:%.*]] = call contract float @llvm.pow.f32(float [[CALL9]], float [[CALL9]])
  %14 = call contract float @llvm.pow.f32(float %call9, float %call9)
  ; CHECK-NEXT:    [[TMP15:%.*]] = call contract double @llvm.sqrt.f64(double [[CALL8]])
  %15 = call contract double @llvm.sqrt.f64(double %call8)
  ; CHECK-NEXT:    [[TMP16:%.*]] = call contract float @llvm.sqrt.f32(float [[TMP14]])
  %16 = call contract float @llvm.sqrt.f32(float %14)
  ; CHECK-NEXT:    [[CALL10:%.*]] = call contract double @__hipstdpar_cbrt_f64(double noundef [[TMP15]]) #[[ATTR4]]
  %call10 = call contract double @cbrt(double noundef %15) #4
  ; CHECK-NEXT:    [[CALL11:%.*]] = call contract float @__hipstdpar_cbrt_f32(float noundef [[TMP16]]) #[[ATTR4]]
  %call11 = call contract float @cbrtf(float noundef %16) #4
  ; CHECK-NEXT:    [[CALL12:%.*]] = call contract double @__hipstdpar_hypot_f64(double noundef [[CALL10]], double noundef [[CALL10]]) #[[ATTR4]]
  %call12 = call contract double @hypot(double noundef %call10, double noundef %call10) #4
  ; CHECK-NEXT:    [[CALL13:%.*]] = call contract float @__hipstdpar_hypot_f32(float noundef [[CALL11]], float noundef [[CALL11]]) #[[ATTR4]]
  %call13 = call contract float @hypotf(float noundef %call11, float noundef %call11) #4
  ; CHECK-NEXT:    [[TMP17:%.*]] = call contract float @llvm.sin.f32(float [[CALL13]])
  %17 = call contract float @llvm.sin.f32(float %call13)
  ; CHECK-NEXT:    [[TMP18:%.*]] = call contract float @llvm.cos.f32(float [[TMP17]])
  %18 = call contract float @llvm.cos.f32(float %17)
  ; CHECK-NEXT:    [[TMP19:%.*]] = call contract double @__hipstdpar_tan_f64(double [[CALL12]])
  %19 = call contract double @llvm.tan.f64(double %call12)
  ; CHECK-NEXT:    [[TMP20:%.*]] = call contract double @__hipstdpar_asin_f64(double [[TMP19]])
  %20 = call contract double @llvm.asin.f64(double %19)
  ; CHECK-NEXT:    [[TMP21:%.*]] = call contract double @__hipstdpar_acos_f64(double [[TMP20]])
  %21 = call contract double @llvm.acos.f64(double %20)
  ; CHECK-NEXT:    [[TMP22:%.*]] = call contract double @__hipstdpar_atan_f64(double [[TMP21]])
  %22 = call contract double @llvm.atan.f64(double %21)
  ; CHECK-NEXT:    [[TMP23:%.*]] = call contract double @__hipstdpar_atan2_f64(double [[TMP22]], double [[TMP22]])
  %23 = call contract double @llvm.atan2.f64(double %22, double %22)
  ; CHECK-NEXT:    [[TMP24:%.*]] = call contract double @__hipstdpar_sinh_f64(double [[TMP23]])
  %24 = call contract double @llvm.sinh.f64(double %23)
  ; CHECK-NEXT:    [[TMP25:%.*]] = call contract double @__hipstdpar_cosh_f64(double [[TMP24]])
  %25 = call contract double @llvm.cosh.f64(double %24)
  ; CHECK-NEXT:    [[TMP26:%.*]] = call contract double @__hipstdpar_tanh_f64(double [[TMP25]])
  %26 = call contract double @llvm.tanh.f64(double %25)
  ; CHECK-NEXT:    [[CALL14:%.*]] = call contract double @__hipstdpar_asinh_f64(double noundef [[TMP26]]) #[[ATTR4]]
  %call14 = call contract double @asinh(double noundef %26) #4
  ; CHECK-NEXT:    [[CALL15:%.*]] = call contract float @__hipstdpar_asinh_f32(float noundef [[TMP18]]) #[[ATTR4]]
  %call15 = call contract float @asinhf(float noundef %18) #4
  ; CHECK-NEXT:    [[CALL16:%.*]] = call contract double @__hipstdpar_acosh_f64(double noundef [[CALL14]]) #[[ATTR4]]
  %call16 = call contract double @acosh(double noundef %call14) #4
  ; CHECK-NEXT:    [[CALL17:%.*]] = call contract float @__hipstdpar_acosh_f32(float noundef [[CALL15]]) #[[ATTR4]]
  %call17 = call contract float @acoshf(float noundef %call15) #4
  ; CHECK-NEXT:    [[CALL18:%.*]] = call contract double @__hipstdpar_atanh_f64(double noundef [[CALL16]]) #[[ATTR4]]
  %call18 = call contract double @atanh(double noundef %call16) #4
  ; CHECK-NEXT:    [[CALL19:%.*]] = call contract float @__hipstdpar_atanh_f32(float noundef [[CALL17]]) #[[ATTR4]]
  %call19 = call contract float @atanhf(float noundef %call17) #4
  ; CHECK-NEXT:    [[CALL20:%.*]] = call contract double @__hipstdpar_erf_f64(double noundef [[CALL18]]) #[[ATTR4]]
  %call20 = call contract double @erf(double noundef %call18) #4
  ; CHECK-NEXT:    [[CALL21:%.*]] = call contract float @__hipstdpar_erf_f32(float noundef [[CALL19]]) #[[ATTR4]]
  %call21 = call contract float @erff(float noundef %call19) #4
  ; CHECK-NEXT:    [[CALL22:%.*]] = call contract double @__hipstdpar_erfc_f64(double noundef [[CALL20]]) #[[ATTR4]]
  %call22 = call contract double @erfc(double noundef %call20) #4
  ; CHECK-NEXT:    [[CALL23:%.*]] = call contract float @__hipstdpar_erfc_f32(float noundef [[CALL21]]) #[[ATTR4]]
  %call23 = call contract float @erfcf(float noundef %call21) #4
  ; CHECK-NEXT:    [[CALL24:%.*]] = call contract double @__hipstdpar_tgamma_f64(double noundef [[CALL22]]) #[[ATTR4]]
  %call24 = call contract double @tgamma(double noundef %call22) #4
  ; CHECK-NEXT:    [[CALL25:%.*]] = call contract float @__hipstdpar_tgamma_f32(float noundef [[CALL23]]) #[[ATTR4]]
  %call25 = call contract float @tgammaf(float noundef %call23) #4
  ; CHECK-NEXT:    [[CALL26:%.*]] = call contract double @__hipstdpar_lgamma_f64(double noundef [[CALL24]]) #[[ATTR3]]
  %call26 = call contract double @lgamma(double noundef %call24) #5
  ; CHECK-NEXT:    [[CALL27:%.*]] = call contract float @__hipstdpar_lgamma_f32(float noundef [[CALL25]]) #[[ATTR3]]
  %call27 = call contract float @lgammaf(float noundef %call25) #5
  ret void
}

declare double @llvm.fabs.f64(double) #1

declare float @llvm.fabs.f32(float) #1

declare hidden double @remainder(double noundef, double noundef) local_unnamed_addr #2

declare hidden float @remainderf(float noundef, float noundef) local_unnamed_addr #2

declare hidden double @remquo(double noundef, double noundef, ptr noundef) local_unnamed_addr #3

declare hidden float @remquof(float noundef, float noundef, ptr noundef) local_unnamed_addr #3

declare double @llvm.fma.f64(double, double, double) #1

declare float @llvm.fma.f32(float, float, float) #1

declare hidden double @fdim(double noundef, double noundef) local_unnamed_addr #2

declare hidden float @fdimf(float noundef, float noundef) local_unnamed_addr #2

declare double @llvm.exp.f64(double) #1

declare float @llvm.exp.f32(float) #1

declare double @llvm.exp2.f64(double) #1

declare float @llvm.exp2.f32(float) #1

declare hidden double @expm1(double noundef) local_unnamed_addr #2

declare hidden float @expm1f(float noundef) local_unnamed_addr #2

declare double @llvm.log.f64(double) #1

declare float @llvm.log.f32(float) #1

declare double @llvm.log10.f64(double) #1

declare float @llvm.log10.f32(float) #1

declare double @llvm.log2.f64(double) #1

declare float @llvm.log2.f32(float) #1

declare hidden double @log1p(double noundef) local_unnamed_addr #2

declare hidden float @log1pf(float noundef) local_unnamed_addr #2

declare float @llvm.pow.f32(float, float) #1

declare double @llvm.sqrt.f64(double) #1

declare float @llvm.sqrt.f32(float) #1

declare hidden double @cbrt(double noundef) local_unnamed_addr #2

declare hidden float @cbrtf(float noundef) local_unnamed_addr #2

declare hidden double @hypot(double noundef, double noundef) local_unnamed_addr #2

declare hidden float @hypotf(float noundef, float noundef) local_unnamed_addr #2

declare float @llvm.sin.f32(float) #1

declare float @llvm.cos.f32(float) #1

declare double @llvm.tan.f64(double) #1

declare double @llvm.asin.f64(double) #1

declare double @llvm.acos.f64(double) #1

declare double @llvm.atan.f64(double) #1

declare double @llvm.atan2.f64(double, double) #1

declare double @llvm.sinh.f64(double) #1

declare double @llvm.cosh.f64(double) #1

declare double @llvm.tanh.f64(double) #1

declare hidden double @asinh(double noundef) local_unnamed_addr #2

declare hidden float @asinhf(float noundef) local_unnamed_addr #2

declare hidden double @acosh(double noundef) local_unnamed_addr #2

declare hidden float @acoshf(float noundef) local_unnamed_addr #2

declare hidden double @atanh(double noundef) local_unnamed_addr #2

declare hidden float @atanhf(float noundef) local_unnamed_addr #2

declare hidden double @erf(double noundef) local_unnamed_addr #2

declare hidden float @erff(float noundef) local_unnamed_addr #2

declare hidden double @erfc(double noundef) local_unnamed_addr #2

declare hidden float @erfcf(float noundef) local_unnamed_addr #2

declare hidden double @tgamma(double noundef) local_unnamed_addr #2

declare hidden float @tgammaf(float noundef) local_unnamed_addr #2

declare hidden double @lgamma(double noundef) local_unnamed_addr #3

declare hidden float @lgammaf(float noundef) local_unnamed_addr #3

attributes #0 = { convergent mustprogress norecurse nounwind }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent mustprogress nofree nounwind willreturn memory(none) }
attributes #3 = { convergent nounwind }
attributes #4 = { convergent nounwind willreturn memory(none) }
attributes #5 = { convergent nounwind }
