; REQUIRES: aarch64-registered-target
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=aarch64-unknown-linux -mattr=+neon,+sve -vector-library=sleefgnuabi < %s | FileCheck %s

; CHECK: declare <2 x double> @_ZGVnN2v_acos(<2 x double>) [[ATTRS:#[0-9]+]]

; CHECK: declare <2 x double> @_ZGVnN2v_acosh(<2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @_ZGVnN2v_asin(<2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @_ZGVnN2v_asinh(<2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @_ZGVnN2v_atan(<2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @_ZGVnN2v_atanh(<2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @_ZGVnN2v_cbrt(<2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @_ZGVnN2v_cos(<2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @_ZGVnN2v_cosh(<2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @_ZGVnN2v_cospi(<2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @_ZGVnN2v_erf(<2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @_ZGVnN2v_erfc(<2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @_ZGVnN2v_exp(<2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @_ZGVnN2v_exp10(<2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @_ZGVnN2v_exp2(<2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @_ZGVnN2v_expm1(<2 x double>) [[ATTRS]]

; CHECK: declare <2 x i32> @_ZGVnN2v_ilogb(<2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @_ZGVnN2v_lgamma(<2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @_ZGVnN2v_log(<2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @_ZGVnN2v_log10(<2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @_ZGVnN2v_log1p(<2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @_ZGVnN2v_log2(<2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @_ZGVnN2v_sin(<2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @_ZGVnN2v_sinh(<2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @_ZGVnN2v_sinpi(<2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @_ZGVnN2v_sqrt(<2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @_ZGVnN2v_tan(<2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @_ZGVnN2v_tanh(<2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @_ZGVnN2v_tgamma(<2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @_ZGVnN2vl8_modf(<2 x double>, ptr noalias nonnull writeonly align 16) [[ATTRS_PTR_ARG:#[0-9]+]]

; CHECK: declare void @_ZGVnN2vl8l8_sincos(<2 x double>, ptr noalias nonnull writeonly align 16, ptr noalias nonnull writeonly align 16) [[ATTRS_PTR_ARG]]

; CHECK: declare void @_ZGVnN2vl8l8_sincospi(<2 x double>, ptr noalias nonnull writeonly align 16, ptr noalias nonnull writeonly align 16) [[ATTRS_PTR_ARG]]

; CHECK: declare <2 x double> @_ZGVnN2vv_atan2(<2 x double>, <2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @_ZGVnN2vv_copysign(<2 x double>, <2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @_ZGVnN2vv_fdim(<2 x double>, <2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @_ZGVnN2vv_fmax(<2 x double>, <2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @_ZGVnN2vv_fmin(<2 x double>, <2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @_ZGVnN2vv_fmod(<2 x double>, <2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @_ZGVnN2vv_hypot(<2 x double>, <2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @_ZGVnN2vv_ldexp(<2 x double>, <2 x i32>) [[ATTRS]]

; CHECK: declare <2 x double> @_ZGVnN2vv_nextafter(<2 x double>, <2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @_ZGVnN2vv_pow(<2 x double>, <2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @_ZGVnN2vvv_fma(<2 x double>, <2 x double>, <2 x double>) [[ATTRS]]

; CHECK: declare <4 x float> @_ZGVnN4v_acosf(<4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @_ZGVnN4v_acoshf(<4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @_ZGVnN4v_asinf(<4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @_ZGVnN4v_asinhf(<4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @_ZGVnN4v_atanf(<4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @_ZGVnN4v_atanhf(<4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @_ZGVnN4v_cbrtf(<4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @_ZGVnN4v_cosf(<4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @_ZGVnN4v_coshf(<4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @_ZGVnN4v_cospif(<4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @_ZGVnN4v_erfcf(<4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @_ZGVnN4v_erff(<4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @_ZGVnN4v_exp10f(<4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @_ZGVnN4v_exp2f(<4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @_ZGVnN4v_expf(<4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @_ZGVnN4v_expm1f(<4 x float>) [[ATTRS]]

; CHECK: declare <4 x i32> @_ZGVnN4v_ilogbf(<4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @_ZGVnN4v_lgammaf(<4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @_ZGVnN4v_log10f(<4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @_ZGVnN4v_log1pf(<4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @_ZGVnN4v_log2f(<4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @_ZGVnN4v_logf(<4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @_ZGVnN4v_sinf(<4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @_ZGVnN4v_sinhf(<4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @_ZGVnN4v_sinpif(<4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @_ZGVnN4v_sqrtf(<4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @_ZGVnN4v_tanf(<4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @_ZGVnN4v_tanhf(<4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @_ZGVnN4v_tgammaf(<4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @_ZGVnN4vl4_modff(<4 x float>, ptr noalias nonnull writeonly align 16) [[ATTRS_PTR_ARG]]

; CHECK: declare void @_ZGVnN4vl4l4_sincosf(<4 x float>, ptr noalias nonnull writeonly align 16, ptr noalias nonnull writeonly align 16) [[ATTRS_PTR_ARG]]

; CHECK: declare void @_ZGVnN4vl4l4_sincospif(<4 x float>, ptr noalias nonnull writeonly align 16, ptr noalias nonnull writeonly align 16) [[ATTRS_PTR_ARG]]

; CHECK: declare <4 x float> @_ZGVnN4vv_atan2f(<4 x float>, <4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @_ZGVnN4vv_copysignf(<4 x float>, <4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @_ZGVnN4vv_fdimf(<4 x float>, <4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @_ZGVnN4vv_fmaxf(<4 x float>, <4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @_ZGVnN4vv_fminf(<4 x float>, <4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @_ZGVnN4vv_fmodf(<4 x float>, <4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @_ZGVnN4vv_hypotf(<4 x float>, <4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @_ZGVnN4vv_ldexpf(<4 x float>, <4 x i32>) [[ATTRS]]

; CHECK: declare <4 x float> @_ZGVnN4vv_nextafterf(<4 x float>, <4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @_ZGVnN4vv_powf(<4 x float>, <4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @_ZGVnN4vvv_fmaf(<4 x float>, <4 x float>, <4 x float>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @_ZGVsMxv_acos(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @_ZGVsMxv_acosf(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @_ZGVsMxv_acosh(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @_ZGVsMxv_acoshf(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @_ZGVsMxv_asin(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @_ZGVsMxv_asinf(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @_ZGVsMxv_asinh(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @_ZGVsMxv_asinhf(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @_ZGVsMxv_atan(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @_ZGVsMxv_atanf(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @_ZGVsMxv_atanh(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @_ZGVsMxv_atanhf(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @_ZGVsMxv_cbrt(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @_ZGVsMxv_cbrtf(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @_ZGVsMxv_cos(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @_ZGVsMxv_cosf(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @_ZGVsMxv_cosh(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @_ZGVsMxv_coshf(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @_ZGVsMxv_cospi(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @_ZGVsMxv_cospif(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @_ZGVsMxv_erf(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @_ZGVsMxv_erfc(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @_ZGVsMxv_erfcf(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @_ZGVsMxv_erff(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @_ZGVsMxv_exp(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @_ZGVsMxv_exp10(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @_ZGVsMxv_exp10f(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @_ZGVsMxv_exp2(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @_ZGVsMxv_exp2f(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @_ZGVsMxv_expf(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @_ZGVsMxv_expm1(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @_ZGVsMxv_expm1f(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x i32> @_ZGVsMxv_ilogb(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x i32> @_ZGVsMxv_ilogbf(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @_ZGVsMxv_lgamma(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @_ZGVsMxv_lgammaf(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @_ZGVsMxv_log(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @_ZGVsMxv_log10(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @_ZGVsMxv_log10f(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @_ZGVsMxv_log1p(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @_ZGVsMxv_log1pf(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @_ZGVsMxv_log2(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @_ZGVsMxv_log2f(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @_ZGVsMxv_logf(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @_ZGVsMxv_sin(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @_ZGVsMxv_sinf(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @_ZGVsMxv_sinh(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @_ZGVsMxv_sinhf(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @_ZGVsMxv_sinpi(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @_ZGVsMxv_sinpif(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @_ZGVsMxv_sqrt(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @_ZGVsMxv_sqrtf(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @_ZGVsMxv_tan(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @_ZGVsMxv_tanf(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @_ZGVsMxv_tanh(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @_ZGVsMxv_tanhf(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @_ZGVsMxv_tgamma(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @_ZGVsMxv_tgammaf(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @_ZGVsMxvv_atan2(<vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @_ZGVsMxvv_atan2f(<vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @_ZGVsMxvv_copysign(<vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @_ZGVsMxvv_copysignf(<vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @_ZGVsMxvv_fdim(<vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @_ZGVsMxvv_fdimf(<vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @_ZGVsMxvv_fmax(<vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @_ZGVsMxvv_fmaxf(<vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @_ZGVsMxvv_fmin(<vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @_ZGVsMxvv_fminf(<vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @_ZGVsMxvv_fmod(<vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @_ZGVsMxvv_fmodf(<vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @_ZGVsMxvv_hypot(<vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @_ZGVsMxvv_hypotf(<vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @_ZGVsMxvv_ldexp(<vscale x 2 x double>, <vscale x 2 x i32>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @_ZGVsMxvv_ldexpf(<vscale x 4 x float>, <vscale x 4 x i32>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @_ZGVsMxvv_nextafter(<vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @_ZGVsMxvv_nextafterf(<vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @_ZGVsMxvv_pow(<vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @_ZGVsMxvv_powf(<vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @_ZGVsMxvvv_fma(<vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @_ZGVsMxvvv_fmaf(<vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @_ZGVsNxvl4_modff(<vscale x 4 x float>, ptr noalias nonnull writeonly align 16) [[ATTRS_PTR_ARG]]

; CHECK: declare void @_ZGVsNxvl4l4_sincosf(<vscale x 4 x float>, ptr noalias nonnull writeonly align 16, ptr noalias nonnull writeonly align 16) [[ATTRS_PTR_ARG]]

; CHECK: declare void @_ZGVsNxvl4l4_sincospif(<vscale x 4 x float>, ptr noalias nonnull writeonly align 16, ptr noalias nonnull writeonly align 16) [[ATTRS_PTR_ARG]]

; CHECK: declare <vscale x 2 x double> @_ZGVsNxvl8_modf(<vscale x 2 x double>, ptr noalias nonnull writeonly align 16) [[ATTRS_PTR_ARG]]

; CHECK: declare void @_ZGVsNxvl8l8_sincos(<vscale x 2 x double>, ptr noalias nonnull writeonly align 16, ptr noalias nonnull writeonly align 16) [[ATTRS_PTR_ARG]]

; CHECK: declare void @_ZGVsNxvl8l8_sincospi(<vscale x 2 x double>, ptr noalias nonnull writeonly align 16, ptr noalias nonnull writeonly align 16) [[ATTRS_PTR_ARG]]

; CHECK: attributes [[ATTRS]] = { nocallback nofree nosync nounwind willreturn }
; CHECK: attributes [[ATTRS_PTR_ARG]] = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }
