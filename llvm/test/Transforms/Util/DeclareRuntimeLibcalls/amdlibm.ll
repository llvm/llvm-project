; REQUIRES: x86-registered-target
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=x86_64-unknown-linux -vector-library=AMDLIBM < %s | FileCheck %s

; CHECK: declare <2 x double> @amd_vrd2_acos(<2 x double>) [[ATTRS:#[0-9]+]]

; CHECK: declare <2 x double> @amd_vrd2_asin(<2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @amd_vrd2_atan(<2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @amd_vrd2_atan2(<2 x double>, <2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @amd_vrd2_cbrt(<2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @amd_vrd2_cdfnorm(<2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @amd_vrd2_cdfnorminv(<2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @amd_vrd2_cos(<2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @amd_vrd2_cosh(<2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @amd_vrd2_erf(<2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @amd_vrd2_erfc(<2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @amd_vrd2_erfcinv(<2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @amd_vrd2_erfinv(<2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @amd_vrd2_exp(<2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @amd_vrd2_exp10(<2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @amd_vrd2_exp2(<2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @amd_vrd2_log(<2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @amd_vrd2_log10(<2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @amd_vrd2_log1p(<2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @amd_vrd2_log2(<2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @amd_vrd2_pow(<2 x double>, <2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @amd_vrd2_round(<2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @amd_vrd2_sin(<2 x double>) [[ATTRS]]

; CHECK: declare void @amd_vrd2_sincos(<2 x double>, ptr noalias nonnull writeonly align 16, ptr noalias nonnull writeonly align 16) [[ATTRS_PTR_ARG:#[0-9]+]]

; CHECK: declare <2 x double> @amd_vrd2_tan(<2 x double>) [[ATTRS]]

; CHECK: declare <2 x double> @amd_vrd2_tanh(<2 x double>) [[ATTRS]]

; CHECK: declare <4 x double> @amd_vrd4_acos(<4 x double>) [[ATTRS]]

; CHECK: declare <4 x double> @amd_vrd4_asin(<4 x double>) [[ATTRS]]

; CHECK: declare <4 x double> @amd_vrd4_atan(<4 x double>) [[ATTRS]]

; CHECK: declare <4 x double> @amd_vrd4_atan2(<4 x double>, <4 x double>) [[ATTRS]]

; CHECK: declare <4 x double> @amd_vrd4_cdfnorm(<4 x double>) [[ATTRS]]

; CHECK: declare <4 x double> @amd_vrd4_cdfnorminv(<4 x double>) [[ATTRS]]

; CHECK: declare <4 x double> @amd_vrd4_cos(<4 x double>) [[ATTRS]]

; CHECK: declare <4 x double> @amd_vrd4_cosh(<4 x double>) [[ATTRS]]

; CHECK: declare <4 x double> @amd_vrd4_erf(<4 x double>) [[ATTRS]]

; CHECK: declare <4 x double> @amd_vrd4_erfc(<4 x double>) [[ATTRS]]

; CHECK: declare <4 x double> @amd_vrd4_erfcinv(<4 x double>) [[ATTRS]]

; CHECK: declare <4 x double> @amd_vrd4_erfinv(<4 x double>) [[ATTRS]]

; CHECK: declare <4 x double> @amd_vrd4_exp(<4 x double>) [[ATTRS]]

; CHECK: declare <4 x double> @amd_vrd4_exp10(<4 x double>) [[ATTRS]]

; CHECK: declare <4 x double> @amd_vrd4_exp2(<4 x double>) [[ATTRS]]

; CHECK: declare <4 x double> @amd_vrd4_log(<4 x double>) [[ATTRS]]

; CHECK: declare <4 x double> @amd_vrd4_log2(<4 x double>) [[ATTRS]]

; CHECK: declare <4 x double> @amd_vrd4_pow(<4 x double>, <4 x double>) [[ATTRS]]

; CHECK: declare <4 x double> @amd_vrd4_round(<4 x double>) [[ATTRS]]

; CHECK: declare <4 x double> @amd_vrd4_sin(<4 x double>) [[ATTRS]]

; CHECK: declare void @amd_vrd4_sincos(<4 x double>, ptr noalias nonnull writeonly align 32, ptr noalias nonnull writeonly align 32) [[ATTRS_PTR_ARG]]

; CHECK: declare <4 x double> @amd_vrd4_tan(<4 x double>) [[ATTRS]]

; CHECK: declare <4 x double> @amd_vrd4_tanh(<4 x double>) [[ATTRS]]

; CHECK: declare <8 x double> @amd_vrd8_acos(<8 x double>) [[ATTRS]]

; CHECK: declare <8 x double> @amd_vrd8_asin(<8 x double>) [[ATTRS]]

; CHECK: declare <8 x double> @amd_vrd8_atan(<8 x double>) [[ATTRS]]

; CHECK: declare <8 x double> @amd_vrd8_atan2(<8 x double>, <8 x double>) [[ATTRS]]

; CHECK: declare <8 x double> @amd_vrd8_cdfnorm(<8 x double>) [[ATTRS]]

; CHECK: declare <8 x double> @amd_vrd8_cdfnorminv(<8 x double>) [[ATTRS]]

; CHECK: declare <8 x double> @amd_vrd8_cos(<8 x double>) [[ATTRS]]

; CHECK: declare <8 x double> @amd_vrd8_cosh(<8 x double>) [[ATTRS]]

; CHECK: declare <8 x double> @amd_vrd8_erf(<8 x double>) [[ATTRS]]

; CHECK: declare <8 x double> @amd_vrd8_erfc(<8 x double>) [[ATTRS]]

; CHECK: declare <8 x double> @amd_vrd8_erfcinv(<8 x double>) [[ATTRS]]

; CHECK: declare <8 x double> @amd_vrd8_erfinv(<8 x double>) [[ATTRS]]

; CHECK: declare <8 x double> @amd_vrd8_exp(<8 x double>) [[ATTRS]]

; CHECK: declare <8 x double> @amd_vrd8_exp10(<8 x double>) [[ATTRS]]

; CHECK: declare <8 x double> @amd_vrd8_exp2(<8 x double>) [[ATTRS]]

; CHECK: declare <8 x double> @amd_vrd8_log(<8 x double>) [[ATTRS]]

; CHECK: declare <8 x double> @amd_vrd8_log2(<8 x double>) [[ATTRS]]

; CHECK: declare <8 x double> @amd_vrd8_pow(<8 x double>, <8 x double>) [[ATTRS]]

; CHECK: declare <8 x double> @amd_vrd8_round(<8 x double>) [[ATTRS]]

; CHECK: declare <8 x double> @amd_vrd8_sin(<8 x double>) [[ATTRS]]

; CHECK: declare void @amd_vrd8_sincos(<8 x double>, ptr noalias nonnull writeonly align 64, ptr noalias nonnull writeonly align 64) [[ATTRS_PTR_ARG]]

; CHECK: declare <8 x double> @amd_vrd8_tan(<8 x double>) [[ATTRS]]

; CHECK: declare <8 x double> @amd_vrd8_tanh(<8 x double>) [[ATTRS]]

; CHECK: declare <16 x float> @amd_vrs16_acosf(<16 x float>) [[ATTRS]]

; CHECK: declare <16 x float> @amd_vrs16_asinf(<16 x float>) [[ATTRS]]

; CHECK: declare <16 x float> @amd_vrs16_atanf(<16 x float>) [[ATTRS]]

; CHECK: declare <16 x float> @amd_vrs16_cosf(<16 x float>) [[ATTRS]]

; CHECK: declare <16 x float> @amd_vrs16_coshf(<16 x float>) [[ATTRS]]

; CHECK: declare <16 x float> @amd_vrs16_erfcf(<16 x float>) [[ATTRS]]

; CHECK: declare <16 x float> @amd_vrs16_erff(<16 x float>) [[ATTRS]]

; CHECK: declare <16 x float> @amd_vrs16_exp10f(<16 x float>) [[ATTRS]]

; CHECK: declare <16 x float> @amd_vrs16_exp2f(<16 x float>) [[ATTRS]]

; CHECK: declare <16 x float> @amd_vrs16_expf(<16 x float>) [[ATTRS]]

; CHECK: declare <16 x float> @amd_vrs16_log10f(<16 x float>) [[ATTRS]]

; CHECK: declare <16 x float> @amd_vrs16_log2f(<16 x float>) [[ATTRS]]

; CHECK: declare <16 x float> @amd_vrs16_logf(<16 x float>) [[ATTRS]]

; CHECK: declare <16 x float> @amd_vrs16_powf(<16 x float>, <16 x float>) [[ATTRS]]

; CHECK: declare <16 x float> @amd_vrs16_roundf(<16 x float>) [[ATTRS]]

; CHECK: declare void @amd_vrs16_sincosf(<16 x float>, ptr noalias nonnull writeonly align 64, ptr noalias nonnull writeonly align 64) [[ATTRS_PTR_ARG]]

; CHECK: declare <16 x float> @amd_vrs16_sinf(<16 x float>) [[ATTRS]]

; CHECK: declare <16 x float> @amd_vrs16_tanf(<16 x float>) [[ATTRS]]

; CHECK: declare <16 x float> @amd_vrs16_tanhf(<16 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @amd_vrs4_acosf(<4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @amd_vrs4_asinf(<4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @amd_vrs4_atanf(<4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @amd_vrs4_cbrtf(<4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @amd_vrs4_cosf(<4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @amd_vrs4_coshf(<4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @amd_vrs4_erfcf(<4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @amd_vrs4_erff(<4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @amd_vrs4_exp10f(<4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @amd_vrs4_exp2f(<4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @amd_vrs4_expf(<4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @amd_vrs4_expm1f(<4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @amd_vrs4_log10f(<4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @amd_vrs4_log1pf(<4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @amd_vrs4_log2f(<4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @amd_vrs4_logf(<4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @amd_vrs4_powf(<4 x float>, <4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @amd_vrs4_roundf(<4 x float>) [[ATTRS]]

; CHECK: declare void @amd_vrs4_sincosf(<4 x float>, ptr noalias nonnull writeonly align 16, ptr noalias nonnull writeonly align 16) [[ATTRS_PTR_ARG]]

; CHECK: declare <4 x float> @amd_vrs4_sinf(<4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @amd_vrs4_tanf(<4 x float>) [[ATTRS]]

; CHECK: declare <4 x float> @amd_vrs4_tanhf(<4 x float>) [[ATTRS]]

; CHECK: declare <8 x float> @amd_vrs8_acosf(<8 x float>) [[ATTRS]]

; CHECK: declare <8 x float> @amd_vrs8_asinf(<8 x float>) [[ATTRS]]

; CHECK: declare <8 x float> @amd_vrs8_atanf(<8 x float>) [[ATTRS]]

; CHECK: declare <8 x float> @amd_vrs8_cbrtf(<8 x float>) [[ATTRS]]

; CHECK: declare <8 x float> @amd_vrs8_cosf(<8 x float>) [[ATTRS]]

; CHECK: declare <8 x float> @amd_vrs8_coshf(<8 x float>) [[ATTRS]]

; CHECK: declare <8 x float> @amd_vrs8_erfcf(<8 x float>) [[ATTRS]]

; CHECK: declare <8 x float> @amd_vrs8_erff(<8 x float>) [[ATTRS]]

; CHECK: declare <8 x float> @amd_vrs8_exp10f(<8 x float>) [[ATTRS]]

; CHECK: declare <8 x float> @amd_vrs8_exp2f(<8 x float>) [[ATTRS]]

; CHECK: declare <8 x float> @amd_vrs8_expf(<8 x float>) [[ATTRS]]

; CHECK: declare <8 x float> @amd_vrs8_log10f(<8 x float>) [[ATTRS]]

; CHECK: declare <8 x float> @amd_vrs8_log2f(<8 x float>) [[ATTRS]]

; CHECK: declare <8 x float> @amd_vrs8_logf(<8 x float>) [[ATTRS]]

; CHECK: declare <8 x float> @amd_vrs8_powf(<8 x float>, <8 x float>) [[ATTRS]]

; CHECK: declare <8 x float> @amd_vrs8_roundf(<8 x float>) [[ATTRS]]

; CHECK: declare void @amd_vrs8_sincosf(<8 x float>, ptr noalias nonnull writeonly align 32, ptr noalias nonnull writeonly align 32) [[ATTRS_PTR_ARG]]

; CHECK: declare <8 x float> @amd_vrs8_sinf(<8 x float>) [[ATTRS]]

; CHECK: declare <8 x float> @amd_vrs8_tanf(<8 x float>) [[ATTRS]]

; CHECK: declare <8 x float> @amd_vrs8_tanhf(<8 x float>) [[ATTRS]]

; CHECK: attributes [[ATTRS]] = { nocallback nofree nosync nounwind willreturn }
; CHECK: attributes [[ATTRS_PTR_ARG]] = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }
