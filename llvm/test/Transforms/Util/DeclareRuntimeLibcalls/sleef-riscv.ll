; REQUIRES: riscv-registered-target, x86-registered-target
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=riscv64-unknown-linux -mattr=+v -vector-library=sleefgnuabi < %s | FileCheck %s --implicit-check-not=_ZGV --implicit-check-not=Sleef_modf --implicit-check-not=Sleef_sincos
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=riscv32-unknown-linux-gnu -mattr=+v -vector-library=sleefgnuabi < %s | FileCheck %s --check-prefix=NONE --implicit-check-not=_ZGV --implicit-check-not=Sleef_
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=x86_64-unknown-linux-gnu -vector-library=sleefgnuabi < %s | FileCheck %s --check-prefix=NONE --implicit-check-not=_ZGV --implicit-check-not=Sleef_
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=riscv64-unknown-linux-gnu -mattr=+v -vector-library=none < %s | FileCheck %s --check-prefix=NONE --implicit-check-not=_ZGV --implicit-check-not=Sleef_

; CHECK: declare <vscale x 2 x double> @Sleef_acosdx_u10rvvm2(<vscale x 2 x double>) [[ATTRS:#[0-9]+]]

; CHECK: declare <vscale x 4 x float> @Sleef_acosfx_u10rvvm2(<vscale x 4 x float>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @Sleef_acoshdx_u10rvvm2(<vscale x 2 x double>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @Sleef_acoshfx_u10rvvm2(<vscale x 4 x float>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @Sleef_asindx_u10rvvm2(<vscale x 2 x double>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @Sleef_asinfx_u10rvvm2(<vscale x 4 x float>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @Sleef_asinhdx_u10rvvm2(<vscale x 2 x double>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @Sleef_asinhfx_u10rvvm2(<vscale x 4 x float>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @Sleef_atan2dx_u10rvvm2(<vscale x 2 x double>, <vscale x 2 x double>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @Sleef_atan2fx_u10rvvm2(<vscale x 4 x float>, <vscale x 4 x float>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @Sleef_atandx_u10rvvm2(<vscale x 2 x double>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @Sleef_atanfx_u10rvvm2(<vscale x 4 x float>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @Sleef_atanhdx_u10rvvm2(<vscale x 2 x double>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @Sleef_atanhfx_u10rvvm2(<vscale x 4 x float>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @Sleef_cbrtdx_u10rvvm2(<vscale x 2 x double>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @Sleef_cbrtfx_u10rvvm2(<vscale x 4 x float>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @Sleef_copysigndx_rvvm2(<vscale x 2 x double>, <vscale x 2 x double>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @Sleef_copysignfx_rvvm2(<vscale x 4 x float>, <vscale x 4 x float>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @Sleef_cosdx_u10rvvm2(<vscale x 2 x double>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @Sleef_cosfx_u10rvvm2(<vscale x 4 x float>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @Sleef_coshdx_u10rvvm2(<vscale x 2 x double>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @Sleef_coshfx_u10rvvm2(<vscale x 4 x float>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @Sleef_cospidx_u05rvvm2(<vscale x 2 x double>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @Sleef_cospifx_u05rvvm2(<vscale x 4 x float>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @Sleef_erfcdx_u15rvvm2(<vscale x 2 x double>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @Sleef_erfcfx_u15rvvm2(<vscale x 4 x float>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @Sleef_erfdx_u10rvvm2(<vscale x 2 x double>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @Sleef_erffx_u10rvvm2(<vscale x 4 x float>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @Sleef_exp10dx_u10rvvm2(<vscale x 2 x double>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @Sleef_exp10fx_u10rvvm2(<vscale x 4 x float>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @Sleef_exp2dx_u10rvvm2(<vscale x 2 x double>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @Sleef_exp2fx_u10rvvm2(<vscale x 4 x float>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @Sleef_expdx_u10rvvm2(<vscale x 2 x double>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @Sleef_expfx_u10rvvm2(<vscale x 4 x float>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @Sleef_expm1dx_u10rvvm2(<vscale x 2 x double>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @Sleef_expm1fx_u10rvvm2(<vscale x 4 x float>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @Sleef_fdimdx_rvvm2(<vscale x 2 x double>, <vscale x 2 x double>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @Sleef_fdimfx_rvvm2(<vscale x 4 x float>, <vscale x 4 x float>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @Sleef_fmadx_rvvm2(<vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @Sleef_fmafx_rvvm2(<vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @Sleef_fmaxdx_rvvm2(<vscale x 2 x double>, <vscale x 2 x double>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @Sleef_fmaxfx_rvvm2(<vscale x 4 x float>, <vscale x 4 x float>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @Sleef_fmindx_rvvm2(<vscale x 2 x double>, <vscale x 2 x double>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @Sleef_fminfx_rvvm2(<vscale x 4 x float>, <vscale x 4 x float>) [[ATTRS]]

; RVV fmod has two vector arguments and no mask. The multi-result functions
; require a different ABI and must remain unavailable until it is supported.
; CHECK: declare <vscale x 2 x double> @Sleef_fmoddx_rvvm2(<vscale x 2 x double>, <vscale x 2 x double>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @Sleef_fmodfx_rvvm2(<vscale x 4 x float>, <vscale x 4 x float>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @Sleef_hypotdx_u05rvvm2(<vscale x 2 x double>, <vscale x 2 x double>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @Sleef_hypotfx_u05rvvm2(<vscale x 4 x float>, <vscale x 4 x float>) [[ATTRS]]

; CHECK: declare <vscale x 2 x i32> @Sleef_ilogbdx_rvvm2(<vscale x 2 x double>) [[ATTRS]]

; CHECK: declare <vscale x 4 x i32> @Sleef_ilogbfx_rvvm2(<vscale x 4 x float>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @Sleef_ldexpdx_rvvm2(<vscale x 2 x double>, <vscale x 2 x i32>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @Sleef_ldexpfx_rvvm2(<vscale x 4 x float>, <vscale x 4 x i32>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @Sleef_lgammadx_u10rvvm2(<vscale x 2 x double>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @Sleef_lgammafx_u10rvvm2(<vscale x 4 x float>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @Sleef_log10dx_u10rvvm2(<vscale x 2 x double>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @Sleef_log10fx_u10rvvm2(<vscale x 4 x float>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @Sleef_log1pdx_u10rvvm2(<vscale x 2 x double>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @Sleef_log1pfx_u10rvvm2(<vscale x 4 x float>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @Sleef_log2dx_u10rvvm2(<vscale x 2 x double>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @Sleef_log2fx_u10rvvm2(<vscale x 4 x float>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @Sleef_logdx_u10rvvm2(<vscale x 2 x double>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @Sleef_logfx_u10rvvm2(<vscale x 4 x float>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @Sleef_nextafterdx_rvvm2(<vscale x 2 x double>, <vscale x 2 x double>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @Sleef_nextafterfx_rvvm2(<vscale x 4 x float>, <vscale x 4 x float>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @Sleef_powdx_u10rvvm2(<vscale x 2 x double>, <vscale x 2 x double>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @Sleef_powfx_u10rvvm2(<vscale x 4 x float>, <vscale x 4 x float>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @Sleef_sindx_u10rvvm2(<vscale x 2 x double>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @Sleef_sinfx_u10rvvm2(<vscale x 4 x float>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @Sleef_sinhdx_u10rvvm2(<vscale x 2 x double>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @Sleef_sinhfx_u10rvvm2(<vscale x 4 x float>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @Sleef_sinpidx_u05rvvm2(<vscale x 2 x double>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @Sleef_sinpifx_u05rvvm2(<vscale x 4 x float>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @Sleef_sqrtdx_u05rvvm2(<vscale x 2 x double>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @Sleef_sqrtfx_u05rvvm2(<vscale x 4 x float>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @Sleef_tandx_u10rvvm2(<vscale x 2 x double>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @Sleef_tanfx_u10rvvm2(<vscale x 4 x float>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @Sleef_tanhdx_u10rvvm2(<vscale x 2 x double>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @Sleef_tanhfx_u10rvvm2(<vscale x 4 x float>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @Sleef_tgammadx_u10rvvm2(<vscale x 2 x double>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @Sleef_tgammafx_u10rvvm2(<vscale x 4 x float>) [[ATTRS]]

; CHECK: attributes [[ATTRS]] = { nocallback nofree nosync nounwind willreturn }

; The vector library selection must not enable functions for another target,
; and the RVV functions must not enter the default system runtime library.
; NONE: declare double @fmod(double, double)
