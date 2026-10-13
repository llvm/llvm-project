; REQUIRES: riscv-registered-target, x86-registered-target
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=riscv64-unknown-linux-gnu -mattr=+v -vector-library=sleefgnuabi < %s | FileCheck %s --check-prefix=RV64 --implicit-check-not=_ZGV --implicit-check-not=Sleef_modf --implicit-check-not=Sleef_sincos
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=riscv32-unknown-linux-gnu -mattr=+v -vector-library=sleefgnuabi < %s | FileCheck %s --check-prefix=NONE --implicit-check-not=_ZGV --implicit-check-not=Sleef_
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=x86_64-unknown-linux-gnu -vector-library=sleefgnuabi < %s | FileCheck %s --check-prefix=NONE --implicit-check-not=_ZGV --implicit-check-not=Sleef_
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=riscv64-unknown-linux-gnu -mattr=+v -vector-library=none < %s | FileCheck %s --check-prefix=NONE --implicit-check-not=_ZGV --implicit-check-not=Sleef_

; RVV fmod has two vector arguments and no mask. The multi-result functions
; require a different ABI and must remain unavailable until it is supported.
; RV64: declare <vscale x 2 x double> @Sleef_fmoddx_rvvm2(<vscale x 2 x double>, <vscale x 2 x double>) [[ATTRS:#[0-9]+]]

; RV64: declare <vscale x 4 x float> @Sleef_fmodfx_rvvm2(<vscale x 4 x float>, <vscale x 4 x float>) [[ATTRS]]

; RV64: attributes [[ATTRS]] = { nocallback nofree nosync nounwind willreturn }

; The vector library selection must not enable functions for another target,
; and the RVV functions must not enter the default system runtime library.
; NONE: declare double @fmod(double, double)
