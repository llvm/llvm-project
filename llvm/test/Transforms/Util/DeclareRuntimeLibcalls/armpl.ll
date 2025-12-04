; REQUIRES: aarch64-registered-target
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=aarch64-unknown-linux -mattr=+neon,+sve -vector-library=ArmPL < %s | FileCheck %s

; CHECK: declare <vscale x 4 x float> @armpl_svmodf_f32_x(<vscale x 4 x float>, ptr noalias nonnull writeonly align 16, <vscale x 4 x i1>) [[ATTRS:#[0-9]+]]

; CHECK: declare <vscale x 2 x double> @armpl_svmodf_f64_x(<vscale x 2 x double>, ptr noalias nonnull writeonly align 16, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare void @armpl_svsincos_f32_x(<vscale x 4 x float>, ptr noalias nonnull writeonly align 16, ptr noalias nonnull writeonly align 16, <vscale x 4 x i1>) [[ATTRS:#[0-9]+]]

; CHECK: declare void @armpl_svsincos_f64_x(<vscale x 2 x double>, ptr noalias nonnull writeonly align 16, ptr noalias nonnull writeonly align 16, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare void @armpl_svsincospi_f32_x(<vscale x 4 x float>, ptr noalias nonnull writeonly align 16, ptr noalias nonnull writeonly align 16, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare void @armpl_svsincospi_f64_x(<vscale x 2 x double>, ptr noalias nonnull writeonly align 16, ptr noalias nonnull writeonly align 16, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <4 x float> @armpl_vmodfq_f32(<4 x float>, ptr noalias nonnull writeonly align 16) [[ATTRS]]

; CHECK: declare <2 x double> @armpl_vmodfq_f64(<2 x double>, ptr noalias nonnull writeonly align 16) [[ATTRS]]

; CHECK: declare void @armpl_vsincospiq_f32(<4 x float>, ptr noalias nonnull writeonly align 16, ptr noalias nonnull writeonly align 16) [[ATTRS]]

; CHECK: declare void @armpl_vsincospiq_f64(<2 x double>, ptr noalias nonnull writeonly align 16, ptr noalias nonnull writeonly align 16) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs void @armpl_vsincosq_f32(<4 x float>, ptr noalias nonnull writeonly align 16, ptr noalias nonnull writeonly align 16) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs void @armpl_vsincosq_f64(<2 x double>, ptr noalias nonnull writeonly align 16, ptr noalias nonnull writeonly align 16) [[ATTRS]]


; CHECK: attributes [[ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }
