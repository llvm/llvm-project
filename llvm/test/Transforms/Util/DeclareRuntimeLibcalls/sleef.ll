; REQUIRES: aarch64-registered-target
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=aarch64-unknown-linux -mattr=+neon,+sve -vector-library=sleefgnuabi < %s | FileCheck %s

; CHECK: declare <2 x double> @_ZGVnN2vl8_modf(<2 x double>, ptr noalias nonnull writeonly align 16) [[ATTRS_PTR_ARG:#[0-9]+]]

; CHECK: declare void @_ZGVnN2vl8l8_sincos(<2 x double>, ptr noalias nonnull writeonly align 16, ptr noalias nonnull writeonly align 16) [[ATTRS_PTR_ARG]]

; CHECK: declare void @_ZGVnN2vl8l8_sincospi(<2 x double>, ptr noalias nonnull writeonly align 16, ptr noalias nonnull writeonly align 16) [[ATTRS_PTR_ARG]]

; CHECK: declare <2 x double> @_ZGVnN2vv_fmod(<2 x double>, <2 x double>) [[ATTRS:#[0-9]+]]


; CHECK: declare <4 x float> @_ZGVnN4vl4_modff(<4 x float>, ptr noalias nonnull writeonly align 16) [[ATTRS_PTR_ARG]]

; CHECK: declare void @_ZGVnN4vl4l4_sincosf(<4 x float>, ptr noalias nonnull writeonly align 16, ptr noalias nonnull writeonly align 16) [[ATTRS_PTR_ARG]]

; CHECK: declare void @_ZGVnN4vl4l4_sincospif(<4 x float>, ptr noalias nonnull writeonly align 16, ptr noalias nonnull writeonly align 16) [[ATTRS_PTR_ARG]]

; CHECK: declare <4 x float> @_ZGVnN4vv_fmodf(<4 x float>, <4 x float>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @_ZGVsMxvv_fmod(<vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @_ZGVsMxvv_fmodf(<vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @_ZGVsNxvl4_modff(<vscale x 4 x float>, ptr noalias nonnull writeonly align 16) [[ATTRS_PTR_ARG]]

; CHECK: declare void @_ZGVsNxvl4l4_sincosf(<vscale x 4 x float>, ptr noalias nonnull writeonly align 16, ptr noalias nonnull writeonly align 16) [[ATTRS_PTR_ARG]]

; CHECK: declare void @_ZGVsNxvl4l4_sincospif(<vscale x 4 x float>, ptr noalias nonnull writeonly align 16, ptr noalias nonnull writeonly align 16) [[ATTRS_PTR_ARG]]

; CHECK: declare <vscale x 2 x double> @_ZGVsNxvl8_modf(<vscale x 2 x double>, ptr noalias nonnull writeonly align 16) [[ATTRS_PTR_ARG]]

; CHECK: declare void @_ZGVsNxvl8l8_sincos(<vscale x 2 x double>, ptr noalias nonnull writeonly align 16, ptr noalias nonnull writeonly align 16) [[ATTRS_PTR_ARG]]

; CHECK: declare void @_ZGVsNxvl8l8_sincospi(<vscale x 2 x double>, ptr noalias nonnull writeonly align 16, ptr noalias nonnull writeonly align 16) [[ATTRS_PTR_ARG]]

; CHECK: attributes [[ATTRS_PTR_ARG]] = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: write) }
; CHECK: attributes [[ATTRS]] = { mustprogress nocallback nofree nosync nounwind willreturn }
