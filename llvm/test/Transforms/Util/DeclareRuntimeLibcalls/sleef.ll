; REQUIRES: aarch64-registered-target
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=aarch64-unknown-linux -mattr=+neon,+sve -vector-library=sleefgnuabi < %s | FileCheck %s

; CHECK: declare <2 x double> @_ZGVnN2vl8_modf(<2 x double>, ptr noalias nonnull writeonly align 16) [[ATTRS:#[0-9]+]]

; CHECK: declare void @_ZGVnN2vl8l8_sincos(<2 x double>, ptr noalias nonnull writeonly align 16, ptr noalias nonnull writeonly align 16) [[ATTRS]]

; CHECK: declare void @_ZGVnN2vl8l8_sincospi(<2 x double>, ptr noalias nonnull writeonly align 16, ptr noalias nonnull writeonly align 16) [[ATTRS]]

; CHECK: declare <4 x float> @_ZGVnN4vl4_modff(<4 x float>, ptr noalias nonnull writeonly align 16) [[ATTRS]]

; CHECK: declare void @_ZGVnN4vl4l4_sincosf(<4 x float>, ptr noalias nonnull writeonly align 16, ptr noalias nonnull writeonly align 16) [[ATTRS]]

; CHECK: declare void @_ZGVnN4vl4l4_sincospif(<4 x float>, ptr noalias nonnull writeonly align 16, ptr noalias nonnull writeonly align 16) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @_ZGVsNxvl4_modff(<vscale x 4 x float>, ptr noalias nonnull writeonly align 16) [[ATTRS]]

; CHECK: declare void @_ZGVsNxvl4l4_sincosf(<vscale x 4 x float>, ptr noalias nonnull writeonly align 16, ptr noalias nonnull writeonly align 16) [[ATTRS]]

; CHECK: declare void @_ZGVsNxvl4l4_sincospif(<vscale x 4 x float>, ptr noalias nonnull writeonly align 16, ptr noalias nonnull writeonly align 16) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @_ZGVsNxvl8_modf(<vscale x 2 x double>, ptr noalias nonnull writeonly align 16) [[ATTRS]]

; CHECK: declare void @_ZGVsNxvl8l8_sincos(<vscale x 2 x double>, ptr noalias nonnull writeonly align 16, ptr noalias nonnull writeonly align 16) [[ATTRS]]

; CHECK: declare void @_ZGVsNxvl8l8_sincospi(<vscale x 2 x double>, ptr noalias nonnull writeonly align 16, ptr noalias nonnull writeonly align 16) [[ATTRS]]

; CHECK: attributes [[ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }
