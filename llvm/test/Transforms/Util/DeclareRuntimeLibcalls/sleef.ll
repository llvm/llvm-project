; REQUIRES: aarch64-registered-target
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=aarch64-unknown-linux -mattr=+neon,+sve -vector-library=sleefgnuabi < %s | FileCheck %s

; CHECK: declare void @_ZGVnN2vl8l8_sincos(<2 x double>, ptr noalias nonnull writeonly align 16, ptr noalias nonnull writeonly align 16) [[ATTRS:#[0-9]+]]

; CHECK: declare void @_ZGVnN2vl8l8_sincospi(<2 x double>, ptr noalias nonnull writeonly align 16, ptr noalias nonnull writeonly align 16) [[ATTRS]]

; CHECK: declare void @_ZGVnN4vl4l4_sincosf(<4 x float>, ptr noalias nonnull writeonly align 16, ptr noalias nonnull writeonly align 16) [[ATTRS]]

; CHECK: declare void @_ZGVnN4vl4l4_sincospif(<4 x float>, ptr noalias nonnull writeonly align 16, ptr noalias nonnull writeonly align 16) [[ATTRS]]

; CHECK: declare void @_ZGVsNxvl4l4_sincosf(<vscale x 4 x float>, ptr noalias nonnull writeonly align 16, ptr noalias nonnull writeonly align 16) [[ATTRS]]

; CHECK: declare void @_ZGVsNxvl4l4_sincospif(<vscale x 4 x float>, ptr noalias nonnull writeonly align 16, ptr noalias nonnull writeonly align 16) [[ATTRS]]

; CHECK: declare void @_ZGVsNxvl8l8_sincos(<vscale x 2 x double>, ptr noalias nonnull writeonly align 16, ptr noalias nonnull writeonly align 16) [[ATTRS]]

; CHECK: declare void @_ZGVsNxvl8l8_sincospi(<vscale x 2 x double>, ptr noalias nonnull writeonly align 16, ptr noalias nonnull writeonly align 16) [[ATTRS]]

; CHECK: attributes [[ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }
