; REQUIRES: x86-registered-target
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=x86_64-unknown-linux -vector-library=AMDLIBM < %s | FileCheck %s

; CHECK: declare void @amd_vrd2_sincos(<2 x double>, ptr noalias nonnull writeonly align 16, ptr noalias nonnull writeonly align 16) [[ATTRS_PTR_ARG:#[0-9]+]]
; CHECK: declare void @amd_vrd4_sincos(<4 x double>, ptr noalias nonnull writeonly align 32, ptr noalias nonnull writeonly align 32) [[ATTRS_PTR_ARG]]
; CHECK: declare void @amd_vrd8_sincos(<8 x double>, ptr noalias nonnull writeonly align 64, ptr noalias nonnull writeonly align 64) [[ATTRS_PTR_ARG]]
; CHECK: declare void @amd_vrs16_sincosf(<16 x float>, ptr noalias nonnull writeonly align 64, ptr noalias nonnull writeonly align 64) [[ATTRS_PTR_ARG]]
; CHECK: declare void @amd_vrs4_sincosf(<4 x float>, ptr noalias nonnull writeonly align 16, ptr noalias nonnull writeonly align 16) [[ATTRS_PTR_ARG]]
; CHECK: declare void @amd_vrs8_sincosf(<8 x float>, ptr noalias nonnull writeonly align 32, ptr noalias nonnull writeonly align 32) [[ATTRS_PTR_ARG]]

; CHECK: attributes [[ATTRS_PTR_ARG]] = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }
