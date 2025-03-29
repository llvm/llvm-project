; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_35 -verify-machineinstrs | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_35 | %ptxas-verify %}

; Verify that we correctly emit code for i8 ldg/ldu. We do not expose 8-bit
; registers in the backend, so these loads need special handling.

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-unknown-unknown"

; CHECK-LABEL: ex_zext
define ptx_kernel void @ex_zext(ptr noalias readonly %data, ptr %res) {
entry:
; CHECK: ld.global.nc.u8
  %val = load i8, ptr %data
; CHECK: cvt.u32.u8
  %valext = zext i8 %val to i32
  store i32 %valext, ptr %res
  ret void
}

; CHECK-LABEL: ex_sext
define ptx_kernel void @ex_sext(ptr noalias readonly %data, ptr %res) {
entry:
; CHECK: ld.global.nc.u8
  %val = load i8, ptr %data
; CHECK: cvt.s32.s8
  %valext = sext i8 %val to i32
  store i32 %valext, ptr %res
  ret void
}

; CHECK-LABEL: ex_zext_v2
define ptx_kernel void @ex_zext_v2(ptr noalias readonly %data, ptr %res) {
entry:
; CHECK: ld.global.nc.v2.u8
  %val = load <2 x i8>, ptr %data
; CHECK: cvt.u32.u16
  %valext = zext <2 x i8> %val to <2 x i32>
  store <2 x i32> %valext, ptr %res
  ret void
}

; CHECK-LABEL: ex_sext_v2
define ptx_kernel void @ex_sext_v2(ptr noalias readonly %data, ptr %res) {
entry:
; CHECK: ld.global.nc.v2.u8
  %val = load <2 x i8>, ptr %data
; CHECK: cvt.s32.s8
  %valext = sext <2 x i8> %val to <2 x i32>
  store <2 x i32> %valext, ptr %res
  ret void
}

