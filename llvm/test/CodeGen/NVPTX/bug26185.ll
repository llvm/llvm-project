; RUN: llc < %s -march=nvptx -mcpu=sm_35 | FileCheck %s

; Verify that we correctly emit code for i8 ldg/ldu. We do not expose 8-bit
; registers in the backend, so these loads need special handling.

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-unknown-unknown"

; CHECK-LABEL: ex_zext
define void @ex_zext(i8* noalias readonly %data, i32* %res) {
entry:
; CHECK: ld.global.nc.u8
  %val = load i8, i8* %data
; CHECK: cvt.u32.u8
  %valext = zext i8 %val to i32
  store i32 %valext, i32* %res
  ret void
}

; CHECK-LABEL: ex_sext
define void @ex_sext(i8* noalias readonly %data, i32* %res) {
entry:
; CHECK: ld.global.nc.u8
  %val = load i8, i8* %data
; CHECK: cvt.s32.s8
  %valext = sext i8 %val to i32
  store i32 %valext, i32* %res
  ret void
}

; CHECK-LABEL: ex_zext_v2
define void @ex_zext_v2(<2 x i8>* noalias readonly %data, <2 x i32>* %res) {
entry:
; CHECK: ld.global.nc.v2.u8
  %val = load <2 x i8>, <2 x i8>* %data
; CHECK: cvt.u32.u16
  %valext = zext <2 x i8> %val to <2 x i32>
  store <2 x i32> %valext, <2 x i32>* %res
  ret void
}

; CHECK-LABEL: ex_sext_v2
define void @ex_sext_v2(<2 x i8>* noalias readonly %data, <2 x i32>* %res) {
entry:
; CHECK: ld.global.nc.v2.u8
  %val = load <2 x i8>, <2 x i8>* %data
; CHECK: cvt.s32.s8
  %valext = sext <2 x i8> %val to <2 x i32>
  store <2 x i32> %valext, <2 x i32>* %res
  ret void
}

!nvvm.annotations = !{!0,!1,!2,!3}
!0 = !{void (i8*, i32*)* @ex_zext, !"kernel", i32 1}
!1 = !{void (i8*, i32*)* @ex_sext, !"kernel", i32 1}
!2 = !{void (<2 x i8>*, <2 x i32>*)* @ex_zext_v2, !"kernel", i32 1}
!3 = !{void (<2 x i8>*, <2 x i32>*)* @ex_sext_v2, !"kernel", i32 1}
