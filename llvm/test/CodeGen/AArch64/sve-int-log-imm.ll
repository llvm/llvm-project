; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s | FileCheck %s

define <vscale x 16 x i8> @orr_i8(<vscale x 16 x i8> %a) {
; CHECK-LABEL: orr_i8:
; CHECK: orr z0.b, z0.b, #0xf
; CHECK-NEXT: ret
  %res = call <vscale x 16 x i8> @llvm.aarch64.sve.orr.imm.nxv16i8(<vscale x 16 x i8> %a, 
                                                                   i64 15)
  ret <vscale x 16 x i8> %res
}

define <vscale x 8 x i16> @orr_i16(<vscale x 8 x i16> %a) {
; CHECK-LABEL: orr_i16:
; CHECK: orr z0.h, z0.h, #0xfc07
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i16> @llvm.aarch64.sve.orr.imm.nxv8i16(<vscale x 8 x i16> %a, 
                                                                   i64 64519)
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @orr_i32(<vscale x 4 x i32> %a) {
; CHECK-LABEL: orr_i32:
; CHECK: orr z0.s, z0.s, #0xffff00
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.orr.imm.nxv4i32(<vscale x 4 x i32> %a, 
                                                                   i64 16776960)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @orr_i64(<vscale x 2 x i64> %a) {
; CHECK-LABEL: orr_i64:
; CHECK: orr z0.d, z0.d, #0xfffc000000000000
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.orr.imm.nxv2i64(<vscale x 2 x i64> %a, 
                                                                   i64 18445618173802708992)
  ret <vscale x 2 x i64> %res
}

define <vscale x 16 x i8> @eor_i8(<vscale x 16 x i8> %a) {
; CHECK-LABEL: eor_i8:
; CHECK: eor z0.b, z0.b, #0xf
; CHECK-NEXT: ret
  %res = call <vscale x 16 x i8> @llvm.aarch64.sve.eor.imm.nxv16i8(<vscale x 16 x i8> %a, 
                                                                   i64 15)
  ret <vscale x 16 x i8> %res
}

define <vscale x 8 x i16> @eor_i16(<vscale x 8 x i16> %a) {
; CHECK-LABEL: eor_i16:
; CHECK: eor z0.h, z0.h, #0xfc07
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i16> @llvm.aarch64.sve.eor.imm.nxv8i16(<vscale x 8 x i16> %a, 
                                                                   i64 64519)
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @eor_i32(<vscale x 4 x i32> %a) {
; CHECK-LABEL: eor_i32:
; CHECK: eor z0.s, z0.s, #0xffff00
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.eor.imm.nxv4i32(<vscale x 4 x i32> %a, 
                                                                   i64 16776960)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @eor_i64(<vscale x 2 x i64> %a) {
; CHECK-LABEL: eor_i64:
; CHECK: eor z0.d, z0.d, #0xfffc000000000000
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.eor.imm.nxv2i64(<vscale x 2 x i64> %a, 
                                                                   i64 18445618173802708992)
  ret <vscale x 2 x i64> %res
}

define <vscale x 16 x i8> @and_i8(<vscale x 16 x i8> %a) {
; CHECK-LABEL: and_i8:
; CHECK: and z0.b, z0.b, #0xf
; CHECK-NEXT: ret
  %res = call <vscale x 16 x i8> @llvm.aarch64.sve.and.imm.nxv16i8(<vscale x 16 x i8> %a, 
                                                                   i64 15)
  ret <vscale x 16 x i8> %res
}

define <vscale x 8 x i16> @and_i16(<vscale x 8 x i16> %a) {
; CHECK-LABEL: and_i16:
; CHECK: and z0.h, z0.h, #0xfc07
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i16> @llvm.aarch64.sve.and.imm.nxv8i16(<vscale x 8 x i16> %a, 
                                                                   i64 64519)
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @and_i32(<vscale x 4 x i32> %a) {
; CHECK-LABEL: and_i32:
; CHECK: and z0.s, z0.s, #0xffff00
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.and.imm.nxv4i32(<vscale x 4 x i32> %a, 
                                                                   i64 16776960)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @and_i64(<vscale x 2 x i64> %a) {
; CHECK-LABEL: and_i64:
; CHECK: and z0.d, z0.d, #0xfffc000000000000
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.and.imm.nxv2i64(<vscale x 2 x i64> %a,
                                                                   i64 18445618173802708992)
  ret <vscale x 2 x i64> %res
}

declare <vscale x 16 x i8> @llvm.aarch64.sve.orr.imm.nxv16i8(<vscale x 16 x i8>, i64)
declare <vscale x 8 x i16> @llvm.aarch64.sve.orr.imm.nxv8i16(<vscale x 8 x i16>, i64)
declare <vscale x 4 x i32> @llvm.aarch64.sve.orr.imm.nxv4i32(<vscale x 4 x i32>, i64)
declare <vscale x 2 x i64> @llvm.aarch64.sve.orr.imm.nxv2i64(<vscale x 2 x i64>, i64)
declare <vscale x 16 x i8> @llvm.aarch64.sve.eor.imm.nxv16i8(<vscale x 16 x i8>, i64)
declare <vscale x 8 x i16> @llvm.aarch64.sve.eor.imm.nxv8i16(<vscale x 8 x i16>, i64)
declare <vscale x 4 x i32> @llvm.aarch64.sve.eor.imm.nxv4i32(<vscale x 4 x i32>, i64)
declare <vscale x 2 x i64> @llvm.aarch64.sve.eor.imm.nxv2i64(<vscale x 2 x i64>, i64)
declare <vscale x 16 x i8> @llvm.aarch64.sve.and.imm.nxv16i8(<vscale x 16 x i8>, i64)
declare <vscale x 8 x i16> @llvm.aarch64.sve.and.imm.nxv8i16(<vscale x 8 x i16>, i64)
declare <vscale x 4 x i32> @llvm.aarch64.sve.and.imm.nxv4i32(<vscale x 4 x i32>, i64)
declare <vscale x 2 x i64> @llvm.aarch64.sve.and.imm.nxv2i64(<vscale x 2 x i64>, i64)
