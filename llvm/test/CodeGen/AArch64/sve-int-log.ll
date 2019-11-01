; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s | FileCheck %s

define <vscale x 2 x i64> @and_d(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: and_d
; CHECK: and z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %res = and <vscale x 2 x i64> %a, %b
  ret <vscale x 2 x i64> %res
}

define <vscale x 4 x i32> @and_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: and_s
; CHECK: and z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %res = and <vscale x 4 x i32> %a, %b
  ret <vscale x 4 x i32> %res
}

define <vscale x 8 x i16> @and_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: and_h
; CHECK: and z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %res = and <vscale x 8 x i16> %a, %b
  ret <vscale x 8 x i16> %res
}

define <vscale x 16 x i8> @and_b(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: and_b
; CHECK: and z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %res = and <vscale x 16 x i8> %a, %b
  ret <vscale x 16 x i8> %res
}                                                                                          
define <vscale x 2 x i64> @or_d(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: or_d
; CHECK: orr z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %res = or <vscale x 2 x i64> %a, %b
  ret <vscale x 2 x i64> %res
}

define <vscale x 4 x i32> @or_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: or_s
; CHECK: orr z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %res = or <vscale x 4 x i32> %a, %b
  ret <vscale x 4 x i32> %res
}

define <vscale x 8 x i16> @or_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: or_h
; CHECK: orr z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %res = or <vscale x 8 x i16> %a, %b
  ret <vscale x 8 x i16> %res
}

define <vscale x 16 x i8> @or_b(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: or_b
; CHECK: orr z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %res = or <vscale x 16 x i8> %a, %b
  ret <vscale x 16 x i8> %res
}                                                                                          

define <vscale x 2 x i64> @xor_d(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: xor_d
; CHECK: eor z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %res = xor <vscale x 2 x i64> %a, %b
  ret <vscale x 2 x i64> %res
}

define <vscale x 4 x i32> @xor_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: xor_s
; CHECK: eor z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %res = xor <vscale x 4 x i32> %a, %b
  ret <vscale x 4 x i32> %res
}

define <vscale x 8 x i16> @xor_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: xor_h
; CHECK: eor z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %res = xor <vscale x 8 x i16> %a, %b
  ret <vscale x 8 x i16> %res
}

define <vscale x 16 x i8> @xor_b(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: xor_b
; CHECK: eor z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %res = xor <vscale x 16 x i8> %a, %b
  ret <vscale x 16 x i8> %res
}

define <vscale x 2 x i64> @bic_d(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: bic_d
; CHECK: bic z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %res =  call <vscale x 2 x i64> @llvm.aarch64.sve.bic.nxv2i64(<vscale x 2 x i64> %a,
                                                                <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %res
}

define <vscale x 4 x i32> @bic_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: bic_s
; CHECK: bic z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %res =  call <vscale x 4 x i32> @llvm.aarch64.sve.bic.nxv4i32(<vscale x 4 x i32> %a,
                                                                <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %res
}

define <vscale x 8 x i16> @bic_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: bic_h
; CHECK: bic z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %res =  call <vscale x 8 x i16> @llvm.aarch64.sve.bic.nxv8i16(<vscale x 8 x i16> %a,
                                                                <vscale x 8 x i16> %b)

  ret <vscale x 8 x i16> %res
}

define <vscale x 16 x i8> @bic_b(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: bic_b
; CHECK: bic z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %res =  call <vscale x 16 x i8> @llvm.aarch64.sve.bic.nxv16i8(<vscale x 16 x i8> %a,
                                                                <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %res
}

declare <vscale x 2 x i64> @llvm.aarch64.sve.bic.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.bic.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.bic.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 16 x i8> @llvm.aarch64.sve.bic.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>)
