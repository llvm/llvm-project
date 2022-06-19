; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve2 < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sme < %s | FileCheck %s

; ADDHNB

define <vscale x 16 x i8> @addhnb_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: addhnb_h:
; CHECK: addhnb z0.b, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.addhnb.nxv8i16(<vscale x 8 x i16> %a,
                                                                  <vscale x 8 x i16> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @addhnb_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: addhnb_s:
; CHECK: addhnb z0.h, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.addhnb.nxv4i32(<vscale x 4 x i32> %a,
                                                                  <vscale x 4 x i32> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @addhnb_d(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: addhnb_d:
; CHECK: addhnb z0.s, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.addhnb.nxv2i64(<vscale x 2 x i64> %a,
                                                                  <vscale x 2 x i64> %b)
  ret <vscale x 4 x i32> %out
}

; ADDHNT

define <vscale x 16 x i8> @addhnt_h(<vscale x 16 x i8> %a, <vscale x 8 x i16> %b, <vscale x 8 x i16> %c) {
; CHECK-LABEL: addhnt_h:
; CHECK: addhnt z0.b, z1.h, z2.h
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.addhnt.nxv8i16(<vscale x 16 x i8> %a,
                                                                  <vscale x 8 x i16> %b,
                                                                  <vscale x 8 x i16> %c)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @addhnt_s(<vscale x 8 x i16> %a, <vscale x 4 x i32> %b, <vscale x 4 x i32> %c) {
; CHECK-LABEL: addhnt_s:
; CHECK: addhnt z0.h, z1.s, z2.s
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.addhnt.nxv4i32(<vscale x 8 x i16> %a,
                                                                  <vscale x 4 x i32> %b,
                                                                  <vscale x 4 x i32> %c)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @addhnt_d(<vscale x 4 x i32> %a, <vscale x 2 x i64> %b, <vscale x 2 x i64> %c) {
; CHECK-LABEL: addhnt_d:
; CHECK: addhnt z0.s, z1.d, z2.d
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.addhnt.nxv2i64(<vscale x 4 x i32> %a,
                                                                  <vscale x 2 x i64> %b,
                                                                  <vscale x 2 x i64> %c)
  ret <vscale x 4 x i32> %out
}

; RADDHNB

define <vscale x 16 x i8> @raddhnb_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: raddhnb_h:
; CHECK: raddhnb z0.b, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.raddhnb.nxv8i16(<vscale x 8 x i16> %a,
                                                                   <vscale x 8 x i16> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @raddhnb_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: raddhnb_s:
; CHECK: raddhnb z0.h, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.raddhnb.nxv4i32(<vscale x 4 x i32> %a,
                                                                   <vscale x 4 x i32> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @raddhnb_d(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: raddhnb_d:
; CHECK: raddhnb z0.s, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.raddhnb.nxv2i64(<vscale x 2 x i64> %a,
                                                                   <vscale x 2 x i64> %b)
  ret <vscale x 4 x i32> %out
}

; RADDHNT

define <vscale x 16 x i8> @raddhnt_h(<vscale x 16 x i8> %a, <vscale x 8 x i16> %b, <vscale x 8 x i16> %c) {
; CHECK-LABEL: raddhnt_h:
; CHECK: raddhnt z0.b, z1.h, z2.h
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.raddhnt.nxv8i16(<vscale x 16 x i8> %a,
                                                                   <vscale x 8 x i16> %b,
                                                                   <vscale x 8 x i16> %c)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @raddhnt_s(<vscale x 8 x i16> %a, <vscale x 4 x i32> %b, <vscale x 4 x i32> %c) {
; CHECK-LABEL: raddhnt_s:
; CHECK: raddhnt z0.h, z1.s, z2.s
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.raddhnt.nxv4i32(<vscale x 8 x i16> %a,
                                                                   <vscale x 4 x i32> %b,
                                                                   <vscale x 4 x i32> %c)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @raddhnt_d(<vscale x 4 x i32> %a, <vscale x 2 x i64> %b, <vscale x 2 x i64> %c) {
; CHECK-LABEL: raddhnt_d:
; CHECK: raddhnt z0.s, z1.d, z2.d
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.raddhnt.nxv2i64(<vscale x 4 x i32> %a,
                                                                   <vscale x 2 x i64> %b,
                                                                   <vscale x 2 x i64> %c)
  ret <vscale x 4 x i32> %out
}

; RSUBHNB

define <vscale x 16 x i8> @rsubhnb_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: rsubhnb_h:
; CHECK: rsubhnb z0.b, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.rsubhnb.nxv8i16(<vscale x 8 x i16> %a,
                                                                   <vscale x 8 x i16> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @rsubhnb_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: rsubhnb_s:
; CHECK: rsubhnb z0.h, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.rsubhnb.nxv4i32(<vscale x 4 x i32> %a,
                                                                   <vscale x 4 x i32> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @rsubhnb_d(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: rsubhnb_d:
; CHECK: rsubhnb z0.s, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.rsubhnb.nxv2i64(<vscale x 2 x i64> %a,
                                                                   <vscale x 2 x i64> %b)
  ret <vscale x 4 x i32> %out
}

; RSUBHNT

define <vscale x 16 x i8> @rsubhnt_h(<vscale x 16 x i8> %a, <vscale x 8 x i16> %b, <vscale x 8 x i16> %c) {
; CHECK-LABEL: rsubhnt_h:
; CHECK: rsubhnt z0.b, z1.h, z2.h
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.rsubhnt.nxv8i16(<vscale x 16 x i8> %a,
                                                                   <vscale x 8 x i16> %b,
                                                                   <vscale x 8 x i16> %c)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @rsubhnt_s(<vscale x 8 x i16> %a, <vscale x 4 x i32> %b, <vscale x 4 x i32> %c) {
; CHECK-LABEL: rsubhnt_s:
; CHECK: rsubhnt z0.h, z1.s, z2.s
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.rsubhnt.nxv4i32(<vscale x 8 x i16> %a,
                                                                   <vscale x 4 x i32> %b,
                                                                   <vscale x 4 x i32> %c)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @rsubhnt_d(<vscale x 4 x i32> %a, <vscale x 2 x i64> %b, <vscale x 2 x i64> %c) {
; CHECK-LABEL: rsubhnt_d:
; CHECK: rsubhnt z0.s, z1.d, z2.d
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.rsubhnt.nxv2i64(<vscale x 4 x i32> %a,
                                                                   <vscale x 2 x i64> %b,
                                                                   <vscale x 2 x i64> %c)
  ret <vscale x 4 x i32> %out
}

; SUBHNB

define <vscale x 16 x i8> @subhnb_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: subhnb_h:
; CHECK: subhnb z0.b, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.subhnb.nxv8i16(<vscale x 8 x i16> %a,
                                                                  <vscale x 8 x i16> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @subhnb_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: subhnb_s:
; CHECK: subhnb z0.h, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.subhnb.nxv4i32(<vscale x 4 x i32> %a,
                                                                  <vscale x 4 x i32> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @subhnb_d(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: subhnb_d:
; CHECK: subhnb z0.s, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.subhnb.nxv2i64(<vscale x 2 x i64> %a,
                                                                  <vscale x 2 x i64> %b)
  ret <vscale x 4 x i32> %out
}

; SUBHNT

define <vscale x 16 x i8> @subhnt_h(<vscale x 16 x i8> %a, <vscale x 8 x i16> %b, <vscale x 8 x i16> %c) {
; CHECK-LABEL: subhnt_h:
; CHECK: subhnt z0.b, z1.h, z2.h
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.subhnt.nxv8i16(<vscale x 16 x i8> %a,
                                                                  <vscale x 8 x i16> %b,
                                                                  <vscale x 8 x i16> %c)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @subhnt_s(<vscale x 8 x i16> %a, <vscale x 4 x i32> %b, <vscale x 4 x i32> %c) {
; CHECK-LABEL: subhnt_s:
; CHECK: subhnt z0.h, z1.s, z2.s
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.subhnt.nxv4i32(<vscale x 8 x i16> %a,
                                                                  <vscale x 4 x i32> %b,
                                                                  <vscale x 4 x i32> %c)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @subhnt_d(<vscale x 4 x i32> %a, <vscale x 2 x i64> %b, <vscale x 2 x i64> %c) {
; CHECK-LABEL: subhnt_d:
; CHECK: subhnt z0.s, z1.d, z2.d
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.subhnt.nxv2i64(<vscale x 4 x i32> %a,
                                                                  <vscale x 2 x i64> %b,
                                                                  <vscale x 2 x i64> %c)
  ret <vscale x 4 x i32> %out
}


declare <vscale x 16 x i8> @llvm.aarch64.sve.addhnb.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.addhnb.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.addhnb.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.addhnt.nxv8i16(<vscale x 16 x i8>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.addhnt.nxv4i32(<vscale x 8 x i16>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.addhnt.nxv2i64(<vscale x 4 x i32>, <vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.raddhnb.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.raddhnb.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.raddhnb.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.raddhnt.nxv8i16(<vscale x 16 x i8>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.raddhnt.nxv4i32(<vscale x 8 x i16>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.raddhnt.nxv2i64(<vscale x 4 x i32>, <vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.subhnb.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.subhnb.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.subhnb.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.subhnt.nxv8i16(<vscale x 16 x i8>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.subhnt.nxv4i32(<vscale x 8 x i16>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.subhnt.nxv2i64(<vscale x 4 x i32>, <vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.rsubhnb.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.rsubhnb.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.rsubhnb.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.rsubhnt.nxv8i16(<vscale x 16 x i8>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.rsubhnt.nxv4i32(<vscale x 8 x i16>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.rsubhnt.nxv2i64(<vscale x 4 x i32>, <vscale x 2 x i64>, <vscale x 2 x i64>)
