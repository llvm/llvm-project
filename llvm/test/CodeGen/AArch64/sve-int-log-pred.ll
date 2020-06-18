; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; WARN-NOT: warning

define <vscale x 16 x i8> @and_pred_i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: and_pred_i8:
; CHECK: and z0.b, p0/m, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.and.nxv2i8(<vscale x 16 x i1> %pg,
                                                              <vscale x 16 x i8> %a,
                                                              <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @and_pred_i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: and_pred_i16:
; CHECK: and z0.h, p0/m, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.and.nxv2i16(<vscale x 8 x i1> %pg,
                                                               <vscale x 8 x i16> %a,
                                                               <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @and_pred_i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: and_pred_i32:
; CHECK: and z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.and.nxv2i32(<vscale x 4 x i1> %pg,
                                                               <vscale x 4 x i32> %a,
                                                               <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @and_pred_i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: and_pred_i64:
; CHECK: and z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.and.nxv2i64(<vscale x 2 x i1> %pg,
                                                               <vscale x 2 x i64> %a,
                                                               <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

define <vscale x 16 x i8> @or_pred_i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: or_pred_i8:
; CHECK: orr z0.b, p0/m, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.orr.nxv2i8(<vscale x 16 x i1> %pg,
                                                              <vscale x 16 x i8> %a,
                                                              <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @or_pred_i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: or_pred_i16:
; CHECK: orr z0.h, p0/m, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.orr.nxv2i16(<vscale x 8 x i1> %pg,
                                                               <vscale x 8 x i16> %a,
                                                               <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @or_pred_i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: or_pred_i32:
; CHECK: orr z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.orr.nxv2i32(<vscale x 4 x i1> %pg,
                                                               <vscale x 4 x i32> %a,
                                                               <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @or_pred_i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: or_pred_i64:
; CHECK: orr z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.orr.nxv2i64(<vscale x 2 x i1> %pg,
                                                               <vscale x 2 x i64> %a,
                                                               <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

define <vscale x 16 x i8> @xor_pred_i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: xor_pred_i8:
; CHECK: eor z0.b, p0/m, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.eor.nxv2i8(<vscale x 16 x i1> %pg,
                                                              <vscale x 16 x i8> %a,
                                                              <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @xor_pred_i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: xor_pred_i16:
; CHECK: eor z0.h, p0/m, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.eor.nxv2i16(<vscale x 8 x i1> %pg,
                                                               <vscale x 8 x i16> %a,
                                                               <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @xor_pred_i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: xor_pred_i32:
; CHECK: eor z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.eor.nxv2i32(<vscale x 4 x i1> %pg,
                                                               <vscale x 4 x i32> %a,
                                                               <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @xor_pred_i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: xor_pred_i64:
; CHECK: eor z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.eor.nxv2i64(<vscale x 2 x i1> %pg,
                                                               <vscale x 2 x i64> %a,
                                                               <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

define <vscale x 16 x i8> @bic_pred_i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: bic_pred_i8:
; CHECK: bic z0.b, p0/m, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.bic.nxv2i8(<vscale x 16 x i1> %pg,
                                                              <vscale x 16 x i8> %a,
                                                              <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @bic_pred_i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: bic_pred_i16:
; CHECK: bic z0.h, p0/m, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.bic.nxv2i16(<vscale x 8 x i1> %pg,
                                                               <vscale x 8 x i16> %a,
                                                               <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}


define <vscale x 4 x i32> @bic_pred_i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: bic_pred_i32:
; CHECK: bic z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.bic.nxv2i32(<vscale x 4 x i1> %pg,
                                                               <vscale x 4 x i32> %a,
                                                               <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @bic_pred_i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: bic_pred_i64:
; CHECK: bic z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.bic.nxv2i64(<vscale x 2 x i1> %pg,
                                                               <vscale x 2 x i64> %a,
                                                               <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

declare <vscale x 16 x i8> @llvm.aarch64.sve.and.nxv2i8(<vscale x 16 x i1>,<vscale x 16 x i8>,<vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.and.nxv2i16(<vscale x 8 x i1>,<vscale x 8 x i16>,<vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.and.nxv2i32(<vscale x 4 x i1>,<vscale x 4 x i32>,<vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.and.nxv2i64(<vscale x 2 x i1>,<vscale x 2 x i64>,<vscale x 2 x i64>)
declare <vscale x 16 x i8> @llvm.aarch64.sve.orr.nxv2i8(<vscale x 16 x i1>,<vscale x 16 x i8>,<vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.orr.nxv2i16(<vscale x 8 x i1>,<vscale x 8 x i16>,<vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.orr.nxv2i32(<vscale x 4 x i1>,<vscale x 4 x i32>,<vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.orr.nxv2i64(<vscale x 2 x i1>,<vscale x 2 x i64>,<vscale x 2 x i64>)
declare <vscale x 16 x i8> @llvm.aarch64.sve.eor.nxv2i8(<vscale x 16 x i1>,<vscale x 16 x i8>,<vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.eor.nxv2i16(<vscale x 8 x i1>,<vscale x 8 x i16>,<vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.eor.nxv2i32(<vscale x 4 x i1>,<vscale x 4 x i32>,<vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.eor.nxv2i64(<vscale x 2 x i1>,<vscale x 2 x i64>,<vscale x 2 x i64>)
declare <vscale x 16 x i8> @llvm.aarch64.sve.bic.nxv2i8(<vscale x 16 x i1>,<vscale x 16 x i8>,<vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.bic.nxv2i16(<vscale x 8 x i1>,<vscale x 8 x i16>,<vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.bic.nxv2i32(<vscale x 4 x i1>,<vscale x 4 x i32>,<vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.bic.nxv2i64(<vscale x 2 x i1>,<vscale x 2 x i64>,<vscale x 2 x i64>)
