; RUN: llc -mtriple=aarch64-linux-gnu -verify-machineinstrs -mattr=+sve -asm-verbose=0 < %s | FileCheck %s

;
; Unpredicated dup instruction (which is an alias for mov):
;   * register + register,
;   * register + immediate
;

define <vscale x 16 x i8> @dup_i8(i8 %b) {
; CHECK-LABEL: dup_i8:
; CHECK: mov z0.b, w0
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.dup.x.nxv16i8(i8 %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 16 x i8> @dup_imm_i8() {
; CHECK-LABEL: dup_imm_i8:
; CHECK: mov z0.b, #16
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.dup.x.nxv16i8(i8 16)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @dup_i16(i16 %b) {
; CHECK-LABEL: dup_i16:
; CHECK: mov z0.h, w0
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.dup.x.nxv8i16(i16 %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 8 x i16> @dup_imm_i16(i16 %b) {
; CHECK-LABEL: dup_imm_i16:
; CHECK: mov z0.h, #16
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.dup.x.nxv8i16(i16 16)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @dup_i32(i32 %b) {
; CHECK-LABEL: dup_i32:
; CHECK: mov z0.s, w0
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.dup.x.nxv4i32(i32 %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 4 x i32> @dup_imm_i32(i32 %b) {
; CHECK-LABEL: dup_imm_i32:
; CHECK: mov z0.s, #16
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.dup.x.nxv4i32(i32 16)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @dup_i64(i64 %b) {
; CHECK-LABEL: dup_i64:
; CHECK: mov z0.d, x0
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 %b)
  ret <vscale x 2 x i64> %out
}

define <vscale x 2 x i64> @dup_imm_i64(i64 %b) {
; CHECK-LABEL: dup_imm_i64:
; CHECK: mov z0.d, #16
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 16)
  ret <vscale x 2 x i64> %out
}

define <vscale x 8 x half> @dup_f16(half %b) {
; CHECK-LABEL: dup_f16:
; CHECK: mov z0.h, h0
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.dup.x.nxv8f16(half %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 8 x half> @dup_imm_f16(half %b) {
; CHECK-LABEL: dup_imm_f16:
; CHECK: mov z0.h, #16.00000000
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.dup.x.nxv8f16(half 16.)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @dup_f32(float %b) {
; CHECK-LABEL: dup_f32:
; CHECK: mov z0.s, s0
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.dup.x.nxv4f32(float %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 4 x float> @dup_imm_f32(float %b) {
; CHECK-LABEL: dup_imm_f32:
; CHECK: mov z0.s, #16.00000000
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.dup.x.nxv4f32(float 16.)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @dup_f64(double %b) {
; CHECK-LABEL: dup_f64:
; CHECK: mov z0.d, d0
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.dup.x.nxv2f64(double %b)
  ret <vscale x 2 x double> %out
}

define <vscale x 2 x double> @dup_imm_f64(double %b) {
; CHECK-LABEL: dup_imm_f64:
; CHECK: mov z0.d, #16.00000000
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.dup.x.nxv2f64(double 16.)
  ret <vscale x 2 x double> %out
}

declare <vscale x 16 x i8> @llvm.aarch64.sve.dup.x.nxv16i8( i8)
declare <vscale x 8 x i16> @llvm.aarch64.sve.dup.x.nxv8i16(i16)
declare <vscale x 4 x i32> @llvm.aarch64.sve.dup.x.nxv4i32(i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64)
declare <vscale x 8 x half> @llvm.aarch64.sve.dup.x.nxv8f16(half)
declare <vscale x 4 x float> @llvm.aarch64.sve.dup.x.nxv4f32(float)
declare <vscale x 2 x double> @llvm.aarch64.sve.dup.x.nxv2f64(double)
