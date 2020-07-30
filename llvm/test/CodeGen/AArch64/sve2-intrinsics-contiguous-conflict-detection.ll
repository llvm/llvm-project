; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve2 -asm-verbose=0 < %s | FileCheck %s

;
; WHILERW
;

define <vscale x 16 x i1> @whilerw_i8(i8* %a, i8* %b) {
; CHECK-LABEL: whilerw_i8:
; CHECK: whilerw  p0.b, x0, x1
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.whilerw.b.nx16i1(i8* %a, i8* %b)
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @whilerw_i16(i16* %a, i16* %b) {
; CHECK-LABEL: whilerw_i16:
; CHECK: whilerw  p0.h, x0, x1
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.whilerw.h.nx8i1(i16* %a, i16* %b)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @whilerw_i32(i32* %a, i32* %b) {
; CHECK-LABEL: whilerw_i32:
; CHECK: whilerw  p0.s, x0, x1
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.whilerw.s.nx4i1(i32* %a, i32* %b)
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @whilerw_i64(i64* %a, i64* %b) {
; CHECK-LABEL: whilerw_i64:
; CHECK: whilerw  p0.d, x0, x1
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i1> @llvm.aarch64.sve.whilerw.d.nx2i1(i64* %a, i64* %b)
  ret <vscale x 2 x i1> %out
}

define <vscale x 8 x i1> @whilerw_bfloat(bfloat* %a, bfloat* %b) {
; CHECK-LABEL: whilerw_bfloat:
; CHECK: whilerw  p0.h, x0, x1
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.whilerw.h.nx8i1.bf16.bf16(bfloat* %a, bfloat* %b)
  ret <vscale x 8 x i1> %out
}

define <vscale x 8 x i1> @whilerw_half(half* %a, half* %b) {
; CHECK-LABEL: whilerw_half:
; CHECK: whilerw  p0.h, x0, x1
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.whilerw.h.nx8i1.f16.f16(half* %a, half* %b)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @whilerw_float(float* %a, float* %b) {
; CHECK-LABEL: whilerw_float:
; CHECK: whilerw  p0.s, x0, x1
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.whilerw.s.nx4i1.f32.f32(float* %a, float* %b)
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @whilerw_double(double* %a, double* %b) {
; CHECK-LABEL: whilerw_double:
; CHECK: whilerw  p0.d, x0, x1
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i1> @llvm.aarch64.sve.whilerw.d.nx2i1.f64.f64(double* %a, double* %b)
  ret <vscale x 2 x i1> %out
}

;
; WHILEWR
;

define <vscale x 16 x i1> @whilewr_i8(i8* %a, i8* %b) {
; CHECK-LABEL: whilewr_i8:
; CHECK: whilewr  p0.b, x0, x1
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.whilewr.b.nx16i1(i8* %a, i8* %b)
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @whilewr_i16(i16* %a, i16* %b) {
; CHECK-LABEL: whilewr_i16:
; CHECK: whilewr  p0.h, x0, x1
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.whilewr.h.nx8i1(i16* %a, i16* %b)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @whilewr_i32(i32* %a, i32* %b) {
; CHECK-LABEL: whilewr_i32:
; CHECK: whilewr  p0.s, x0, x1
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.whilewr.s.nx4i1(i32* %a, i32* %b)
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @whilewr_i64(i64* %a, i64* %b) {
; CHECK-LABEL: whilewr_i64:
; CHECK: whilewr  p0.d, x0, x1
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i1> @llvm.aarch64.sve.whilewr.d.nx2i1(i64* %a, i64* %b)
  ret <vscale x 2 x i1> %out
}

define <vscale x 8 x i1> @whilewr_bfloat(bfloat* %a, bfloat* %b) {
; CHECK-LABEL: whilewr_bfloat:
; CHECK: whilewr  p0.h, x0, x1
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.whilewr.h.nx8i1.bf16.bf16(bfloat* %a, bfloat* %b)
  ret <vscale x 8 x i1> %out
}

define <vscale x 8 x i1> @whilewr_half(half* %a, half* %b) {
; CHECK-LABEL: whilewr_half:
; CHECK: whilewr  p0.h, x0, x1
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.whilewr.h.nx8i1.f16.f16(half* %a, half* %b)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @whilewr_float(float* %a, float* %b) {
; CHECK-LABEL: whilewr_float:
; CHECK: whilewr  p0.s, x0, x1
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.whilewr.s.nx4i1.f32.f32(float* %a, float* %b)
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @whilewr_double(double* %a, double* %b) {
; CHECK-LABEL: whilewr_double:
; CHECK: whilewr  p0.d, x0, x1
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i1> @llvm.aarch64.sve.whilewr.d.nx2i1.f64.f64(double* %a, double* %b)
  ret <vscale x 2 x i1> %out
}

declare <vscale x 16 x i1> @llvm.aarch64.sve.whilerw.b.nx16i1(i8* %a, i8* %b)
declare <vscale x 8 x i1> @llvm.aarch64.sve.whilerw.h.nx8i1(i16* %a, i16* %b)
declare <vscale x 4 x i1> @llvm.aarch64.sve.whilerw.s.nx4i1(i32* %a, i32* %b)
declare <vscale x 2 x i1> @llvm.aarch64.sve.whilerw.d.nx2i1(i64* %a, i64* %b)

declare <vscale x 8 x i1> @llvm.aarch64.sve.whilerw.h.nx8i1.bf16.bf16(bfloat* %a, bfloat* %b)
declare <vscale x 8 x i1> @llvm.aarch64.sve.whilerw.h.nx8i1.f16.f16(half* %a, half* %b)
declare <vscale x 4 x i1> @llvm.aarch64.sve.whilerw.s.nx4i1.f32.f32(float* %a, float* %b)
declare <vscale x 2 x i1> @llvm.aarch64.sve.whilerw.d.nx2i1.f64.f64(double* %a, double* %b)

declare <vscale x 16 x i1> @llvm.aarch64.sve.whilewr.b.nx16i1(i8* %a, i8* %b)
declare <vscale x 8 x i1> @llvm.aarch64.sve.whilewr.h.nx8i1(i16* %a, i16* %b)
declare <vscale x 4 x i1> @llvm.aarch64.sve.whilewr.s.nx4i1(i32* %a, i32* %b)
declare <vscale x 2 x i1> @llvm.aarch64.sve.whilewr.d.nx2i1(i64* %a, i64* %b)

declare <vscale x 8 x i1> @llvm.aarch64.sve.whilewr.h.nx8i1.bf16.bf16(bfloat* %a, bfloat* %b)
declare <vscale x 8 x i1> @llvm.aarch64.sve.whilewr.h.nx8i1.f16.f16(half* %a, half* %b)
declare <vscale x 4 x i1> @llvm.aarch64.sve.whilewr.s.nx4i1.f32.f32(float* %a, float* %b)
declare <vscale x 2 x i1> @llvm.aarch64.sve.whilewr.d.nx2i1.f64.f64(double* %a, double* %b)
