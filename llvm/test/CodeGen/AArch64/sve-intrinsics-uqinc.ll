; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve -asm-verbose=0 < %s | FileCheck %s

; Since UQDEC{B|H|W|D|P} and UQINC{B|H|W|D|P} have identical semantics, the tests for
;   * @llvm.aarch64.sve.uqinc{b|h|w|d|p}, and
;   * @llvm.aarch64.sve.uqdec{b|h|w|d|p}
; should also be identical (with the instruction name being adjusted). When
; updating this file remember to make similar changes in the file testing the
; other intrinsic.

;
; UQINCH (vector)
;

define <vscale x 8 x i16> @uqinch(<vscale x 8 x i16> %a) {
; CHECK-LABEL: uqinch:
; CHECK: uqinch z0.h, pow2
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.uqinch.nxv8i16(<vscale x 8 x i16> %a,
                                                                  i32 0, i32 1)
  ret <vscale x 8 x i16> %out
}

;
; UQINCW (vector)
;

define <vscale x 4 x i32> @uqincw(<vscale x 4 x i32> %a) {
; CHECK-LABEL: uqincw:
; CHECK: uqincw z0.s, vl1, mul #2
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.uqincw.nxv4i32(<vscale x 4 x i32> %a,
                                                                  i32 1, i32 2)
  ret <vscale x 4 x i32> %out
}

;
; UQINCD (vector)
;

define <vscale x 2 x i64> @uqincd(<vscale x 2 x i64> %a) {
; CHECK-LABEL: uqincd:
; CHECK: uqincd z0.d, vl2, mul #3
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.uqincd.nxv2i64(<vscale x 2 x i64> %a,
                                                                  i32 2, i32 3)
  ret <vscale x 2 x i64> %out
}

;
; UQINCP (vector)
;

define <vscale x 8 x i16> @uqincp_b16(<vscale x 8 x i16> %a, <vscale x 8 x i1> %b) {
; CHECK-LABEL: uqincp_b16:
; CHECK: uqincp z0.h, p0
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.uqincp.nxv8i16(<vscale x 8 x i16> %a,
                                                                  <vscale x 8 x i1> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @uqincp_b32(<vscale x 4 x i32> %a, <vscale x 4 x i1> %b) {
; CHECK-LABEL: uqincp_b32:
; CHECK: uqincp z0.s, p0
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.uqincp.nxv4i32(<vscale x 4 x i32> %a,
                                                                  <vscale x 4 x i1> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @uqincp_b64(<vscale x 2 x i64> %a, <vscale x 2 x i1> %b) {
; CHECK-LABEL: uqincp_b64:
; CHECK: uqincp z0.d, p0
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.uqincp.nxv2i64(<vscale x 2 x i64> %a,
                                                                  <vscale x 2 x i1> %b)
  ret <vscale x 2 x i64> %out
}

;
; UQINCB (scalar)
;

define i32 @uqincb_n32(i32 %a) {
; CHECK-LABEL: uqincb_n32:
; CHECK: uqincb w0, vl3, mul #4
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.uqincb.n32(i32 %a, i32 3, i32 4)
  ret i32 %out
}

define i64 @uqincb_n64(i64 %a) {
; CHECK-LABEL: uqincb_n64:
; CHECK: uqincb x0, vl4, mul #5
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.uqincb.n64(i64 %a, i32 4, i32 5)
  ret i64 %out
}

;
; UQINCH (scalar)
;

define i32 @uqinch_n32(i32 %a) {
; CHECK-LABEL: uqinch_n32:
; CHECK: uqinch w0, vl5, mul #6
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.uqinch.n32(i32 %a, i32 5, i32 6)
  ret i32 %out
}

define i64 @uqinch_n64(i64 %a) {
; CHECK-LABEL: uqinch_n64:
; CHECK: uqinch x0, vl6, mul #7
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.uqinch.n64(i64 %a, i32 6, i32 7)
  ret i64 %out
}

;
; UQINCW (scalar)
;

define i32 @uqincw_n32(i32 %a) {
; CHECK-LABEL: uqincw_n32:
; CHECK: uqincw w0, vl7, mul #8
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.uqincw.n32(i32 %a, i32 7, i32 8)
  ret i32 %out
}

define i64 @uqincw_n64(i64 %a) {
; CHECK-LABEL: uqincw_n64:
; CHECK: uqincw x0, vl8, mul #9
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.uqincw.n64(i64 %a, i32 8, i32 9)
  ret i64 %out
}

;
; UQINCD (scalar)
;

define i32 @uqincd_n32(i32 %a) {
; CHECK-LABEL: uqincd_n32:
; CHECK: uqincd w0, vl16, mul #10
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.uqincd.n32(i32 %a, i32 9, i32 10)
  ret i32 %out
}

define i64 @uqincd_n64(i64 %a) {
; CHECK-LABEL: uqincd_n64:
; CHECK: uqincd x0, vl32, mul #11
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.uqincd.n64(i64 %a, i32 10, i32 11)
  ret i64 %out
}

;
; UQINCP (scalar)
;

define i32 @uqincp_n32_b8(i32 %a, <vscale x 16 x i1> %b) {
; CHECK-LABEL: uqincp_n32_b8:
; CHECK: uqincp w0, p0.b
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.uqincp.n32.nxv16i1(i32 %a, <vscale x 16 x i1> %b)
  ret i32 %out
}

define i32 @uqincp_n32_b16(i32 %a, <vscale x 8 x i1> %b) {
; CHECK-LABEL: uqincp_n32_b16:
; CHECK: uqincp w0, p0.h
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.uqincp.n32.nxv8i1(i32 %a, <vscale x 8 x i1> %b)
  ret i32 %out
}

define i32 @uqincp_n32_b32(i32 %a, <vscale x 4 x i1> %b) {
; CHECK-LABEL: uqincp_n32_b32:
; CHECK: uqincp w0, p0.s
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.uqincp.n32.nxv4i1(i32 %a, <vscale x 4 x i1> %b)
  ret i32 %out
}

define i32 @uqincp_n32_b64(i32 %a, <vscale x 2 x i1> %b) {
; CHECK-LABEL: uqincp_n32_b64:
; CHECK: uqincp w0, p0.d
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.uqincp.n32.nxv2i1(i32 %a, <vscale x 2 x i1> %b)
  ret i32 %out
}

define i64 @uqincp_n64_b8(i64 %a, <vscale x 16 x i1> %b) {
; CHECK-LABEL: uqincp_n64_b8:
; CHECK: uqincp x0, p0.b
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.uqincp.n64.nxv16i1(i64 %a, <vscale x 16 x i1> %b)
  ret i64 %out
}

define i64 @uqincp_n64_b16(i64 %a, <vscale x 8 x i1> %b) {
; CHECK-LABEL: uqincp_n64_b16:
; CHECK: uqincp x0, p0.h
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.uqincp.n64.nxv8i1(i64 %a, <vscale x 8 x i1> %b)
  ret i64 %out
}

define i64 @uqincp_n64_b32(i64 %a, <vscale x 4 x i1> %b) {
; CHECK-LABEL: uqincp_n64_b32:
; CHECK: uqincp x0, p0.s
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.uqincp.n64.nxv4i1(i64 %a, <vscale x 4 x i1> %b)
  ret i64 %out
}

define i64 @uqincp_n64_b64(i64 %a, <vscale x 2 x i1> %b) {
; CHECK-LABEL: uqincp_n64_b64:
; CHECK: uqincp x0, p0.d
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.uqincp.n64.nxv2i1(i64 %a, <vscale x 2 x i1> %b)
  ret i64 %out
}

; uqinc{h|w|d}(vector, pattern, multiplier)
declare <vscale x 8 x i16> @llvm.aarch64.sve.uqinch.nxv8i16(<vscale x 8 x i16>, i32, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.uqincw.nxv4i32(<vscale x 4 x i32>, i32, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.uqincd.nxv2i64(<vscale x 2 x i64>, i32, i32)

; uqinc{b|h|w|d}(scalar, pattern, multiplier)
declare i32 @llvm.aarch64.sve.uqincb.n32(i32, i32, i32)
declare i64 @llvm.aarch64.sve.uqincb.n64(i64, i32, i32)
declare i32 @llvm.aarch64.sve.uqinch.n32(i32, i32, i32)
declare i64 @llvm.aarch64.sve.uqinch.n64(i64, i32, i32)
declare i32 @llvm.aarch64.sve.uqincw.n32(i32, i32, i32)
declare i64 @llvm.aarch64.sve.uqincw.n64(i64, i32, i32)
declare i32 @llvm.aarch64.sve.uqincd.n32(i32, i32, i32)
declare i64 @llvm.aarch64.sve.uqincd.n64(i64, i32, i32)

; uqincp(scalar, predicate)
declare i32 @llvm.aarch64.sve.uqincp.n32.nxv16i1(i32, <vscale x 16 x i1>)
declare i32 @llvm.aarch64.sve.uqincp.n32.nxv8i1(i32, <vscale x 8 x i1>)
declare i32 @llvm.aarch64.sve.uqincp.n32.nxv4i1(i32, <vscale x 4 x i1>)
declare i32 @llvm.aarch64.sve.uqincp.n32.nxv2i1(i32, <vscale x 2 x i1>)

declare i64 @llvm.aarch64.sve.uqincp.n64.nxv16i1(i64, <vscale x 16 x i1>)
declare i64 @llvm.aarch64.sve.uqincp.n64.nxv8i1(i64, <vscale x 8 x i1>)
declare i64 @llvm.aarch64.sve.uqincp.n64.nxv4i1(i64, <vscale x 4 x i1>)
declare i64 @llvm.aarch64.sve.uqincp.n64.nxv2i1(i64, <vscale x 2 x i1>)

; uqincp(vector, predicate)
declare <vscale x 8 x i16> @llvm.aarch64.sve.uqincp.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i1>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.uqincp.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i1>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.uqincp.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i1>)
