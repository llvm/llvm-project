; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve -asm-verbose=0 < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning

; Since SQDEC{B|H|W|D|P} and SQINC{B|H|W|D|P} have identical semantics, the tests for
;   * @llvm.aarch64.sve.sqinc{b|h|w|d|p}, and
;   * @llvm.aarch64.sve.sqdec{b|h|w|d|p}
; should also be identical (with the instruction name being adjusted). When
; updating this file remember to make similar changes in the file testing the
; other intrinsic.

;
; SQINCH (vector)
;

define <vscale x 8 x i16> @sqinch(<vscale x 8 x i16> %a) {
; CHECK-LABEL: sqinch:
; CHECK: sqinch z0.h, pow2
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.sqinch.nxv8i16(<vscale x 8 x i16> %a,
                                                                  i32 0, i32 1)
  ret <vscale x 8 x i16> %out
}

;
; SQINCW (vector)
;

define <vscale x 4 x i32> @sqincw(<vscale x 4 x i32> %a) {
; CHECK-LABEL: sqincw:
; CHECK: sqincw z0.s, vl1, mul #2
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.sqincw.nxv4i32(<vscale x 4 x i32> %a,
                                                                  i32 1, i32 2)
  ret <vscale x 4 x i32> %out
}

;
; SQINCD (vector)
;

define <vscale x 2 x i64> @sqincd(<vscale x 2 x i64> %a) {
; CHECK-LABEL: sqincd:
; CHECK: sqincd z0.d, vl2, mul #3
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.sqincd.nxv2i64(<vscale x 2 x i64> %a,
                                                                  i32 2, i32 3)
  ret <vscale x 2 x i64> %out
}

;
; SQINCP (vector)
;

define <vscale x 8 x i16> @sqincp_b16(<vscale x 8 x i16> %a, <vscale x 8 x i1> %b) {
; CHECK-LABEL: sqincp_b16:
; CHECK: sqincp z0.h, p0
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.sqincp.nxv8i16(<vscale x 8 x i16> %a,
                                                                  <vscale x 8 x i1> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @sqincp_b32(<vscale x 4 x i32> %a, <vscale x 4 x i1> %b) {
; CHECK-LABEL: sqincp_b32:
; CHECK: sqincp z0.s, p0
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.sqincp.nxv4i32(<vscale x 4 x i32> %a,
                                                                  <vscale x 4 x i1> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @sqincp_b64(<vscale x 2 x i64> %a, <vscale x 2 x i1> %b) {
; CHECK-LABEL: sqincp_b64:
; CHECK: sqincp z0.d, p0
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.sqincp.nxv2i64(<vscale x 2 x i64> %a,
                                                                  <vscale x 2 x i1> %b)
  ret <vscale x 2 x i64> %out
}

;
; SQINCB (scalar)
;

define i32 @sqincb_n32_i32(i32 %a) {
; CHECK-LABEL: sqincb_n32_i32:
; CHECK: sqincb x0, w0, vl3, mul #4
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.sqincb.n32(i32 %a, i32 3, i32 4)
  ret i32 %out
}

define i64 @sqincb_n32_i64(i32 %a) {
; CHECK-LABEL: sqincb_n32_i64:
; CHECK: sqincb x0, w0, vl3, mul #4
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.sqincb.n32(i32 %a, i32 3, i32 4)
  %out_sext = sext i32 %out to i64

  ret i64 %out_sext
}

define i64 @sqincb_n64(i64 %a) {
; CHECK-LABEL: sqincb_n64:
; CHECK: sqincb x0, vl4, mul #5
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.sqincb.n64(i64 %a, i32 4, i32 5)
  ret i64 %out
}

;
; SQINCH (scalar)
;

define i32 @sqinch_n32_i32(i32 %a) {
; CHECK-LABEL: sqinch_n32_i32:
; CHECK: sqinch x0, w0, vl5, mul #6
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.sqinch.n32(i32 %a, i32 5, i32 6)
  ret i32 %out
}

define i64 @sqinch_n32_i64(i32 %a) {
; CHECK-LABEL: sqinch_n32_i64:
; CHECK: sqinch x0, w0, vl3, mul #4
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.sqinch.n32(i32 %a, i32 3, i32 4)
  %out_sext = sext i32 %out to i64

  ret i64 %out_sext
}

define i64 @sqinch_n64(i64 %a) {
; CHECK-LABEL: sqinch_n64:
; CHECK: sqinch x0, vl6, mul #7
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.sqinch.n64(i64 %a, i32 6, i32 7)
  ret i64 %out
}

;
; SQINCW (scalar)
;

define i32 @sqincw_n32_i32(i32 %a) {
; CHECK-LABEL: sqincw_n32_i32:
; CHECK: sqincw x0, w0, vl7, mul #8
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.sqincw.n32(i32 %a, i32 7, i32 8)
  ret i32 %out
}

define i64 @sqincw_n32_i64(i32 %a) {
; CHECK-LABEL: sqincw_n32_i64:
; CHECK: sqincw x0, w0, vl3, mul #4
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.sqincw.n32(i32 %a, i32 3, i32 4)
  %out_sext = sext i32 %out to i64

  ret i64 %out_sext
}

define i64 @sqincw_n64(i64 %a) {
; CHECK-LABEL: sqincw_n64:
; CHECK: sqincw x0, vl8, mul #9
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.sqincw.n64(i64 %a, i32 8, i32 9)
  ret i64 %out
}

;
; SQINCD (scalar)
;

define i32 @sqincd_n32_i32(i32 %a) {
; CHECK-LABEL: sqincd_n32_i32:
; CHECK: sqincd x0, w0, vl16, mul #10
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.sqincd.n32(i32 %a, i32 9, i32 10)
  ret i32 %out
}

define i64 @sqincd_n32_i64(i32 %a) {
; CHECK-LABEL: sqincd_n32_i64:
; CHECK: sqincd x0, w0, vl3, mul #4
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.sqincd.n32(i32 %a, i32 3, i32 4)
  %out_sext = sext i32 %out to i64

  ret i64 %out_sext
}

define i64 @sqincd_n64(i64 %a) {
; CHECK-LABEL: sqincd_n64:
; CHECK: sqincd x0, vl32, mul #11
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.sqincd.n64(i64 %a, i32 10, i32 11)
  ret i64 %out
}

;
; SQINCP (scalar)
;

define i32 @sqincp_n32_b8_i32(i32 %a, <vscale x 16 x i1> %b) {
; CHECK-LABEL: sqincp_n32_b8_i32:
; CHECK: sqincp x0, p0.b, w0
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.sqincp.n32.nxv16i1(i32 %a, <vscale x 16 x i1> %b)
  ret i32 %out
}

define i64 @sqincp_n32_b8_i64(i32 %a, <vscale x 16 x i1> %b) {
; CHECK-LABEL: sqincp_n32_b8_i64:
; CHECK: sqincp x0, p0.b, w0
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.sqincp.n32.nxv16i1(i32 %a, <vscale x 16 x i1> %b)
  %out_sext = sext i32 %out to i64

  ret i64 %out_sext
}

define i32 @sqincp_n32_b16_i32(i32 %a, <vscale x 8 x i1> %b) {
; CHECK-LABEL: sqincp_n32_b16_i32:
; CHECK: sqincp x0, p0.h, w0
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.sqincp.n32.nxv8i1(i32 %a, <vscale x 8 x i1> %b)
  ret i32 %out
}

define i64 @sqincp_n32_b16_i64(i32 %a, <vscale x 8 x i1> %b) {
; CHECK-LABEL: sqincp_n32_b16_i64:
; CHECK: sqincp x0, p0.h, w0
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.sqincp.n32.nxv8i1(i32 %a, <vscale x 8 x i1> %b)
  %out_sext = sext i32 %out to i64

  ret i64 %out_sext
}

define i32 @sqincp_n32_b32_i32(i32 %a, <vscale x 4 x i1> %b) {
; CHECK-LABEL: sqincp_n32_b32_i32:
; CHECK: sqincp x0, p0.s, w0
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.sqincp.n32.nxv4i1(i32 %a, <vscale x 4 x i1> %b)
  ret i32 %out
}

define i64 @sqincp_n32_b32_i64(i32 %a, <vscale x 4 x i1> %b) {
; CHECK-LABEL: sqincp_n32_b32_i64:
; CHECK: sqincp x0, p0.s, w0
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.sqincp.n32.nxv4i1(i32 %a, <vscale x 4 x i1> %b)
  %out_sext = sext i32 %out to i64

  ret i64 %out_sext
}

define i32 @sqincp_n32_b64_i32(i32 %a, <vscale x 2 x i1> %b) {
; CHECK-LABEL: sqincp_n32_b64_i32:
; CHECK: sqincp x0, p0.d, w0
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.sqincp.n32.nxv2i1(i32 %a, <vscale x 2 x i1> %b)
  ret i32 %out
}

define i64 @sqincp_n32_b64_i64(i32 %a, <vscale x 2 x i1> %b) {
; CHECK-LABEL: sqincp_n32_b64_i64:
; CHECK: sqincp x0, p0.d, w0
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.sqincp.n32.nxv2i1(i32 %a, <vscale x 2 x i1> %b)
  %out_sext = sext i32 %out to i64

  ret i64 %out_sext
}

define i64 @sqincp_n64_b8(i64 %a, <vscale x 16 x i1> %b) {
; CHECK-LABEL: sqincp_n64_b8:
; CHECK: sqincp x0, p0.b
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.sqincp.n64.nxv16i1(i64 %a, <vscale x 16 x i1> %b)
  ret i64 %out
}

define i64 @sqincp_n64_b16(i64 %a, <vscale x 8 x i1> %b) {
; CHECK-LABEL: sqincp_n64_b16:
; CHECK: sqincp x0, p0.h
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.sqincp.n64.nxv8i1(i64 %a, <vscale x 8 x i1> %b)
  ret i64 %out
}

define i64 @sqincp_n64_b32(i64 %a, <vscale x 4 x i1> %b) {
; CHECK-LABEL: sqincp_n64_b32:
; CHECK: sqincp x0, p0.s
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.sqincp.n64.nxv4i1(i64 %a, <vscale x 4 x i1> %b)
  ret i64 %out
}

define i64 @sqincp_n64_b64(i64 %a, <vscale x 2 x i1> %b) {
; CHECK-LABEL: sqincp_n64_b64:
; CHECK: sqincp x0, p0.d
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.sqincp.n64.nxv2i1(i64 %a, <vscale x 2 x i1> %b)
  ret i64 %out
}

; sqinc{h|w|d}(vector, pattern, multiplier)
declare <vscale x 8 x i16> @llvm.aarch64.sve.sqinch.nxv8i16(<vscale x 8 x i16>, i32, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sqincw.nxv4i32(<vscale x 4 x i32>, i32, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sqincd.nxv2i64(<vscale x 2 x i64>, i32, i32)

; sqinc{b|h|w|d}(scalar, pattern, multiplier)
declare i32 @llvm.aarch64.sve.sqincb.n32(i32, i32, i32)
declare i64 @llvm.aarch64.sve.sqincb.n64(i64, i32, i32)
declare i32 @llvm.aarch64.sve.sqinch.n32(i32, i32, i32)
declare i64 @llvm.aarch64.sve.sqinch.n64(i64, i32, i32)
declare i32 @llvm.aarch64.sve.sqincw.n32(i32, i32, i32)
declare i64 @llvm.aarch64.sve.sqincw.n64(i64, i32, i32)
declare i32 @llvm.aarch64.sve.sqincd.n32(i32, i32, i32)
declare i64 @llvm.aarch64.sve.sqincd.n64(i64, i32, i32)

; sqincp(scalar, predicate)
declare i32 @llvm.aarch64.sve.sqincp.n32.nxv16i1(i32, <vscale x 16 x i1>)
declare i32 @llvm.aarch64.sve.sqincp.n32.nxv8i1(i32, <vscale x 8 x i1>)
declare i32 @llvm.aarch64.sve.sqincp.n32.nxv4i1(i32, <vscale x 4 x i1>)
declare i32 @llvm.aarch64.sve.sqincp.n32.nxv2i1(i32, <vscale x 2 x i1>)

declare i64 @llvm.aarch64.sve.sqincp.n64.nxv16i1(i64, <vscale x 16 x i1>)
declare i64 @llvm.aarch64.sve.sqincp.n64.nxv8i1(i64, <vscale x 8 x i1>)
declare i64 @llvm.aarch64.sve.sqincp.n64.nxv4i1(i64, <vscale x 4 x i1>)
declare i64 @llvm.aarch64.sve.sqincp.n64.nxv2i1(i64, <vscale x 2 x i1>)

; sqincp(vector, predicate)
declare <vscale x 8 x i16> @llvm.aarch64.sve.sqincp.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i1>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sqincp.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i1>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sqincp.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i1>)
