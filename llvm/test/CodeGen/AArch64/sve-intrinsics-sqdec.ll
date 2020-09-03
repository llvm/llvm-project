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
; SQDECH (vector)
;

define <vscale x 8 x i16> @sqdech(<vscale x 8 x i16> %a) {
; CHECK-LABEL: sqdech:
; CHECK: sqdech z0.h, pow2
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.sqdech.nxv8i16(<vscale x 8 x i16> %a,
                                                                  i32 0, i32 1)
  ret <vscale x 8 x i16> %out
}

;
; SQDECW (vector)
;

define <vscale x 4 x i32> @sqdecw(<vscale x 4 x i32> %a) {
; CHECK-LABEL: sqdecw:
; CHECK: sqdecw z0.s, vl1, mul #2
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.sqdecw.nxv4i32(<vscale x 4 x i32> %a,
                                                                  i32 1, i32 2)
  ret <vscale x 4 x i32> %out
}

;
; SQDECD (vector)
;

define <vscale x 2 x i64> @sqdecd(<vscale x 2 x i64> %a) {
; CHECK-LABEL: sqdecd:
; CHECK: sqdecd z0.d, vl2, mul #3
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.sqdecd.nxv2i64(<vscale x 2 x i64> %a,
                                                                  i32 2, i32 3)
  ret <vscale x 2 x i64> %out
}

;
; SQDECP (vector)
;

define <vscale x 8 x i16> @sqdecp_b16(<vscale x 8 x i16> %a, <vscale x 8 x i1> %b) {
; CHECK-LABEL: sqdecp_b16:
; CHECK: sqdecp z0.h, p0
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.sqdecp.nxv8i16(<vscale x 8 x i16> %a,
                                                                  <vscale x 8 x i1> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @sqdecp_b32(<vscale x 4 x i32> %a, <vscale x 4 x i1> %b) {
; CHECK-LABEL: sqdecp_b32:
; CHECK: sqdecp z0.s, p0
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.sqdecp.nxv4i32(<vscale x 4 x i32> %a,
                                                                  <vscale x 4 x i1> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @sqdecp_b64(<vscale x 2 x i64> %a, <vscale x 2 x i1> %b) {
; CHECK-LABEL: sqdecp_b64:
; CHECK: sqdecp z0.d, p0
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.sqdecp.nxv2i64(<vscale x 2 x i64> %a,
                                                                  <vscale x 2 x i1> %b)
  ret <vscale x 2 x i64> %out
}

;
; SQDECB (scalar)
;

define i32 @sqdecb_n32_i32(i32 %a) {
; CHECK-LABEL: sqdecb_n32_i32:
; CHECK: sqdecb x0, w0, vl3, mul #4
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.sqdecb.n32(i32 %a, i32 3, i32 4)
  ret i32 %out
}

define i64 @sqdecb_n32_i64(i32 %a) {
; CHECK-LABEL: sqdecb_n32_i64:
; CHECK: sqdecb x0, w0, vl3, mul #4
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.sqdecb.n32(i32 %a, i32 3, i32 4)
  %out_sext = sext i32 %out to i64

  ret i64 %out_sext
}

define i64 @sqdecb_n64(i64 %a) {
; CHECK-LABEL: sqdecb_n64:
; CHECK: sqdecb x0, vl4, mul #5
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.sqdecb.n64(i64 %a, i32 4, i32 5)
  ret i64 %out
}

;
; SQDECH (scalar)
;

define i32 @sqdech_n32_i32(i32 %a) {
; CHECK-LABEL: sqdech_n32_i32:
; CHECK: sqdech x0, w0, vl5, mul #6
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.sqdech.n32(i32 %a, i32 5, i32 6)
  ret i32 %out
}

define i64 @sqdech_n32_i64(i32 %a) {
; CHECK-LABEL: sqdech_n32_i64:
; CHECK: sqdech x0, w0, vl3, mul #4
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.sqdech.n32(i32 %a, i32 3, i32 4)
  %out_sext = sext i32 %out to i64

  ret i64 %out_sext
}

define i64 @sqdech_n64(i64 %a) {
; CHECK-LABEL: sqdech_n64:
; CHECK: sqdech x0, vl6, mul #7
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.sqdech.n64(i64 %a, i32 6, i32 7)
  ret i64 %out
}

;
; SQDECW (scalar)
;

define i32 @sqdecw_n32_i32(i32 %a) {
; CHECK-LABEL: sqdecw_n32_i32:
; CHECK: sqdecw x0, w0, vl7, mul #8
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.sqdecw.n32(i32 %a, i32 7, i32 8)
  ret i32 %out
}

define i64 @sqdecw_n32_i64(i32 %a) {
; CHECK-LABEL: sqdecw_n32_i64:
; CHECK: sqdecw x0, w0, vl3, mul #4
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.sqdecw.n32(i32 %a, i32 3, i32 4)
  %out_sext = sext i32 %out to i64

  ret i64 %out_sext
}

define i64 @sqdecw_n64(i64 %a) {
; CHECK-LABEL: sqdecw_n64:
; CHECK: sqdecw x0, vl8, mul #9
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.sqdecw.n64(i64 %a, i32 8, i32 9)
  ret i64 %out
}

;
; SQDECD (scalar)
;

define i32 @sqdecd_n32_i32(i32 %a) {
; CHECK-LABEL: sqdecd_n32_i32:
; CHECK: sqdecd x0, w0, vl16, mul #10
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.sqdecd.n32(i32 %a, i32 9, i32 10)
  ret i32 %out
}

define i64 @sqdecd_n32_i64(i32 %a) {
; CHECK-LABEL: sqdecd_n32_i64:
; CHECK: sqdecd x0, w0, vl3, mul #4
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.sqdecd.n32(i32 %a, i32 3, i32 4)
  %out_sext = sext i32 %out to i64

  ret i64 %out_sext
}

define i64 @sqdecd_n64(i64 %a) {
; CHECK-LABEL: sqdecd_n64:
; CHECK: sqdecd x0, vl32, mul #11
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.sqdecd.n64(i64 %a, i32 10, i32 11)
  ret i64 %out
}

;
; SQDECP (scalar)
;

define i32 @sqdecp_n32_b8_i32(i32 %a, <vscale x 16 x i1> %b) {
; CHECK-LABEL: sqdecp_n32_b8_i32:
; CHECK: sqdecp x0, p0.b, w0
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.sqdecp.n32.nxv16i1(i32 %a, <vscale x 16 x i1> %b)
  ret i32 %out
}

define i64 @sqdecp_n32_b8_i64(i32 %a, <vscale x 16 x i1> %b) {
; CHECK-LABEL: sqdecp_n32_b8_i64:
; CHECK: sqdecp x0, p0.b, w0
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.sqdecp.n32.nxv16i1(i32 %a, <vscale x 16 x i1> %b)
  %out_sext = sext i32 %out to i64

  ret i64 %out_sext
}

define i32 @sqdecp_n32_b16_i32(i32 %a, <vscale x 8 x i1> %b) {
; CHECK-LABEL: sqdecp_n32_b16_i32:
; CHECK: sqdecp x0, p0.h, w0
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.sqdecp.n32.nxv8i1(i32 %a, <vscale x 8 x i1> %b)
  ret i32 %out
}

define i64 @sqdecp_n32_b16_i64(i32 %a, <vscale x 8 x i1> %b) {
; CHECK-LABEL: sqdecp_n32_b16_i64:
; CHECK: sqdecp x0, p0.h, w0
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.sqdecp.n32.nxv8i1(i32 %a, <vscale x 8 x i1> %b)
  %out_sext = sext i32 %out to i64

  ret i64 %out_sext
}

define i32 @sqdecp_n32_b32_i32(i32 %a, <vscale x 4 x i1> %b) {
; CHECK-LABEL: sqdecp_n32_b32_i32:
; CHECK: sqdecp x0, p0.s, w0
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.sqdecp.n32.nxv4i1(i32 %a, <vscale x 4 x i1> %b)
  ret i32 %out
}

define i64 @sqdecp_n32_b32_i64(i32 %a, <vscale x 4 x i1> %b) {
; CHECK-LABEL: sqdecp_n32_b32_i64:
; CHECK: sqdecp x0, p0.s, w0
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.sqdecp.n32.nxv4i1(i32 %a, <vscale x 4 x i1> %b)
  %out_sext = sext i32 %out to i64

  ret i64 %out_sext
}

define i32 @sqdecp_n32_b64_i32(i32 %a, <vscale x 2 x i1> %b) {
; CHECK-LABEL: sqdecp_n32_b64_i32:
; CHECK: sqdecp x0, p0.d, w0
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.sqdecp.n32.nxv2i1(i32 %a, <vscale x 2 x i1> %b)
  ret i32 %out
}

define i64 @sqdecp_n32_b64_i64(i32 %a, <vscale x 2 x i1> %b) {
; CHECK-LABEL: sqdecp_n32_b64_i64:
; CHECK: sqdecp x0, p0.d, w0
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.sqdecp.n32.nxv2i1(i32 %a, <vscale x 2 x i1> %b)
  %out_sext = sext i32 %out to i64

  ret i64 %out_sext
}

define i64 @sqdecp_n64_b8(i64 %a, <vscale x 16 x i1> %b) {
; CHECK-LABEL: sqdecp_n64_b8:
; CHECK: sqdecp x0, p0.b
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.sqdecp.n64.nxv16i1(i64 %a, <vscale x 16 x i1> %b)
  ret i64 %out
}

define i64 @sqdecp_n64_b16(i64 %a, <vscale x 8 x i1> %b) {
; CHECK-LABEL: sqdecp_n64_b16:
; CHECK: sqdecp x0, p0.h
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.sqdecp.n64.nxv8i1(i64 %a, <vscale x 8 x i1> %b)
  ret i64 %out
}

define i64 @sqdecp_n64_b32(i64 %a, <vscale x 4 x i1> %b) {
; CHECK-LABEL: sqdecp_n64_b32:
; CHECK: sqdecp x0, p0.s
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.sqdecp.n64.nxv4i1(i64 %a, <vscale x 4 x i1> %b)
  ret i64 %out
}

define i64 @sqdecp_n64_b64(i64 %a, <vscale x 2 x i1> %b) {
; CHECK-LABEL: sqdecp_n64_b64:
; CHECK: sqdecp x0, p0.d
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.sqdecp.n64.nxv2i1(i64 %a, <vscale x 2 x i1> %b)
  ret i64 %out
}

; sqdec{h|w|d}(vector, pattern, multiplier)
declare <vscale x 8 x i16> @llvm.aarch64.sve.sqdech.nxv8i16(<vscale x 8 x i16>, i32, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sqdecw.nxv4i32(<vscale x 4 x i32>, i32, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sqdecd.nxv2i64(<vscale x 2 x i64>, i32, i32)

; sqdec{b|h|w|d}(scalar, pattern, multiplier)
declare i32 @llvm.aarch64.sve.sqdecb.n32(i32, i32, i32)
declare i64 @llvm.aarch64.sve.sqdecb.n64(i64, i32, i32)
declare i32 @llvm.aarch64.sve.sqdech.n32(i32, i32, i32)
declare i64 @llvm.aarch64.sve.sqdech.n64(i64, i32, i32)
declare i32 @llvm.aarch64.sve.sqdecw.n32(i32, i32, i32)
declare i64 @llvm.aarch64.sve.sqdecw.n64(i64, i32, i32)
declare i32 @llvm.aarch64.sve.sqdecd.n32(i32, i32, i32)
declare i64 @llvm.aarch64.sve.sqdecd.n64(i64, i32, i32)

; sqdecp(scalar, predicate)
declare i32 @llvm.aarch64.sve.sqdecp.n32.nxv16i1(i32, <vscale x 16 x i1>)
declare i32 @llvm.aarch64.sve.sqdecp.n32.nxv8i1(i32, <vscale x 8 x i1>)
declare i32 @llvm.aarch64.sve.sqdecp.n32.nxv4i1(i32, <vscale x 4 x i1>)
declare i32 @llvm.aarch64.sve.sqdecp.n32.nxv2i1(i32, <vscale x 2 x i1>)

declare i64 @llvm.aarch64.sve.sqdecp.n64.nxv16i1(i64, <vscale x 16 x i1>)
declare i64 @llvm.aarch64.sve.sqdecp.n64.nxv8i1(i64, <vscale x 8 x i1>)
declare i64 @llvm.aarch64.sve.sqdecp.n64.nxv4i1(i64, <vscale x 4 x i1>)
declare i64 @llvm.aarch64.sve.sqdecp.n64.nxv2i1(i64, <vscale x 2 x i1>)

; sqdecp(vector, predicate)
declare <vscale x 8 x i16> @llvm.aarch64.sve.sqdecp.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i1>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sqdecp.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i1>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sqdecp.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i1>)
