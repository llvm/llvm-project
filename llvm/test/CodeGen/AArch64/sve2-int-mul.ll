; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve2 < %s | FileCheck %s

;
; MUL with SPLAT
;
define <vscale x 8 x i16> @mul_i16_imm(<vscale x 8 x i16> %a) {
; CHECK-LABEL: mul_i16_imm
; CHECK: mov w[[W:[0-9]+]], #255
; CHECK-NEXT: mov z1.h, w[[W]]
; CHECK-NEXT: mul z0.h, z0.h, z1.h
  %elt = insertelement <vscale x 8 x i16> undef, i16 255, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %res = mul <vscale x 8 x i16> %a, %splat
  ret <vscale x 8 x i16> %res
}

define <vscale x 8 x i16> @mul_i16_imm_neg(<vscale x 8 x i16> %a) {
; CHECK-LABEL: mul_i16_imm_neg
; CHECK: mov w[[W:[0-9]+]], #-200
; CHECK-NEXT: mov z1.h, w[[W]]
; CHECK-NEXT: mul z0.h, z0.h, z1.h
  %elt = insertelement <vscale x 8 x i16> undef, i16 -200, i32 0
  %splat = shufflevector <vscale x 8 x i16> %elt, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %res = mul <vscale x 8 x i16> %a, %splat
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @mul_i32_imm(<vscale x 4 x i32> %a) {
; CHECK-LABEL: mul_i32_imm
; CHECK: mov w[[W:[0-9]+]], #255
; CHECK-NEXT: mov z1.s, w[[W]]
; CHECK-NEXT: mul z0.s, z0.s, z1.s
  %elt = insertelement <vscale x 4 x i32> undef, i32 255, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %res = mul <vscale x 4 x i32> %a, %splat
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x i32> @mul_i32_imm_neg(<vscale x 4 x i32> %a) {
; CHECK-LABEL: mul_i32_imm_neg
; CHECK: mov w[[W:[0-9]+]], #-200
; CHECK-NEXT: mov z1.s, w[[W]]
; CHECK-NEXT: mul z0.s, z0.s, z1.s
  %elt = insertelement <vscale x 4 x i32> undef, i32 -200, i32 0
  %splat = shufflevector <vscale x 4 x i32> %elt, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %res = mul <vscale x 4 x i32> %a, %splat
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @mul_i64_imm(<vscale x 2 x i64> %a) {
; CHECK-LABEL: mul_i64_imm
; CHECK: mov w[[X:[0-9]+]], #255
; CHECK-NEXT: z1.d, x[[X]]
; CHECK-NEXT: mul z0.d, z0.d, z1.d
  %elt = insertelement <vscale x 2 x i64> undef, i64 255, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %res = mul <vscale x 2 x i64> %a, %splat
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @mul_i64_imm_neg(<vscale x 2 x i64> %a) {
; CHECK-LABEL: mul_i64_imm_neg
; CHECK: mov x[[X:[0-9]+]], #-200
; CHECK-NEXT: z1.d, x[[X]]
; CHECK-NEXT: mul z0.d, z0.d, z1.d
  %elt = insertelement <vscale x 2 x i64> undef, i64 -200, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %res = mul <vscale x 2 x i64> %a, %splat
  ret <vscale x 2 x i64> %res
}

;
; MUL (vector, unpredicated)
;
define <vscale x 16 x i8> @mul_i8(<vscale x 16 x i8> %a,
                                  <vscale x 16 x i8> %b) {
; CHECK-LABEL: mul_i8
; CHECK: mul z0.b, z0.b, z1.b
; CHECK-NEXT: ret
  %res = mul <vscale x 16 x i8> %a, %b
  ret <vscale x 16 x i8> %res
}

define <vscale x 8 x i16> @mul_i16(<vscale x 8 x i16> %a,
                                  <vscale x 8 x i16> %b) {
; CHECK-LABEL: mul_i16
; CHECK: mul z0.h, z0.h, z1.h
; CHECK-NEXT: ret
  %res = mul <vscale x 8 x i16> %a, %b
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @mul_i32(<vscale x 4 x i32> %a,
                                  <vscale x 4 x i32> %b) {
; CHECK-LABEL: mul_i32
; CHECK: mul z0.s, z0.s, z1.s
; CHECK-NEXT: ret
  %res = mul <vscale x 4 x i32> %a, %b
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @mul_i64(<vscale x 2 x i64> %a,
                                  <vscale x 2 x i64> %b) {
; CHECK-LABEL: mul_i64
; CHECK: mul z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %res = mul <vscale x 2 x i64> %a, %b
  ret <vscale x 2 x i64> %res
}

;
; SMULH (vector, unpredicated)
;
define <vscale x 16 x i8> @smulh_i8(<vscale x 16 x i8> %a,
                                    <vscale x 16 x i8> %b) {
; CHECK-LABEL: smulh_i8
; CHECK: smulh z0.b, z0.b, z1.b
; CHECK-NEXT: ret
  %sel = call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 31)
  %res = call <vscale x 16 x i8> @llvm.aarch64.sve.smulh.nxv16i8(<vscale x 16 x i1> %sel, <vscale x 16 x i8> %a,
                                                                 <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %res
}

define <vscale x 8 x i16> @smulh_i16(<vscale x 8 x i16> %a,
                                     <vscale x 8 x i16> %b) {
; CHECK-LABEL: smulh_i16
; CHECK: smulh z0.h, z0.h, z1.h
; CHECK-NEXT: ret
  %sel = call <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32 31)
  %res = call <vscale x 8 x i16> @llvm.aarch64.sve.smulh.nxv8i16(<vscale x 8 x i1> %sel, <vscale x 8 x i16> %a,
                                                                 <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @smulh_i32(<vscale x 4 x i32> %a,
                                     <vscale x 4 x i32> %b) {
; CHECK-LABEL: smulh_i32
; CHECK: smulh z0.s, z0.s, z1.s
; CHECK-NEXT: ret
  %sel = call <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32 31)
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.smulh.nxv4i32(<vscale x 4 x i1> %sel, <vscale x 4 x i32> %a,
                                                                 <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @smulh_i64(<vscale x 2 x i64> %a,
                                     <vscale x 2 x i64> %b) {
; CHECK-LABEL: smulh_i64
; CHECK: smulh z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %sel = call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.smulh.nxv2i64(<vscale x 2 x i1> %sel, <vscale x 2 x i64> %a,
                                                                 <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %res
}

;
; UMULH (vector, unpredicated)
;
define <vscale x 16 x i8> @umulh_i8(<vscale x 16 x i8> %a,
                                    <vscale x 16 x i8> %b) {
; CHECK-LABEL: umulh_i8
; CHECK: umulh z0.b, z0.b, z1.b
; CHECK-NEXT: ret
  %sel = call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 31)
  %res = call <vscale x 16 x i8> @llvm.aarch64.sve.umulh.nxv16i8(<vscale x 16 x i1> %sel, <vscale x 16 x i8> %a,
                                                                 <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %res
}

define <vscale x 8 x i16> @umulh_i16(<vscale x 8 x i16> %a,
                                     <vscale x 8 x i16> %b) {
; CHECK-LABEL: umulh_i16
; CHECK: umulh z0.h, z0.h, z1.h
; CHECK-NEXT: ret
  %sel = call <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32 31)
  %res = call <vscale x 8 x i16> @llvm.aarch64.sve.umulh.nxv8i16(<vscale x 8 x i1> %sel, <vscale x 8 x i16> %a,
                                                                 <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @umulh_i32(<vscale x 4 x i32> %a,
                                     <vscale x 4 x i32> %b) {
; CHECK-LABEL: umulh_i32
; CHECK: umulh z0.s, z0.s, z1.s
; CHECK-NEXT: ret
  %sel = call <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32 31)
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.umulh.nxv4i32(<vscale x 4 x i1> %sel, <vscale x 4 x i32> %a,
                                                                 <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @umulh_i64(<vscale x 2 x i64> %a,
                                     <vscale x 2 x i64> %b) {
; CHECK-LABEL: umulh_i64
; CHECK: umulh z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %sel = call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.umulh.nxv2i64(<vscale x 2 x i1> %sel, <vscale x 2 x i64> %a,
                                                                 <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %res
}

;
; PMUL (vector, unpredicated)
;
define <vscale x 16 x i8> @pmul_i8(<vscale x 16 x i8> %a,
                                   <vscale x 16 x i8> %b) {
; CHECK-LABEL: pmul_i8
; CHECK: pmul z0.b, z0.b, z1.b
; CHECK-NEXT: ret
  %res = call <vscale x 16 x i8> @llvm.aarch64.sve.pmul.nxv16i8(<vscale x 16 x i8> %a,
                                                                <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %res
}

;
; SQDMULH (vector, unpredicated)
;
define <vscale x 16 x i8> @sqdmulh_i8(<vscale x 16 x i8> %a,
                                      <vscale x 16 x i8> %b) {
; CHECK-LABEL: sqdmulh_i8
; CHECK: sqdmulh z0.b, z0.b, z1.b
; CHECK-NEXT: ret
  %res = call <vscale x 16 x i8> @llvm.aarch64.sve.sqdmulh.nxv16i8(<vscale x 16 x i8> %a,
                                                                   <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %res
}

define <vscale x 8 x i16> @sqdmulh_i16(<vscale x 8 x i16> %a,
                                       <vscale x 8 x i16> %b) {
; CHECK-LABEL: sqdmulh_i16
; CHECK: sqdmulh z0.h, z0.h, z1.h
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i16> @llvm.aarch64.sve.sqdmulh.nxv8i16(<vscale x 8 x i16> %a,
                                                                   <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @sqdmulh_i32(<vscale x 4 x i32> %a,
                                       <vscale x 4 x i32> %b) {
; CHECK-LABEL: sqdmulh_i32
; CHECK: sqdmulh z0.s, z0.s, z1.s
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.sqdmulh.nxv4i32(<vscale x 4 x i32> %a,
                                                                   <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @sqdmulh_i64(<vscale x 2 x i64> %a,
                                       <vscale x 2 x i64> %b) {
; CHECK-LABEL: sqdmulh_i64
; CHECK: sqdmulh z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.sqdmulh.nxv2i64(<vscale x 2 x i64> %a,
                                                                   <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %res
}

;
; SQRDMULH (vector, unpredicated)
;
define <vscale x 16 x i8> @sqrdmulh_i8(<vscale x 16 x i8> %a,
                                       <vscale x 16 x i8> %b) {
; CHECK-LABEL: sqrdmulh_i8
; CHECK: sqrdmulh z0.b, z0.b, z1.b
; CHECK-NEXT: ret
  %res = call <vscale x 16 x i8> @llvm.aarch64.sve.sqrdmulh.nxv16i8(<vscale x 16 x i8> %a,
                                                                    <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %res
}

define <vscale x 8 x i16> @sqrdmulh_i16(<vscale x 8 x i16> %a,
                                        <vscale x 8 x i16> %b) {
; CHECK-LABEL: sqrdmulh_i16
; CHECK: sqrdmulh z0.h, z0.h, z1.h
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i16> @llvm.aarch64.sve.sqrdmulh.nxv8i16(<vscale x 8 x i16> %a,
                                                                    <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @sqrdmulh_i32(<vscale x 4 x i32> %a,
                                        <vscale x 4 x i32> %b) {
; CHECK-LABEL: sqrdmulh_i32
; CHECK: sqrdmulh z0.s, z0.s, z1.s
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.sqrdmulh.nxv4i32(<vscale x 4 x i32> %a,
                                                                    <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @sqrdmulh_i64(<vscale x 2 x i64> %a,
                                        <vscale x 2 x i64> %b) {
; CHECK-LABEL: sqrdmulh_i64
; CHECK: sqrdmulh z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.sqrdmulh.nxv2i64(<vscale x 2 x i64> %a,
                                                                    <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %res
}

declare <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32)
declare <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32)
declare <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32)
declare <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32)
declare <vscale x 16 x  i8> @llvm.aarch64.sve.smulh.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x  i8>, <vscale x 16 x  i8>)
declare <vscale x  8 x i16> @llvm.aarch64.sve.smulh.nxv8i16(<vscale x  8 x i1>, <vscale x  8 x i16>, <vscale x  8 x i16>)
declare <vscale x  4 x i32> @llvm.aarch64.sve.smulh.nxv4i32(<vscale x  4 x i1>, <vscale x  4 x i32>, <vscale x  4 x i32>)
declare <vscale x  2 x i64> @llvm.aarch64.sve.smulh.nxv2i64(<vscale x  2 x i1>, <vscale x  2 x i64>, <vscale x  2 x i64>)
declare <vscale x 16 x  i8> @llvm.aarch64.sve.umulh.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x  i8>, <vscale x 16 x  i8>)
declare <vscale x  8 x i16> @llvm.aarch64.sve.umulh.nxv8i16(<vscale x  8 x i1>, <vscale x  8 x i16>, <vscale x  8 x i16>)
declare <vscale x  4 x i32> @llvm.aarch64.sve.umulh.nxv4i32(<vscale x  4 x i1>, <vscale x  4 x i32>, <vscale x  4 x i32>)
declare <vscale x  2 x i64> @llvm.aarch64.sve.umulh.nxv2i64(<vscale x  2 x i1>, <vscale x  2 x i64>, <vscale x  2 x i64>)
declare <vscale x 16 x i8> @llvm.aarch64.sve.pmul.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 16 x i8> @llvm.aarch64.sve.sqdmulh.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.sqdmulh.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sqdmulh.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sqdmulh.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>)
declare <vscale x 16 x i8> @llvm.aarch64.sve.sqrdmulh.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.sqrdmulh.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sqrdmulh.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sqrdmulh.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>)
