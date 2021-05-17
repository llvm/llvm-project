; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve -mattr=+use-experimental-zeroing-pseudos < %s | FileCheck %s

;
; ASR
;

define <vscale x 16 x i8> @asr_i8_zero(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: asr_i8_zero:
; CHECK:      movprfx z0.b, p0/z, z0.b
; CHECK-NEXT: asr z0.b, p0/m, z0.b, z1.b
; CHECK-NEXT: ret
  %a_z = select <vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 16 x i8> zeroinitializer
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.asr.nxv16i8(<vscale x 16 x i1> %pg,
                                                          <vscale x 16 x i8> %a_z,
                                                          <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @asr_i16_zero(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: asr_i16_zero:
; CHECK:      movprfx z0.h, p0/z, z0.h
; CHECK-NEXT: asr z0.h, p0/m, z0.h, z1.h
; CHECK-NEXT: ret
  %a_z = select <vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 8 x i16> zeroinitializer
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.asr.nxv8i16(<vscale x 8 x i1> %pg,
                                                          <vscale x 8 x i16> %a_z,
                                                          <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @asr_i32_zero(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: asr_i32_zero:
; CHECK:      movprfx z0.s, p0/z, z0.s
; CHECK-NEXT: asr z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %a_z = select <vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> zeroinitializer
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.asr.nxv4i32(<vscale x 4 x i1> %pg,
                                                          <vscale x 4 x i32> %a_z,
                                                          <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @asr_i64_zero(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: asr_i64_zero:
; CHECK:      movprfx z0.d, p0/z, z0.d
; CHECK-NEXT: asr z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %a_z = select <vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 2 x i64> zeroinitializer
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.asr.nxv2i64(<vscale x 2 x i1> %pg,
                                                          <vscale x 2 x i64> %a_z,
                                                          <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

define <vscale x 16 x i8> @asr_wide_i8_zero(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: asr_wide_i8_zero:
; CHECK-NOT:  movprfx
; CHECK: asr z0.b, p0/m, z0.b, z1.d
  %a_z = select <vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 16 x i8> zeroinitializer
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.asr.wide.nxv16i8(<vscale x 16 x i1> %pg,
                                                               <vscale x 16 x i8> %a_z,
                                                               <vscale x 2 x i64> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @asr_wide_i16_zero(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: asr_wide_i16_zero:
; CHECK-NOT:  movprfx
; CHECK: asr z0.h, p0/m, z0.h, z1.d
  %a_z = select <vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 8 x i16> zeroinitializer
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.asr.wide.nxv8i16(<vscale x 8 x i1> %pg,
                                                               <vscale x 8 x i16> %a_z,
                                                               <vscale x 2 x i64> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @asr_wide_i32_zero(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: asr_wide_i32_zero:
; CHECK-NOT:  movprfx
; CHECK: asr z0.s, p0/m, z0.s, z1.d
  %a_z = select <vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> zeroinitializer
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.asr.wide.nxv4i32(<vscale x 4 x i1> %pg,
                                                               <vscale x 4 x i32> %a_z,
                                                               <vscale x 2 x i64> %b)
  ret <vscale x 4 x i32> %out
}

;
; ASRD
;

define <vscale x 16 x i8> @asrd_i8_zero(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a) {
; CHECK-LABEL: asrd_i8_zero:
; CHECK:      movprfx z0.b, p0/z, z0.b
; CHECK-NEXT: asrd z0.b, p0/m, z0.b, #1
; CHECK-NEXT: ret
  %a_z = select <vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 16 x i8> zeroinitializer
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.asrd.nxv16i8(<vscale x 16 x i1> %pg,
                                                           <vscale x 16 x i8> %a_z,
                                                           i32 1)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @asrd_i16_zero(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a) {
; CHECK-LABEL: asrd_i16_zero:
; CHECK:      movprfx z0.h, p0/z, z0.h
; CHECK-NEXT: asrd z0.h, p0/m, z0.h, #2
; CHECK-NEXT: ret
  %a_z = select <vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 8 x i16> zeroinitializer
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.asrd.nxv8i16(<vscale x 8 x i1> %pg,
                                                           <vscale x 8 x i16> %a_z,
                                                           i32 2)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @asrd_i32_zero(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a) {
; CHECK-LABEL: asrd_i32_zero:
; CHECK:      movprfx z0.s, p0/z, z0.s
; CHECK-NEXT: asrd z0.s, p0/m, z0.s, #31
; CHECK-NEXT: ret
  %a_z = select <vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> zeroinitializer
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.asrd.nxv4i32(<vscale x 4 x i1> %pg,
                                                           <vscale x 4 x i32> %a_z,
                                                           i32 31)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @asrd_i64_zero(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a) {
; CHECK-LABEL: asrd_i64_zero:
; CHECK:      movprfx z0.d, p0/z, z0.d
; CHECK-NEXT: asrd z0.d, p0/m, z0.d, #64
; CHECK-NEXT: ret
  %a_z = select <vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 2 x i64> zeroinitializer
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.asrd.nxv2i64(<vscale x 2 x i1> %pg,
                                                           <vscale x 2 x i64> %a_z,
                                                           i32 64)
  ret <vscale x 2 x i64> %out
}

;
; LSL
;

define <vscale x 16 x i8> @lsl_i8_zero(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: lsl_i8_zero:
; CHECK:      movprfx z0.b, p0/z, z0.b
; CHECK-NEXT: lsl z0.b, p0/m, z0.b, z1.b
; CHECK-NEXT: ret
  %a_z = select <vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 16 x i8> zeroinitializer
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.lsl.nxv16i8(<vscale x 16 x i1> %pg,
                                                          <vscale x 16 x i8> %a_z,
                                                          <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @lsl_i16_zero(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: lsl_i16_zero:
; CHECK:      movprfx z0.h, p0/z, z0.h
; CHECK-NEXT: lsl z0.h, p0/m, z0.h, z1.h
; CHECK-NEXT: ret
  %a_z = select <vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 8 x i16> zeroinitializer
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.lsl.nxv8i16(<vscale x 8 x i1> %pg,
                                                          <vscale x 8 x i16> %a_z,
                                                          <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @lsl_i32_zero(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: lsl_i32_zero:
; CHECK:      movprfx z0.s, p0/z, z0.s
; CHECK-NEXT: lsl z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %a_z = select <vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> zeroinitializer
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.lsl.nxv4i32(<vscale x 4 x i1> %pg,
                                                          <vscale x 4 x i32> %a_z,
                                                          <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @lsl_i64_zero(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: lsl_i64_zero:
; CHECK:      movprfx z0.d, p0/z, z0.d
; CHECK-NEXT: lsl z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %a_z = select <vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 2 x i64> zeroinitializer
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.lsl.nxv2i64(<vscale x 2 x i1> %pg,
                                                          <vscale x 2 x i64> %a_z,
                                                          <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

define <vscale x 16 x i8> @lsl_wide_i8_zero(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: lsl_wide_i8_zero:
; CHECK-NOT:  movprfx
; CHECK: lsl z0.b, p0/m, z0.b, z1.d
  %a_z = select <vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 16 x i8> zeroinitializer
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.lsl.wide.nxv16i8(<vscale x 16 x i1> %pg,
                                                               <vscale x 16 x i8> %a_z,
                                                               <vscale x 2 x i64> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @lsl_wide_i16_zero(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: lsl_wide_i16_zero:
; CHECK-NOT:  movprfx
; CHECK: lsl z0.h, p0/m, z0.h, z1.d
  %a_z = select <vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 8 x i16> zeroinitializer
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.lsl.wide.nxv8i16(<vscale x 8 x i1> %pg,
                                                               <vscale x 8 x i16> %a_z,
                                                               <vscale x 2 x i64> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @lsl_wide_i32_zero(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: lsl_wide_i32_zero:
; CHECK-NOT:  movprfx
; CHECK: lsl z0.s, p0/m, z0.s, z1.d
  %a_z = select <vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> zeroinitializer
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.lsl.wide.nxv4i32(<vscale x 4 x i1> %pg,
                                                               <vscale x 4 x i32> %a_z,
                                                               <vscale x 2 x i64> %b)
  ret <vscale x 4 x i32> %out
}

;
; LSR
;

define <vscale x 16 x i8> @lsr_i8_zero(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: lsr_i8_zero:
; CHECK:      movprfx z0.b, p0/z, z0.b
; CHECK-NEXT: lsr z0.b, p0/m, z0.b, z1.b
; CHECK-NEXT: ret
  %a_z = select <vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 16 x i8> zeroinitializer
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.lsr.nxv16i8(<vscale x 16 x i1> %pg,
                                                          <vscale x 16 x i8> %a_z,
                                                          <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @lsr_i16_zero(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: lsr_i16_zero:
; CHECK:      movprfx z0.h, p0/z, z0.h
; CHECK-NEXT: lsr z0.h, p0/m, z0.h, z1.h
; CHECK-NEXT: ret
  %a_z = select <vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 8 x i16> zeroinitializer
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.lsr.nxv8i16(<vscale x 8 x i1> %pg,
                                                          <vscale x 8 x i16> %a_z,
                                                          <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @lsr_i32_zero(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: lsr_i32_zero:
; CHECK:      movprfx z0.s, p0/z, z0.s
; CHECK-NEXT: lsr z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT: ret
  %a_z = select <vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> zeroinitializer
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.lsr.nxv4i32(<vscale x 4 x i1> %pg,
                                                          <vscale x 4 x i32> %a_z,
                                                          <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @lsr_i64_zero(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: lsr_i64_zero:
; CHECK:      movprfx z0.d, p0/z, z0.d
; CHECK-NEXT: lsr z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT: ret
  %a_z = select <vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 2 x i64> zeroinitializer
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.lsr.nxv2i64(<vscale x 2 x i1> %pg,
                                                          <vscale x 2 x i64> %a_z,
                                                          <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

define <vscale x 16 x i8> @lsr_wide_i8_zero(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: lsr_wide_i8_zero:
; CHECK-NOT:  movprfx
; CHECK: lsr z0.b, p0/m, z0.b, z1.d
  %a_z = select <vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 16 x i8> zeroinitializer
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.lsr.wide.nxv16i8(<vscale x 16 x i1> %pg,
                                                               <vscale x 16 x i8> %a_z,
                                                               <vscale x 2 x i64> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @lsr_wide_i16_zero(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: lsr_wide_i16_zero:
; CHECK-NOT:  movprfx
; CHECK: lsr z0.h, p0/m, z0.h, z1.d
  %a_z = select <vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 8 x i16> zeroinitializer
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.lsr.wide.nxv8i16(<vscale x 8 x i1> %pg,
                                                               <vscale x 8 x i16> %a_z,
                                                               <vscale x 2 x i64> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @lsr_wide_i32_zero(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: lsr_wide_i32_zero:
; CHECK-NOT:  movprfx
; CHECK: lsr z0.s, p0/m, z0.s, z1.d
  %a_z = select <vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> zeroinitializer
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.lsr.wide.nxv4i32(<vscale x 4 x i1> %pg,
                                                               <vscale x 4 x i32> %a_z,
                                                               <vscale x 2 x i64> %b)
  ret <vscale x 4 x i32> %out
}

declare <vscale x 16 x i8> @llvm.aarch64.sve.asr.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.asr.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.asr.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.asr.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.asr.wide.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 2 x i64>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.asr.wide.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 2 x i64>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.asr.wide.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.asrd.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, i32)
declare <vscale x 8 x i16> @llvm.aarch64.sve.asrd.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.asrd.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.asrd.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, i32)

declare <vscale x 16 x i8> @llvm.aarch64.sve.lsl.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.lsl.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.lsl.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.lsl.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.lsl.wide.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 2 x i64>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.lsl.wide.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 2 x i64>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.lsl.wide.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.lsr.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.lsr.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.lsr.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.lsr.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.lsr.wide.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 2 x i64>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.lsr.wide.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 2 x i64>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.lsr.wide.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 2 x i64>)
