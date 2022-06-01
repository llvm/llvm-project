; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve -asm-verbose=0 < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sme -asm-verbose=0 < %s | FileCheck %s

; NOTE: invalid, upper and lower bound immediate values of the reg+imm
; addressing mode are checked only for the byte version of each
; instruction (`st<N>b`), as the code for detecting the immediate is
; common to all instructions, and varies only for the number of
; elements of the structured store, which is <N> = 2, 3, 4.

;
; ST2B
;

define void @st2b_i8_valid_imm(<vscale x 16 x i8> %v0, <vscale x 16 x i8> %v1, <vscale x 16 x i1> %pred, <vscale x 16 x i8>* %addr) {
; CHECK-LABEL: st2b_i8_valid_imm:
; CHECK: st2b { z0.b, z1.b }, p0, [x0, #2, mul vl]
; CHECK-NEXT: ret
  %base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %addr, i64 2, i64 0
  call void @llvm.aarch64.sve.st2.nxv16i8(<vscale x 16 x i8> %v0,
                                          <vscale x 16 x i8> %v1,
                                          <vscale x 16 x i1> %pred,
                                          i8* %base)
  ret void
}

define void @st2b_i8_invalid_imm_not_multiple_of_2(<vscale x 16 x i8> %v0, <vscale x 16 x i8> %v1, <vscale x 16 x i1> %pred, <vscale x 16 x i8>* %addr) {
; CHECK-LABEL: st2b_i8_invalid_imm_not_multiple_of_2:
; CHECK: rdvl x[[N:[0-9]+]], #3
; CHECK-NEXT: st2b { z0.b, z1.b }, p0, [x0, x[[N]]]
; CHECK-NEXT: ret
  %base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %addr, i64 3, i64 0
  call void @llvm.aarch64.sve.st2.nxv16i8(<vscale x 16 x i8> %v0,
                                          <vscale x 16 x i8> %v1,
                                          <vscale x 16 x i1> %pred,
                                          i8* %base)
  ret void
}

define void @st2b_i8_invalid_imm_out_of_lower_bound(<vscale x 16 x i8> %v0, <vscale x 16 x i8> %v1, <vscale x 16 x i1> %pred, <vscale x 16 x i8>* %addr) {
; CHECK-LABEL: st2b_i8_invalid_imm_out_of_lower_bound:
; CHECK: rdvl x[[N:[0-9]+]], #-18
; CHECK-NEXT: st2b { z0.b, z1.b }, p0, [x0, x[[N]]]
; CHECK-NEXT: ret
  %base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %addr, i64 -18, i64 0
  call void @llvm.aarch64.sve.st2.nxv16i8(<vscale x 16 x i8> %v0,
                                          <vscale x 16 x i8> %v1,
                                          <vscale x 16 x i1> %pred,
                                          i8* %base)
  ret void
}

define void @st2b_i8_invalid_imm_out_of_upper_bound(<vscale x 16 x i8> %v0, <vscale x 16 x i8> %v1, <vscale x 16 x i1> %pred, <vscale x 16 x i8>* %addr) {
; CHECK-LABEL: st2b_i8_invalid_imm_out_of_upper_bound:
; CHECK: rdvl x[[N:[0-9]+]], #16
; CHECK-NEXT: st2b { z0.b, z1.b }, p0, [x0, x[[N]]]
; CHECK-NEXT: ret
  %base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %addr, i64 16, i64 0
  call void @llvm.aarch64.sve.st2.nxv16i8(<vscale x 16 x i8> %v0,
                                          <vscale x 16 x i8> %v1,
                                          <vscale x 16 x i1> %pred,
                                          i8* %base)
  ret void
}

define void @st2b_i8_valid_imm_lower_bound(<vscale x 16 x i8> %v0, <vscale x 16 x i8> %v1, <vscale x 16 x i1> %pred, <vscale x 16 x i8>* %addr) {
; CHECK-LABEL: st2b_i8_valid_imm_lower_bound:
; CHECK: st2b { z0.b, z1.b }, p0, [x0, #-16, mul vl]
; CHECK-NEXT: ret
  %base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %addr, i64 -16, i64 0
  call void @llvm.aarch64.sve.st2.nxv16i8(<vscale x 16 x i8> %v0,
                                          <vscale x 16 x i8> %v1,
                                          <vscale x 16 x i1> %pred,
                                          i8* %base)
  ret void
}

define void @st2b_i8_valid_imm_upper_bound(<vscale x 16 x i8> %v0, <vscale x 16 x i8> %v1, <vscale x 16 x i1> %pred, <vscale x 16 x i8>* %addr) {
; CHECK-LABEL: st2b_i8_valid_imm_upper_bound:
; CHECK: st2b { z0.b, z1.b }, p0, [x0, #14, mul vl]
; CHECK-NEXT: ret
  %base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %addr, i64 14, i64 0
  call void @llvm.aarch64.sve.st2.nxv16i8(<vscale x 16 x i8> %v0,
                                          <vscale x 16 x i8> %v1,
                                          <vscale x 16 x i1> %pred,
                                          i8* %base)
  ret void
}

;
; ST2H
;

define void @st2h_i16(<vscale x 8 x i16> %v0, <vscale x 8 x i16> %v1, <vscale x 8 x i1> %pred, <vscale x 8 x i16>* %addr) {
; CHECK-LABEL: st2h_i16:
; CHECK: st2h { z0.h, z1.h }, p0, [x0, #2, mul vl]
; CHECK-NEXT: ret
  %base = getelementptr <vscale x 8 x i16>, <vscale x 8 x i16>* %addr, i64 2, i64 0
  call void @llvm.aarch64.sve.st2.nxv8i16(<vscale x 8 x i16> %v0,
                                          <vscale x 8 x i16> %v1,
                                          <vscale x 8 x i1> %pred,
                                          i16* %base)
  ret void
}

define void @st2h_f16(<vscale x 8 x half> %v0, <vscale x 8 x half> %v1, <vscale x 8 x i1> %pred, <vscale x 8 x half>* %addr) {
; CHECK-LABEL: st2h_f16:
; CHECK: st2h { z0.h, z1.h }, p0, [x0, #2, mul vl]
; CHECK-NEXT: ret
  %base = getelementptr <vscale x 8 x half>, <vscale x 8 x half>* %addr, i64 2, i64 0
  call void @llvm.aarch64.sve.st2.nxv8f16(<vscale x 8 x half> %v0,
                                          <vscale x 8 x half> %v1,
                                          <vscale x 8 x i1> %pred,
                                          half* %base)
  ret void
}

;
; ST2W
;

define void @st2w_i32(<vscale x 4 x i32> %v0, <vscale x 4 x i32> %v1, <vscale x 4 x i1> %pred, <vscale x 4 x i32>* %addr) {
; CHECK-LABEL: st2w_i32:
; CHECK: st2w { z0.s, z1.s }, p0, [x0, #4, mul vl]
; CHECK-NEXT: ret
  %base = getelementptr <vscale x 4 x i32>, <vscale x 4 x i32>* %addr, i64 4, i64 0
  call void @llvm.aarch64.sve.st2.nxv4i32(<vscale x 4 x i32> %v0,
                                          <vscale x 4 x i32> %v1,
                                          <vscale x 4 x i1> %pred,
                                          i32* %base)
  ret void
}

define void @st2w_f32(<vscale x 4 x float> %v0, <vscale x 4 x float> %v1, <vscale x 4 x i1> %pred, <vscale x 4 x float>* %addr) {
; CHECK-LABEL: st2w_f32:
; CHECK: st2w { z0.s, z1.s }, p0, [x0, #6, mul vl]
; CHECK-NEXT: ret
  %base = getelementptr <vscale x 4 x float>, <vscale x 4 x float>* %addr, i64 6, i64 0
  call void @llvm.aarch64.sve.st2.nxv4f32(<vscale x 4 x float> %v0,
                                          <vscale x 4 x float> %v1,
                                          <vscale x 4 x i1> %pred,
                                          float* %base)
  ret void
}

;
; ST2D
;

define void @st2d_i64(<vscale x 2 x i64> %v0, <vscale x 2 x i64> %v1, <vscale x 2 x i1> %pred, <vscale x 2 x i64>* %addr) {
; CHECK-LABEL: st2d_i64:
; CHECK: st2d { z0.d, z1.d }, p0, [x0, #8, mul vl]
; CHECK-NEXT: ret
  %base = getelementptr <vscale x 2 x i64>, <vscale x 2 x i64>* %addr, i64 8, i64 0
  call void @llvm.aarch64.sve.st2.nxv2i64(<vscale x 2 x i64> %v0,
                                          <vscale x 2 x i64> %v1,
                                          <vscale x 2 x i1> %pred,
                                          i64* %base)
  ret void
}

define void @st2d_f64(<vscale x 2 x double> %v0, <vscale x 2 x double> %v1, <vscale x 2 x i1> %pred, <vscale x 2 x double>* %addr) {
; CHECK-LABEL: st2d_f64:
; CHECK: st2d { z0.d, z1.d }, p0, [x0, #10, mul vl]
; CHECK-NEXT: ret
  %base = getelementptr <vscale x 2 x double>, <vscale x 2 x double>* %addr, i64 10, i64 0
  call void @llvm.aarch64.sve.st2.nxv2f64(<vscale x 2 x double> %v0,
                                          <vscale x 2 x double> %v1,
                                          <vscale x 2 x i1> %pred,
                                          double* %base)
  ret void
}

;
; ST3B
;

define void @st3b_i8_valid_imm(<vscale x 16 x i8> %v0, <vscale x 16 x i8> %v1, <vscale x 16 x i8> %v2, <vscale x 16 x i1> %pred, <vscale x 16 x i8>* %addr) {
; CHECK-LABEL: st3b_i8_valid_imm:
; CHECK: st3b { z0.b, z1.b, z2.b }, p0, [x0, #3, mul vl]
; CHECK-NEXT: ret
  %base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %addr, i64 3, i64 0
  call void @llvm.aarch64.sve.st3.nxv16i8(<vscale x 16 x i8> %v0,
                                          <vscale x 16 x i8> %v1,
                                          <vscale x 16 x i8> %v2,
                                          <vscale x 16 x i1> %pred,
                                          i8* %base)
  ret void
}

define void @st3b_i8_invalid_imm_not_multiple_of_3_01(<vscale x 16 x i8> %v0, <vscale x 16 x i8> %v1, <vscale x 16 x i8> %v2, <vscale x 16 x i1> %pred, <vscale x 16 x i8>* %addr) {
; CHECK-LABEL: st3b_i8_invalid_imm_not_multiple_of_3_01:
; CHECK: rdvl x[[N:[0-9]+]], #4
; CHECK-NEXT: st3b { z0.b, z1.b, z2.b }, p0, [x0, x[[N]]]
; CHECK-NEXT: ret
  %base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %addr, i64 4, i64 0
  call void @llvm.aarch64.sve.st3.nxv16i8(<vscale x 16 x i8> %v0,
                                          <vscale x 16 x i8> %v1,
                                          <vscale x 16 x i8> %v2,
                                          <vscale x 16 x i1> %pred,
                                          i8* %base)
  ret void
}

define void @st3b_i8_invalid_imm_not_multiple_of_3_02(<vscale x 16 x i8> %v0, <vscale x 16 x i8> %v1, <vscale x 16 x i8> %v2, <vscale x 16 x i1> %pred, <vscale x 16 x i8>* %addr) {
; CHECK-LABEL: st3b_i8_invalid_imm_not_multiple_of_3_02:
; CHECK: rdvl x[[N:[0-9]+]], #5
; CHECK-NEXT: st3b { z0.b, z1.b, z2.b }, p0, [x0, x[[N]]]
; CHECK-NEXT: ret
  %base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %addr, i64 5, i64 0
  call void @llvm.aarch64.sve.st3.nxv16i8(<vscale x 16 x i8> %v0,
                                          <vscale x 16 x i8> %v1,
                                          <vscale x 16 x i8> %v2,
                                          <vscale x 16 x i1> %pred,
                                          i8* %base)
  ret void
}

define void @st3b_i8_invalid_imm_out_of_lower_bound(<vscale x 16 x i8> %v0, <vscale x 16 x i8> %v1, <vscale x 16 x i8> %v2, <vscale x 16 x i1> %pred, <vscale x 16 x i8>* %addr) {
; CHECK-LABEL: st3b_i8_invalid_imm_out_of_lower_bound:
; CHECK: rdvl x[[N:[0-9]+]], #-27
; CHECK-NEXT: st3b { z0.b, z1.b, z2.b }, p0, [x0, x[[N]]]
; CHECK-NEXT: ret
  %base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %addr, i64 -27, i64 0
  call void @llvm.aarch64.sve.st3.nxv16i8(<vscale x 16 x i8> %v0,
                                          <vscale x 16 x i8> %v1,
                                          <vscale x 16 x i8> %v2,
                                          <vscale x 16 x i1> %pred,
                                          i8* %base)
  ret void
}

define void @st3b_i8_invalid_imm_out_of_upper_bound(<vscale x 16 x i8> %v0, <vscale x 16 x i8> %v1, <vscale x 16 x i8> %v2, <vscale x 16 x i1> %pred, <vscale x 16 x i8>* %addr) {
; CHECK-LABEL: st3b_i8_invalid_imm_out_of_upper_bound:
; CHECK: rdvl x[[N:[0-9]+]], #24
; CHECK-NEXT: st3b { z0.b, z1.b, z2.b }, p0, [x0, x[[N]]]
; CHECK-NEXT: ret
  %base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %addr, i64 24, i64 0
  call void @llvm.aarch64.sve.st3.nxv16i8(<vscale x 16 x i8> %v0,
                                          <vscale x 16 x i8> %v1,
                                          <vscale x 16 x i8> %v2,
                                          <vscale x 16 x i1> %pred,
                                          i8* %base)
  ret void
}

define void @st3b_i8_valid_imm_lower_bound(<vscale x 16 x i8> %v0, <vscale x 16 x i8> %v1, <vscale x 16 x i8> %v2, <vscale x 16 x i1> %pred, <vscale x 16 x i8>* %addr) {
; CHECK-LABEL: st3b_i8_valid_imm_lower_bound:
; CHECK: st3b { z0.b, z1.b, z2.b }, p0, [x0, #-24, mul vl]
; CHECK-NEXT: ret
  %base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %addr, i64 -24, i64 0
  call void @llvm.aarch64.sve.st3.nxv16i8(<vscale x 16 x i8> %v0,
                                          <vscale x 16 x i8> %v1,
                                          <vscale x 16 x i8> %v2,
                                          <vscale x 16 x i1> %pred,
                                          i8* %base)
  ret void
}

define void @st3b_i8_valid_imm_upper_bound(<vscale x 16 x i8> %v0, <vscale x 16 x i8> %v1, <vscale x 16 x i8> %v2, <vscale x 16 x i1> %pred, <vscale x 16 x i8>* %addr) {
; CHECK-LABEL: st3b_i8_valid_imm_upper_bound:
; CHECK: st3b { z0.b, z1.b, z2.b }, p0, [x0, #21, mul vl]
; CHECK-NEXT: ret
  %base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %addr, i64 21, i64 0
  call void @llvm.aarch64.sve.st3.nxv16i8(<vscale x 16 x i8> %v0,
                                          <vscale x 16 x i8> %v1,
                                          <vscale x 16 x i8> %v2,
                                          <vscale x 16 x i1> %pred,
                                          i8* %base)
  ret void
}

;
; ST3H
;

define void @st3h_i16(<vscale x 8 x i16> %v0, <vscale x 8 x i16> %v1, <vscale x 8 x i16> %v2, <vscale x 8 x i1> %pred, <vscale x 8 x i16>* %addr) {
; CHECK-LABEL: st3h_i16:
; CHECK: st3h { z0.h, z1.h, z2.h }, p0, [x0, #6, mul vl]
; CHECK-NEXT: ret
  %base = getelementptr <vscale x 8 x i16>, <vscale x 8 x i16>* %addr, i64 6, i64 0
  call void @llvm.aarch64.sve.st3.nxv8i16(<vscale x 8 x i16> %v0,
                                          <vscale x 8 x i16> %v1,
                                          <vscale x 8 x i16> %v2,
                                          <vscale x 8 x i1> %pred,
                                          i16* %base)
  ret void
}

define void @st3h_f16(<vscale x 8 x half> %v0, <vscale x 8 x half> %v1, <vscale x 8 x half> %v2, <vscale x 8 x i1> %pred, <vscale x 8 x half>* %addr) {
; CHECK-LABEL: st3h_f16:
; CHECK: st3h { z0.h, z1.h, z2.h }, p0, [x0, #9, mul vl]
; CHECK-NEXT: ret
  %base = getelementptr <vscale x 8 x half>, <vscale x 8 x half>* %addr, i64 9, i64 0
  call void @llvm.aarch64.sve.st3.nxv8f16(<vscale x 8 x half> %v0,
                                          <vscale x 8 x half> %v1,
                                          <vscale x 8 x half> %v2,
                                          <vscale x 8 x i1> %pred,
                                          half* %base)
  ret void
}

;
; ST3W
;

define void @st3w_i32(<vscale x 4 x i32> %v0, <vscale x 4 x i32> %v1, <vscale x 4 x i32> %v2, <vscale x 4 x i1> %pred, <vscale x 4 x i32>* %addr) {
; CHECK-LABEL: st3w_i32:
; CHECK: st3w { z0.s, z1.s, z2.s }, p0, [x0, #12, mul vl]
; CHECK-NEXT: ret
  %base = getelementptr <vscale x 4 x i32>, <vscale x 4 x i32>* %addr, i64 12, i64 0
  call void @llvm.aarch64.sve.st3.nxv4i32(<vscale x 4 x i32> %v0,
                                          <vscale x 4 x i32> %v1,
                                          <vscale x 4 x i32> %v2,
                                          <vscale x 4 x i1> %pred,
                                          i32* %base)
  ret void
}

define void @st3w_f32(<vscale x 4 x float> %v0, <vscale x 4 x float> %v1, <vscale x 4 x float> %v2, <vscale x 4 x i1> %pred, <vscale x 4 x float>* %addr) {
; CHECK-LABEL: st3w_f32:
; CHECK: st3w { z0.s, z1.s, z2.s }, p0, [x0, #15, mul vl]
; CHECK-NEXT: ret
  %base = getelementptr <vscale x 4 x float>, <vscale x 4 x float>* %addr, i64 15, i64 0
  call void @llvm.aarch64.sve.st3.nxv4f32(<vscale x 4 x float> %v0,
                                          <vscale x 4 x float> %v1,
                                          <vscale x 4 x float> %v2,
                                          <vscale x 4 x i1> %pred,
                                          float* %base)
  ret void
}

;
; ST3D
;

define void @st3d_i64(<vscale x 2 x i64> %v0, <vscale x 2 x i64> %v1, <vscale x 2 x i64> %v2, <vscale x 2 x i1> %pred, <vscale x 2 x i64>* %addr) {
; CHECK-LABEL: st3d_i64:
; CHECK: st3d { z0.d, z1.d, z2.d }, p0, [x0, #18, mul vl]
; CHECK-NEXT: ret
  %base = getelementptr <vscale x 2 x i64>, <vscale x 2 x i64>* %addr, i64 18, i64 0
  call void @llvm.aarch64.sve.st3.nxv2i64(<vscale x 2 x i64> %v0,
                                          <vscale x 2 x i64> %v1,
                                          <vscale x 2 x i64> %v2,
                                          <vscale x 2 x i1> %pred,
                                          i64* %base)
  ret void
}

define void @st3d_f64(<vscale x 2 x double> %v0, <vscale x 2 x double> %v1, <vscale x 2 x double> %v2, <vscale x 2 x i1> %pred, <vscale x 2 x double>* %addr) {
; CHECK-LABEL: st3d_f64:
; CHECK: st3d { z0.d, z1.d, z2.d }, p0, [x0, #-3, mul vl]
; CHECK-NEXT: ret
  %base = getelementptr <vscale x 2 x double>, <vscale x 2 x double>* %addr, i64 -3, i64 0
  call void @llvm.aarch64.sve.st3.nxv2f64(<vscale x 2 x double> %v0,
                                          <vscale x 2 x double> %v1,
                                          <vscale x 2 x double> %v2,
                                          <vscale x 2 x i1> %pred,
                                          double* %base)
  ret void
}

;
; ST4B
;

define void @st4b_i8_valid_imm(<vscale x 16 x i8> %v0, <vscale x 16 x i8> %v1, <vscale x 16 x i8> %v2, <vscale x 16 x i8> %v3, <vscale x 16 x i1> %pred, <vscale x 16 x i8>* %addr) {
; CHECK-LABEL: st4b_i8_valid_imm:
; CHECK: st4b { z0.b, z1.b, z2.b, z3.b }, p0, [x0, #4, mul vl]
; CHECK-NEXT: ret
  %base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %addr, i64 4, i64 0
  call void @llvm.aarch64.sve.st4.nxv16i8(<vscale x 16 x i8> %v0,
                                          <vscale x 16 x i8> %v1,
                                          <vscale x 16 x i8> %v2,
                                          <vscale x 16 x i8> %v3,
                                          <vscale x 16 x i1> %pred,
                                          i8* %base)
  ret void
}

define void @st4b_i8_invalid_imm_not_multiple_of_4_01(<vscale x 16 x i8> %v0, <vscale x 16 x i8> %v1, <vscale x 16 x i8> %v2, <vscale x 16 x i8> %v3, <vscale x 16 x i1> %pred, <vscale x 16 x i8>* %addr) {
; CHECK-LABEL: st4b_i8_invalid_imm_not_multiple_of_4_01:
; CHECK: rdvl x[[N:[0-9]+]], #5
; CHECK-NEXT: st4b { z0.b, z1.b, z2.b, z3.b }, p0, [x0, x[[N]]]
; CHECK-NEXT: ret
  %base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %addr, i64 5, i64 0
  call void @llvm.aarch64.sve.st4.nxv16i8(<vscale x 16 x i8> %v0,
                                          <vscale x 16 x i8> %v1,
                                          <vscale x 16 x i8> %v2,
                                          <vscale x 16 x i8> %v3,
                                          <vscale x 16 x i1> %pred,
                                          i8* %base)
  ret void
}

define void @st4b_i8_invalid_imm_not_multiple_of_4_02(<vscale x 16 x i8> %v0, <vscale x 16 x i8> %v1, <vscale x 16 x i8> %v2, <vscale x 16 x i8> %v3, <vscale x 16 x i1> %pred, <vscale x 16 x i8>* %addr) {
; CHECK-LABEL: st4b_i8_invalid_imm_not_multiple_of_4_02:
; CHECK: rdvl x[[N:[0-9]+]], #6
; CHECK-NEXT: st4b { z0.b, z1.b, z2.b, z3.b }, p0, [x0, x[[N]]]
; CHECK-NEXT: ret
  %base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %addr, i64 6, i64 0
  call void @llvm.aarch64.sve.st4.nxv16i8(<vscale x 16 x i8> %v0,
                                          <vscale x 16 x i8> %v1,
                                          <vscale x 16 x i8> %v2,
                                          <vscale x 16 x i8> %v3,
                                          <vscale x 16 x i1> %pred,
                                          i8* %base)
  ret void
}

define void @st4b_i8_invalid_imm_not_multiple_of_4_03(<vscale x 16 x i8> %v0, <vscale x 16 x i8> %v1, <vscale x 16 x i8> %v2, <vscale x 16 x i8> %v3, <vscale x 16 x i1> %pred, <vscale x 16 x i8>* %addr) {
; CHECK-LABEL: st4b_i8_invalid_imm_not_multiple_of_4_03:
; CHECK: rdvl x[[N:[0-9]+]], #7
; CHECK-NEXT: st4b { z0.b, z1.b, z2.b, z3.b }, p0, [x0, x[[N]]]
; CHECK-NEXT: ret
  %base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %addr, i64 7, i64 0
  call void @llvm.aarch64.sve.st4.nxv16i8(<vscale x 16 x i8> %v0,
                                          <vscale x 16 x i8> %v1,
                                          <vscale x 16 x i8> %v2,
                                          <vscale x 16 x i8> %v3,
                                          <vscale x 16 x i1> %pred,
                                          i8* %base)
  ret void
}

define void @st4b_i8_invalid_imm_out_of_lower_bound(<vscale x 16 x i8> %v0, <vscale x 16 x i8> %v1, <vscale x 16 x i8> %v2, <vscale x 16 x i8> %v3, <vscale x 16 x i1> %pred, <vscale x 16 x i8>* %addr) {
; CHECK-LABEL: st4b_i8_invalid_imm_out_of_lower_bound:
; FIXME: optimize OFFSET computation so that xOFFSET = (mul (RDVL #4) #9)
; xM = -9 * 2^6
; xP = RDVL * 2^-4
; xBASE = RDVL * 2^-4 * -9 * 2^6 = RDVL * -36
; CHECK: rdvl x[[N:[0-9]+]], #1
; CHECK-DAG:  mov  x[[M:[0-9]+]], #-576
; CHECK-DAG:  lsr  x[[P:[0-9]+]], x[[N]], #4
; CHECK-DAG:  mul  x[[OFFSET:[0-9]+]], x[[P]], x[[M]]
; CHECK-NEXT: st4b { z0.b, z1.b, z2.b, z3.b }, p0, [x0, x[[OFFSET]]]
; CHECK-NEXT: ret
  %base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %addr, i64 -36, i64 0
  call void @llvm.aarch64.sve.st4.nxv16i8(<vscale x 16 x i8> %v0,
                                          <vscale x 16 x i8> %v1,
                                          <vscale x 16 x i8> %v2,
                                          <vscale x 16 x i8> %v3,
                                          <vscale x 16 x i1> %pred,
                                          i8* %base)
  ret void
}

define void @st4b_i8_invalid_imm_out_of_upper_bound(<vscale x 16 x i8> %v0, <vscale x 16 x i8> %v1, <vscale x 16 x i8> %v2, <vscale x 16 x i8> %v3, <vscale x 16 x i1> %pred, <vscale x 16 x i8>* %addr) {
; CHECK-LABEL: st4b_i8_invalid_imm_out_of_upper_bound:
; FIXME: optimize OFFSET computation so that xOFFSET = (shl (RDVL #16) #1)
; xM = 2^9
; xP = RDVL * 2^-4
; xOFFSET = RDVL * 2^-4 * 2^9 = RDVL * 32
; CHECK: rdvl x[[N:[0-9]+]], #1
; CHECK-DAG:  mov  w[[M:[0-9]+]], #512
; CHECK-DAG:  lsr  x[[P:[0-9]+]], x[[N]], #4
; CHECK-DAG:  mul  x[[OFFSET:[0-9]+]], x[[P]], x[[M]]
; CHECK-NEXT: st4b { z0.b, z1.b, z2.b, z3.b }, p0, [x0, x[[OFFSET]]]
; CHECK-NEXT: ret
  %base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %addr, i64 32, i64 0
  call void @llvm.aarch64.sve.st4.nxv16i8(<vscale x 16 x i8> %v0,
                                          <vscale x 16 x i8> %v1,
                                          <vscale x 16 x i8> %v2,
                                          <vscale x 16 x i8> %v3,
                                          <vscale x 16 x i1> %pred,
                                          i8* %base)
  ret void
}

define void @st4b_i8_valid_imm_lower_bound(<vscale x 16 x i8> %v0, <vscale x 16 x i8> %v1, <vscale x 16 x i8> %v2, <vscale x 16 x i8> %v3, <vscale x 16 x i1> %pred, <vscale x 16 x i8>* %addr) {
; CHECK-LABEL: st4b_i8_valid_imm_lower_bound:
; CHECK: st4b { z0.b, z1.b, z2.b, z3.b }, p0, [x0, #-32, mul vl]
; CHECK-NEXT: ret
  %base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %addr, i64 -32, i64 0
  call void @llvm.aarch64.sve.st4.nxv16i8(<vscale x 16 x i8> %v0,
                                          <vscale x 16 x i8> %v1,
                                          <vscale x 16 x i8> %v2,
                                          <vscale x 16 x i8> %v3,
                                          <vscale x 16 x i1> %pred,
                                          i8* %base)
  ret void
}

define void @st4b_i8_valid_imm_upper_bound(<vscale x 16 x i8> %v0, <vscale x 16 x i8> %v1, <vscale x 16 x i8> %v2, <vscale x 16 x i8> %v3, <vscale x 16 x i1> %pred, <vscale x 16 x i8>* %addr) {
; CHECK-LABEL: st4b_i8_valid_imm_upper_bound:
; CHECK: st4b { z0.b, z1.b, z2.b, z3.b }, p0, [x0, #28, mul vl]
; CHECK-NEXT: ret
  %base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %addr, i64 28, i64 0
  call void @llvm.aarch64.sve.st4.nxv16i8(<vscale x 16 x i8> %v0,
                                          <vscale x 16 x i8> %v1,
                                          <vscale x 16 x i8> %v2,
                                          <vscale x 16 x i8> %v3,
                                          <vscale x 16 x i1> %pred,
                                          i8* %base)
  ret void
}

;
; ST4H
;

define void @st4h_i16(<vscale x 8 x i16> %v0, <vscale x 8 x i16> %v1, <vscale x 8 x i16> %v2, <vscale x 8 x i16> %v3, <vscale x 8 x i1> %pred, <vscale x 8 x i16>* %addr) {
; CHECK-LABEL: st4h_i16:
; CHECK: st4h { z0.h, z1.h, z2.h, z3.h }, p0, [x0, #8, mul vl]
; CHECK-NEXT: ret
  %base = getelementptr <vscale x 8 x i16>, <vscale x 8 x i16>* %addr, i64 8, i64 0
  call void @llvm.aarch64.sve.st4.nxv8i16(<vscale x 8 x i16> %v0,
                                          <vscale x 8 x i16> %v1,
                                          <vscale x 8 x i16> %v2,
                                          <vscale x 8 x i16> %v3,
                                          <vscale x 8 x i1> %pred,
                                          i16* %base)
  ret void
}

define void @st4h_f16(<vscale x 8 x half> %v0, <vscale x 8 x half> %v1, <vscale x 8 x half> %v2, <vscale x 8 x half> %v3, <vscale x 8 x i1> %pred, <vscale x 8 x half>* %addr) {
; CHECK-LABEL: st4h_f16:
; CHECK: st4h { z0.h, z1.h, z2.h, z3.h }, p0, [x0, #12, mul vl]
; CHECK-NEXT: ret
  %base = getelementptr <vscale x 8 x half>, <vscale x 8 x half>* %addr, i64 12, i64 0
  call void @llvm.aarch64.sve.st4.nxv8f16(<vscale x 8 x half> %v0,
                                          <vscale x 8 x half> %v1,
                                          <vscale x 8 x half> %v2,
                                          <vscale x 8 x half> %v3,
                                          <vscale x 8 x i1> %pred,
                                          half* %base)
  ret void
}

;
; ST4W
;

define void @st4w_i32(<vscale x 4 x i32> %v0, <vscale x 4 x i32> %v1, <vscale x 4 x i32> %v2, <vscale x 4 x i32> %v3, <vscale x 4 x i1> %pred, <vscale x 4 x i32>* %addr) {
; CHECK-LABEL: st4w_i32:
; CHECK: st4w { z0.s, z1.s, z2.s, z3.s }, p0, [x0, #16, mul vl]
; CHECK-NEXT: ret
  %base = getelementptr <vscale x 4 x i32>, <vscale x 4 x i32>* %addr, i64 16, i64 0
  call void @llvm.aarch64.sve.st4.nxv4i32(<vscale x 4 x i32> %v0,
                                          <vscale x 4 x i32> %v1,
                                          <vscale x 4 x i32> %v2,
                                          <vscale x 4 x i32> %v3,
                                          <vscale x 4 x i1> %pred,
                                          i32* %base)
  ret void
}

define void @st4w_f32(<vscale x 4 x float> %v0, <vscale x 4 x float> %v1, <vscale x 4 x float> %v2, <vscale x 4 x float> %v3, <vscale x 4 x i1> %pred, <vscale x 4 x float>* %addr) {
; CHECK-LABEL: st4w_f32:
; CHECK: st4w { z0.s, z1.s, z2.s, z3.s }, p0, [x0, #20, mul vl]
; CHECK-NEXT: ret
  %base = getelementptr <vscale x 4 x float>, <vscale x 4 x float>* %addr, i64 20, i64 0
  call void @llvm.aarch64.sve.st4.nxv4f32(<vscale x 4 x float> %v0,
                                          <vscale x 4 x float> %v1,
                                          <vscale x 4 x float> %v2,
                                          <vscale x 4 x float> %v3,
                                          <vscale x 4 x i1> %pred,
                                          float* %base)
  ret void
}

;
; ST4D
;

define void @st4d_i64(<vscale x 2 x i64> %v0, <vscale x 2 x i64> %v1, <vscale x 2 x i64> %v2, <vscale x 2 x i64> %v3, <vscale x 2 x i1> %pred, <vscale x 2 x i64>* %addr) {
; CHECK-LABEL: st4d_i64:
; CHECK: st4d { z0.d, z1.d, z2.d, z3.d }, p0, [x0, #24, mul vl]
; CHECK-NEXT: ret
  %base = getelementptr <vscale x 2 x i64>, <vscale x 2 x i64>* %addr, i64 24, i64 0
  call void @llvm.aarch64.sve.st4.nxv2i64(<vscale x 2 x i64> %v0,
                                          <vscale x 2 x i64> %v1,
                                          <vscale x 2 x i64> %v2,
                                          <vscale x 2 x i64> %v3,
                                          <vscale x 2 x i1> %pred,
                                          i64* %base)
  ret void
}

define void @st4d_f64(<vscale x 2 x double> %v0, <vscale x 2 x double> %v1, <vscale x 2 x double> %v2, <vscale x 2 x double> %v3, <vscale x 2 x i1> %pred, <vscale x 2 x double>* %addr) {
; CHECK-LABEL: st4d_f64:
; CHECK: st4d { z0.d, z1.d, z2.d, z3.d }, p0, [x0, #28, mul vl]
; CHECK-NEXT: ret
  %base = getelementptr <vscale x 2 x double>, <vscale x 2 x double>* %addr, i64 28, i64 0
  call void @llvm.aarch64.sve.st4.nxv2f64(<vscale x 2 x double> %v0,
                                          <vscale x 2 x double> %v1,
                                          <vscale x 2 x double> %v2,
                                          <vscale x 2 x double> %v3,
                                          <vscale x 2 x i1> %pred,
                                          double* %base)
  ret void
}

declare void @llvm.aarch64.sve.st2.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i1>, i8*)
declare void @llvm.aarch64.sve.st2.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i1>, i16*)
declare void @llvm.aarch64.sve.st2.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i1>, i32*)
declare void @llvm.aarch64.sve.st2.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i1>, i64*)
declare void @llvm.aarch64.sve.st2.nxv8f16(<vscale x 8 x half>, <vscale x 8 x half>, <vscale x 8 x i1>, half*)
declare void @llvm.aarch64.sve.st2.nxv4f32(<vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x i1>, float*)
declare void @llvm.aarch64.sve.st2.nxv2f64(<vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x i1>, double*)

declare void @llvm.aarch64.sve.st3.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i1>, i8*)
declare void @llvm.aarch64.sve.st3.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i1>, i16*)
declare void @llvm.aarch64.sve.st3.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i1>, i32*)
declare void @llvm.aarch64.sve.st3.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i1>, i64*)
declare void @llvm.aarch64.sve.st3.nxv8f16(<vscale x 8 x half>, <vscale x 8 x half>, <vscale x 8 x half>, <vscale x 8 x i1>, half*)
declare void @llvm.aarch64.sve.st3.nxv4f32(<vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x i1>, float*)
declare void @llvm.aarch64.sve.st3.nxv2f64(<vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x i1>, double*)

declare void @llvm.aarch64.sve.st4.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i1>, i8*)
declare void @llvm.aarch64.sve.st4.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i1>, i16*)
declare void @llvm.aarch64.sve.st4.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i1>, i32*)
declare void @llvm.aarch64.sve.st4.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i1>, i64*)
declare void @llvm.aarch64.sve.st4.nxv8f16(<vscale x 8 x half>, <vscale x 8 x half>, <vscale x 8 x half>, <vscale x 8 x half>, <vscale x 8 x i1>, half*)
declare void @llvm.aarch64.sve.st4.nxv4f32(<vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x i1>, float*)
declare void @llvm.aarch64.sve.st4.nxv2f64(<vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x i1>, double*)
