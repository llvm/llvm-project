; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve -asm-verbose=0 < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; WARN-NOT: warning

;
; ST2B
;

define void @st2b_i8(<vscale x 16 x i8> %v0, <vscale x 16 x i8> %v1, <vscale x 16 x i1> %pred, i8* %addr, i64 %offset) {
; CHECK-LABEL: st2b_i8:
; CHECK: st2b { z0.b, z1.b }, p0, [x0, x1]
; CHECK-NEXT: ret
  %1 = getelementptr i8, i8* %addr, i64 %offset
  call void @llvm.aarch64.sve.st2.nxv16i8(<vscale x 16 x i8> %v0,
                                          <vscale x 16 x i8> %v1,
                                          <vscale x 16 x i1> %pred,
                                          i8* %1)
  ret void
}

;
; ST2H
;

define void @st2h_i16(<vscale x 8 x i16> %v0, <vscale x 8 x i16> %v1, <vscale x 8 x i1> %pred, i16* %addr, i64 %offset) {
; CHECK-LABEL: st2h_i16:
; CHECK: st2h { z0.h, z1.h }, p0, [x0, x1, lsl #1]
; CHECK-NEXT: ret
  %1 = getelementptr i16, i16* %addr, i64 %offset
  call void @llvm.aarch64.sve.st2.nxv8i16(<vscale x 8 x i16> %v0,
                                          <vscale x 8 x i16> %v1,
                                          <vscale x 8 x i1> %pred,
                                          i16* %1)
  ret void
}

define void @st2h_f16(<vscale x 8 x half> %v0, <vscale x 8 x half> %v1, <vscale x 8 x i1> %pred, half* %addr, i64 %offset) {
; CHECK-LABEL: st2h_f16:
; CHECK: st2h { z0.h, z1.h }, p0, [x0, x1, lsl #1]
; CHECK-NEXT: ret
  %1 = getelementptr half, half* %addr, i64 %offset
  call void @llvm.aarch64.sve.st2.nxv8f16(<vscale x 8 x half> %v0,
                                          <vscale x 8 x half> %v1,
                                          <vscale x 8 x i1> %pred,
                                          half* %1)
  ret void
}

;
; ST2W
;

define void @st2w_i32(<vscale x 4 x i32> %v0, <vscale x 4 x i32> %v1, <vscale x 4 x i1> %pred, i32* %addr, i64 %offset) {
; CHECK-LABEL: st2w_i32:
; CHECK: st2w { z0.s, z1.s }, p0, [x0, x1, lsl #2]
; CHECK-NEXT: ret
  %1 = getelementptr i32, i32* %addr, i64 %offset
  call void @llvm.aarch64.sve.st2.nxv4i32(<vscale x 4 x i32> %v0,
                                          <vscale x 4 x i32> %v1,
                                          <vscale x 4 x i1> %pred,
                                          i32* %1)
  ret void
}

define void @st2w_f32(<vscale x 4 x float> %v0, <vscale x 4 x float> %v1, <vscale x 4 x i1> %pred, float* %addr, i64 %offset) {
; CHECK-LABEL: st2w_f32:
; CHECK: st2w { z0.s, z1.s }, p0, [x0, x1, lsl #2]
; CHECK-NEXT: ret
  %1 = getelementptr float, float* %addr, i64 %offset
  call void @llvm.aarch64.sve.st2.nxv4f32(<vscale x 4 x float> %v0,
                                          <vscale x 4 x float> %v1,
                                          <vscale x 4 x i1> %pred,
                                          float* %1)
  ret void
}

;
; ST2D
;

define void @st2d_i64(<vscale x 2 x i64> %v0, <vscale x 2 x i64> %v1, <vscale x 2 x i1> %pred, i64* %addr, i64 %offset) {
; CHECK-LABEL: st2d_i64:
; CHECK: st2d { z0.d, z1.d }, p0, [x0, x1, lsl #3]
; CHECK-NEXT: ret
  %1 = getelementptr i64, i64* %addr, i64 %offset
  call void @llvm.aarch64.sve.st2.nxv2i64(<vscale x 2 x i64> %v0,
                                          <vscale x 2 x i64> %v1,
                                          <vscale x 2 x i1> %pred,
                                          i64* %1)
  ret void
}

define void @st2d_f64(<vscale x 2 x double> %v0, <vscale x 2 x double> %v1, <vscale x 2 x i1> %pred, double* %addr, i64 %offset) {
; CHECK-LABEL: st2d_f64:
; CHECK: st2d { z0.d, z1.d }, p0, [x0, x1, lsl #3]
; CHECK-NEXT: ret
  %1 = getelementptr double, double* %addr, i64 %offset
  call void @llvm.aarch64.sve.st2.nxv2f64(<vscale x 2 x double> %v0,
                                          <vscale x 2 x double> %v1,
                                          <vscale x 2 x i1> %pred,
                                          double* %1)
  ret void
}

;
; ST3B
;

define void @st3b_i8(<vscale x 16 x i8> %v0, <vscale x 16 x i8> %v1, <vscale x 16 x i8> %v2, <vscale x 16 x i1> %pred, i8* %addr, i64 %offset) {
; CHECK-LABEL: st3b_i8:
; CHECK: st3b { z0.b, z1.b, z2.b }, p0, [x0, x1]
; CHECK-NEXT: ret
  %1 = getelementptr i8, i8* %addr, i64 %offset
  call void @llvm.aarch64.sve.st3.nxv16i8(<vscale x 16 x i8> %v0,
                                          <vscale x 16 x i8> %v1,
                                          <vscale x 16 x i8> %v2,
                                          <vscale x 16 x i1> %pred,
                                          i8* %1)
  ret void
}

;
; ST3H
;

define void @st3h_i16(<vscale x 8 x i16> %v0, <vscale x 8 x i16> %v1, <vscale x 8 x i16> %v2, <vscale x 8 x i1> %pred, i16* %addr, i64 %offset) {
; CHECK-LABEL: st3h_i16:
; CHECK: st3h { z0.h, z1.h, z2.h }, p0, [x0, x1, lsl #1]
; CHECK-NEXT: ret
  %1 = getelementptr i16, i16* %addr, i64 %offset
  call void @llvm.aarch64.sve.st3.nxv8i16(<vscale x 8 x i16> %v0,
                                          <vscale x 8 x i16> %v1,
                                          <vscale x 8 x i16> %v2,
                                          <vscale x 8 x i1> %pred,
                                          i16* %1)
  ret void
}

define void @st3h_f16(<vscale x 8 x half> %v0, <vscale x 8 x half> %v1, <vscale x 8 x half> %v2, <vscale x 8 x i1> %pred, half* %addr, i64 %offset) {
; CHECK-LABEL: st3h_f16:
; CHECK: st3h { z0.h, z1.h, z2.h }, p0, [x0, x1, lsl #1]
; CHECK-NEXT: ret
  %1 = getelementptr half, half* %addr, i64 %offset
  call void @llvm.aarch64.sve.st3.nxv8f16(<vscale x 8 x half> %v0,
                                          <vscale x 8 x half> %v1,
                                          <vscale x 8 x half> %v2,
                                          <vscale x 8 x i1> %pred,
                                          half* %1)
  ret void
}

;
; ST3W
;

define void @st3w_i32(<vscale x 4 x i32> %v0, <vscale x 4 x i32> %v1, <vscale x 4 x i32> %v2, <vscale x 4 x i1> %pred, i32* %addr, i64 %offset) {
; CHECK-LABEL: st3w_i32:
; CHECK: st3w { z0.s, z1.s, z2.s }, p0, [x0, x1, lsl #2]
; CHECK-NEXT: ret
  %1 = getelementptr i32, i32* %addr, i64 %offset
  call void @llvm.aarch64.sve.st3.nxv4i32(<vscale x 4 x i32> %v0,
                                          <vscale x 4 x i32> %v1,
                                          <vscale x 4 x i32> %v2,
                                          <vscale x 4 x i1> %pred,
                                          i32* %1)
  ret void
}

define void @st3w_f32(<vscale x 4 x float> %v0, <vscale x 4 x float> %v1, <vscale x 4 x float> %v2, <vscale x 4 x i1> %pred, float* %addr, i64 %offset) {
; CHECK-LABEL: st3w_f32:
; CHECK: st3w { z0.s, z1.s, z2.s }, p0, [x0, x1, lsl #2]
; CHECK-NEXT: ret
  %1 = getelementptr float, float* %addr, i64 %offset
  call void @llvm.aarch64.sve.st3.nxv4f32(<vscale x 4 x float> %v0,
                                          <vscale x 4 x float> %v1,
                                          <vscale x 4 x float> %v2,
                                          <vscale x 4 x i1> %pred,
                                          float* %1)
  ret void
}

;
; ST3D
;

define void @st3d_i64(<vscale x 2 x i64> %v0, <vscale x 2 x i64> %v1, <vscale x 2 x i64> %v2, <vscale x 2 x i1> %pred, i64* %addr, i64 %offset) {
; CHECK-LABEL: st3d_i64:
; CHECK: st3d { z0.d, z1.d, z2.d }, p0, [x0, x1, lsl #3]
; CHECK-NEXT: ret
  %1 = getelementptr i64, i64* %addr, i64 %offset
  call void @llvm.aarch64.sve.st3.nxv2i64(<vscale x 2 x i64> %v0,
                                          <vscale x 2 x i64> %v1,
                                          <vscale x 2 x i64> %v2,
                                          <vscale x 2 x i1> %pred,
                                          i64* %1)
  ret void
}

define void @st3d_f64(<vscale x 2 x double> %v0, <vscale x 2 x double> %v1, <vscale x 2 x double> %v2, <vscale x 2 x i1> %pred, double* %addr, i64 %offset) {
; CHECK-LABEL: st3d_f64:
; CHECK: st3d { z0.d, z1.d, z2.d }, p0, [x0, x1, lsl #3]
; CHECK-NEXT: ret
  %1 = getelementptr double, double* %addr, i64 %offset
  call void @llvm.aarch64.sve.st3.nxv2f64(<vscale x 2 x double> %v0,
                                          <vscale x 2 x double> %v1,
                                          <vscale x 2 x double> %v2,
                                          <vscale x 2 x i1> %pred,
                                          double* %1)
  ret void
}

;
; ST4B
;

define void @st4b_i8(<vscale x 16 x i8> %v0, <vscale x 16 x i8> %v1, <vscale x 16 x i8> %v2, <vscale x 16 x i8> %v3, <vscale x 16 x i1> %pred, i8* %addr, i64 %offset) {
; CHECK-LABEL: st4b_i8:
; CHECK: st4b { z0.b, z1.b, z2.b, z3.b }, p0, [x0, x1]
; CHECK-NEXT: ret
  %1 = getelementptr i8, i8* %addr, i64 %offset
  call void @llvm.aarch64.sve.st4.nxv16i8(<vscale x 16 x i8> %v0,
                                          <vscale x 16 x i8> %v1,
                                          <vscale x 16 x i8> %v2,
                                          <vscale x 16 x i8> %v3,
                                          <vscale x 16 x i1> %pred,
                                          i8* %1)
  ret void
}

;
; ST4H
;

define void @st4h_i16(<vscale x 8 x i16> %v0, <vscale x 8 x i16> %v1, <vscale x 8 x i16> %v2, <vscale x 8 x i16> %v3, <vscale x 8 x i1> %pred, i16* %addr, i64 %offset) {
; CHECK-LABEL: st4h_i16:
; CHECK: st4h { z0.h, z1.h, z2.h, z3.h }, p0, [x0, x1, lsl #1]
; CHECK-NEXT: ret
  %1 = getelementptr i16, i16* %addr, i64 %offset
  call void @llvm.aarch64.sve.st4.nxv8i16(<vscale x 8 x i16> %v0,
                                          <vscale x 8 x i16> %v1,
                                          <vscale x 8 x i16> %v2,
                                          <vscale x 8 x i16> %v3,
                                          <vscale x 8 x i1> %pred,
                                          i16* %1)
  ret void
}

define void @st4h_f16(<vscale x 8 x half> %v0, <vscale x 8 x half> %v1, <vscale x 8 x half> %v2, <vscale x 8 x half> %v3, <vscale x 8 x i1> %pred, half* %addr, i64 %offset) {
; CHECK-LABEL: st4h_f16:
; CHECK: st4h { z0.h, z1.h, z2.h, z3.h }, p0, [x0, x1, lsl #1]
; CHECK-NEXT: ret
  %1 = getelementptr half, half* %addr, i64 %offset
  call void @llvm.aarch64.sve.st4.nxv8f16(<vscale x 8 x half> %v0,
                                          <vscale x 8 x half> %v1,
                                          <vscale x 8 x half> %v2,
                                          <vscale x 8 x half> %v3,
                                          <vscale x 8 x i1> %pred,
                                          half* %1)
  ret void
}

;
; ST4W
;

define void @st4w_i32(<vscale x 4 x i32> %v0, <vscale x 4 x i32> %v1, <vscale x 4 x i32> %v2, <vscale x 4 x i32> %v3, <vscale x 4 x i1> %pred, i32* %addr, i64 %offset) {
; CHECK-LABEL: st4w_i32:
; CHECK: st4w { z0.s, z1.s, z2.s, z3.s }, p0, [x0, x1, lsl #2]
; CHECK-NEXT: ret
  %1 = getelementptr i32, i32* %addr, i64 %offset
  call void @llvm.aarch64.sve.st4.nxv4i32(<vscale x 4 x i32> %v0,
                                          <vscale x 4 x i32> %v1,
                                          <vscale x 4 x i32> %v2,
                                          <vscale x 4 x i32> %v3,
                                          <vscale x 4 x i1> %pred,
                                          i32* %1)
  ret void
}

define void @st4w_f32(<vscale x 4 x float> %v0, <vscale x 4 x float> %v1, <vscale x 4 x float> %v2, <vscale x 4 x float> %v3, <vscale x 4 x i1> %pred, float* %addr, i64 %offset) {
; CHECK-LABEL: st4w_f32:
; CHECK: st4w { z0.s, z1.s, z2.s, z3.s }, p0, [x0, x1, lsl #2]
; CHECK-NEXT: ret
  %1 = getelementptr float, float* %addr, i64 %offset
  call void @llvm.aarch64.sve.st4.nxv4f32(<vscale x 4 x float> %v0,
                                          <vscale x 4 x float> %v1,
                                          <vscale x 4 x float> %v2,
                                          <vscale x 4 x float> %v3,
                                          <vscale x 4 x i1> %pred,
                                          float* %1)
  ret void
}

;
; ST4D
;

define void @st4d_i64(<vscale x 2 x i64> %v0, <vscale x 2 x i64> %v1, <vscale x 2 x i64> %v2, <vscale x 2 x i64> %v3, <vscale x 2 x i1> %pred, i64* %addr, i64 %offset) {
; CHECK-LABEL: st4d_i64:
; CHECK: st4d { z0.d, z1.d, z2.d, z3.d }, p0, [x0, x1, lsl #3]
; CHECK-NEXT: ret
  %1 = getelementptr i64, i64* %addr, i64 %offset
  call void @llvm.aarch64.sve.st4.nxv2i64(<vscale x 2 x i64> %v0,
                                          <vscale x 2 x i64> %v1,
                                          <vscale x 2 x i64> %v2,
                                          <vscale x 2 x i64> %v3,
                                          <vscale x 2 x i1> %pred,
                                          i64* %1)
  ret void
}

define void @st4d_f64(<vscale x 2 x double> %v0, <vscale x 2 x double> %v1, <vscale x 2 x double> %v2, <vscale x 2 x double> %v3, <vscale x 2 x i1> %pred, double* %addr, i64 %offset) {
; CHECK-LABEL: st4d_f64:
; CHECK: st4d { z0.d, z1.d, z2.d, z3.d }, p0, [x0, x1, lsl #3]
; CHECK-NEXT: ret
  %1 = getelementptr double, double* %addr, i64 %offset
  call void @llvm.aarch64.sve.st4.nxv2f64(<vscale x 2 x double> %v0,
                                          <vscale x 2 x double> %v1,
                                          <vscale x 2 x double> %v2,
                                          <vscale x 2 x double> %v3,
                                          <vscale x 2 x i1> %pred,
                                          double* %1)
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
