; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; WARN-NOT: warning

;
; CLS
;

define <vscale x 16 x i8> @cls_i8(<vscale x 16 x i8> %a, <vscale x 16 x i1> %pg, <vscale x 16 x i8> %b) {
; CHECK-LABEL: cls_i8:
; CHECK: cls z0.b, p0/m, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.cls.nxv16i8(<vscale x 16 x i8> %a,
                                                               <vscale x 16 x i1> %pg,
                                                               <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @cls_i16(<vscale x 8 x i16> %a, <vscale x 8 x i1> %pg, <vscale x 8 x i16> %b) {
; CHECK-LABEL: cls_i16:
; CHECK: cls z0.h, p0/m, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.cls.nxv8i16(<vscale x 8 x i16> %a,
                                                               <vscale x 8 x i1> %pg,
                                                               <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @cls_i32(<vscale x 4 x i32> %a, <vscale x 4 x i1> %pg, <vscale x 4 x i32> %b) {
; CHECK-LABEL: cls_i32:
; CHECK: cls z0.s, p0/m, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.cls.nxv4i32(<vscale x 4 x i32> %a,
                                                               <vscale x 4 x i1> %pg,
                                                               <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @cls_i64(<vscale x 2 x i64> %a, <vscale x 2 x i1> %pg, <vscale x 2 x i64> %b) {
; CHECK-LABEL: cls_i64:
; CHECK: cls z0.d, p0/m, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.cls.nxv2i64(<vscale x 2 x i64> %a,
                                                               <vscale x 2 x i1> %pg,
                                                               <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

;
; CLZ
;

define <vscale x 16 x i8> @clz_i8(<vscale x 16 x i8> %a, <vscale x 16 x i1> %pg, <vscale x 16 x i8> %b) {
; CHECK-LABEL: clz_i8:
; CHECK: clz z0.b, p0/m, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.clz.nxv16i8(<vscale x 16 x i8> %a,
                                                               <vscale x 16 x i1> %pg,
                                                               <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @clz_i16(<vscale x 8 x i16> %a, <vscale x 8 x i1> %pg, <vscale x 8 x i16> %b) {
; CHECK-LABEL: clz_i16:
; CHECK: clz z0.h, p0/m, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.clz.nxv8i16(<vscale x 8 x i16> %a,
                                                               <vscale x 8 x i1> %pg,
                                                               <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @clz_i32(<vscale x 4 x i32> %a, <vscale x 4 x i1> %pg, <vscale x 4 x i32> %b) {
; CHECK-LABEL: clz_i32:
; CHECK: clz z0.s, p0/m, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.clz.nxv4i32(<vscale x 4 x i32> %a,
                                                               <vscale x 4 x i1> %pg,
                                                               <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @clz_i64(<vscale x 2 x i64> %a, <vscale x 2 x i1> %pg, <vscale x 2 x i64> %b) {
; CHECK-LABEL: clz_i64:
; CHECK: clz z0.d, p0/m, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.clz.nxv2i64(<vscale x 2 x i64> %a,
                                                               <vscale x 2 x i1> %pg,
                                                               <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

;
; CNT
;

define <vscale x 16 x i8> @cnt_i8(<vscale x 16 x i8> %a, <vscale x 16 x i1> %pg, <vscale x 16 x i8> %b) {
; CHECK-LABEL: cnt_i8:
; CHECK: cnt z0.b, p0/m, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.cnt.nxv16i8(<vscale x 16 x i8> %a,
                                                               <vscale x 16 x i1> %pg,
                                                               <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @cnt_i16(<vscale x 8 x i16> %a, <vscale x 8 x i1> %pg, <vscale x 8 x i16> %b) {
; CHECK-LABEL: cnt_i16:
; CHECK: cnt z0.h, p0/m, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.cnt.nxv8i16(<vscale x 8 x i16> %a,
                                                               <vscale x 8 x i1> %pg,
                                                               <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @cnt_i32(<vscale x 4 x i32> %a, <vscale x 4 x i1> %pg, <vscale x 4 x i32> %b) {
; CHECK-LABEL: cnt_i32:
; CHECK: cnt z0.s, p0/m, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.cnt.nxv4i32(<vscale x 4 x i32> %a,
                                                               <vscale x 4 x i1> %pg,
                                                               <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @cnt_i64(<vscale x 2 x i64> %a, <vscale x 2 x i1> %pg, <vscale x 2 x i64> %b) {
; CHECK-LABEL: cnt_i64:
; CHECK: cnt z0.d, p0/m, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.cnt.nxv2i64(<vscale x 2 x i64> %a,
                                                               <vscale x 2 x i1> %pg,
                                                               <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

define <vscale x 8 x i16> @cnt_f16(<vscale x 8 x i16> %a, <vscale x 8 x i1> %pg, <vscale x 8 x half> %b) {
; CHECK-LABEL: cnt_f16:
; CHECK: cnt z0.h, p0/m, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.cnt.nxv8f16(<vscale x 8 x i16> %a,
                                                               <vscale x 8 x i1> %pg,
                                                               <vscale x 8 x half> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 8 x i16> @cnt_bf16(<vscale x 8 x i16> %a, <vscale x 8 x i1> %pg, <vscale x 8 x bfloat> %b) #0 {
; CHECK-LABEL: cnt_bf16:
; CHECK: cnt z0.h, p0/m, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.cnt.nxv8bf16(<vscale x 8 x i16> %a,
                                                                <vscale x 8 x i1> %pg,
                                                                <vscale x 8 x bfloat> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @cnt_f32(<vscale x 4 x i32> %a, <vscale x 4 x i1> %pg, <vscale x 4 x float> %b) {
; CHECK-LABEL: cnt_f32:
; CHECK: cnt z0.s, p0/m, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.cnt.nxv4f32(<vscale x 4 x i32> %a,
                                                               <vscale x 4 x i1> %pg,
                                                               <vscale x 4 x float> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @cnt_f64(<vscale x 2 x i64> %a, <vscale x 2 x i1> %pg, <vscale x 2 x double> %b) {
; CHECK-LABEL: cnt_f64:
; CHECK: cnt z0.d, p0/m, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.cnt.nxv2f64(<vscale x 2 x i64> %a,
                                                               <vscale x 2 x i1> %pg,
                                                               <vscale x 2 x double> %b)
  ret <vscale x 2 x i64> %out
}

declare <vscale x 16 x i8> @llvm.aarch64.sve.cls.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i1>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.cls.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i1>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.cls.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i1>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.cls.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i1>, <vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.clz.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i1>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.clz.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i1>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.clz.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i1>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.clz.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i1>, <vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.cnt.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i1>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.cnt.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i1>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.cnt.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i1>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.cnt.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i1>, <vscale x 2 x i64>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.cnt.nxv8f16(<vscale x 8 x i16>, <vscale x 8 x i1>, <vscale x 8 x half>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.cnt.nxv8bf16(<vscale x 8 x i16>, <vscale x 8 x i1>, <vscale x 8 x bfloat>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.cnt.nxv4f32(<vscale x 4 x i32>, <vscale x 4 x i1>, <vscale x 4 x float>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.cnt.nxv2f64(<vscale x 2 x i64>, <vscale x 2 x i1>, <vscale x 2 x double>)

; +bf16 is required for the bfloat version.
attributes #0 = { "target-features"="+sve,+bf16" }
