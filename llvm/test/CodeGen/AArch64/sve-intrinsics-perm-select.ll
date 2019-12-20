; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s | FileCheck %s

;
; CLASTA (Vectors)
;

define <vscale x 16 x i8> @clasta_i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: clasta_i8:
; CHECK: clasta z0.b, p0, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.clasta.nxv16i8(<vscale x 16 x i1> %pg,
                                                                  <vscale x 16 x i8> %a,
                                                                  <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @clasta_i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: clasta_i16:
; CHECK: clasta z0.h, p0, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.clasta.nxv8i16(<vscale x 8 x i1> %pg,
                                                                  <vscale x 8 x i16> %a,
                                                                  <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @clasta_i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: clasta_i32:
; CHECK: clasta z0.s, p0, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.clasta.nxv4i32(<vscale x 4 x i1> %pg,
                                                                  <vscale x 4 x i32> %a,
                                                                  <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @clasta_i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: clasta_i64:
; CHECK: clasta z0.d, p0, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.clasta.nxv2i64(<vscale x 2 x i1> %pg,
                                                                  <vscale x 2 x i64> %a,
                                                                  <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

define <vscale x 8 x half> @clasta_f16(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a, <vscale x 8 x half> %b) {
; CHECK-LABEL: clasta_f16:
; CHECK: clasta z0.h, p0, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.clasta.nxv8f16(<vscale x 8 x i1> %pg,
                                                                   <vscale x 8 x half> %a,
                                                                   <vscale x 8 x half> %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @clasta_f32(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: clasta_f32:
; CHECK: clasta z0.s, p0, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.clasta.nxv4f32(<vscale x 4 x i1> %pg,
                                                                    <vscale x 4 x float> %a,
                                                                    <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @clasta_f64(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> %b) {
; CHECK-LABEL: clasta_f64:
; CHECK: clasta z0.d, p0, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.clasta.nxv2f64(<vscale x 2 x i1> %pg,
                                                                     <vscale x 2 x double> %a,
                                                                     <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %out
}

;
; CLASTA (Scalar)
;

define i8 @clasta_n_i8(<vscale x 16 x i1> %pg, i8 %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: clasta_n_i8:
; CHECK: clasta w0, p0, w0, z0.b
; CHECK-NEXT: ret
  %out = call i8 @llvm.aarch64.sve.clasta.n.nxv16i8(<vscale x 16 x i1> %pg,
                                                    i8 %a,
                                                    <vscale x 16 x i8> %b)
  ret i8 %out
}

define i16 @clasta_n_i16(<vscale x 8 x i1> %pg, i16 %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: clasta_n_i16:
; CHECK: clasta w0, p0, w0, z0.h
; CHECK-NEXT: ret
  %out = call i16 @llvm.aarch64.sve.clasta.n.nxv8i16(<vscale x 8 x i1> %pg,
                                                     i16 %a,
                                                     <vscale x 8 x i16> %b)
  ret i16 %out
}

define i32 @clasta_n_i32(<vscale x 4 x i1> %pg, i32 %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: clasta_n_i32:
; CHECK: clasta w0, p0, w0, z0.s
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.clasta.n.nxv4i32(<vscale x 4 x i1> %pg,
                                                     i32 %a,
                                                     <vscale x 4 x i32> %b)
  ret i32 %out
}

define i64 @clasta_n_i64(<vscale x 2 x i1> %pg, i64 %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: clasta_n_i64:
; CHECK: clasta x0, p0, x0, z0.d
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.clasta.n.nxv2i64(<vscale x 2 x i1> %pg,
                                                     i64 %a,
                                                     <vscale x 2 x i64> %b)
  ret i64 %out
}

define half @clasta_n_f16(<vscale x 8 x i1> %pg, half %a, <vscale x 8 x half> %b) {
; CHECK-LABEL: clasta_n_f16:
; CHECK: clasta h0, p0, h0, z1.h
; CHECK-NEXT: ret
  %out = call half @llvm.aarch64.sve.clasta.n.nxv8f16(<vscale x 8 x i1> %pg,
                                                      half %a,
                                                      <vscale x 8 x half> %b)
  ret half %out
}

define float @clasta_n_f32(<vscale x 4 x i1> %pg, float %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: clasta_n_f32:
; CHECK: clasta s0, p0, s0, z1.s
; CHECK-NEXT: ret
  %out = call float @llvm.aarch64.sve.clasta.n.nxv4f32(<vscale x 4 x i1> %pg,
                                                       float %a,
                                                       <vscale x 4 x float> %b)
  ret float %out
}

define double @clasta_n_f64(<vscale x 2 x i1> %pg, double %a, <vscale x 2 x double> %b) {
; CHECK-LABEL: clasta_n_f64:
; CHECK: clasta d0, p0, d0, z1.d
; CHECK-NEXT: ret
  %out = call double @llvm.aarch64.sve.clasta.n.nxv2f64(<vscale x 2 x i1> %pg,
                                                        double %a,
                                                        <vscale x 2 x double> %b)
  ret double %out
}

;
; CLASTB (Vectors)
;

define <vscale x 16 x i8> @clastb_i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: clastb_i8:
; CHECK: clastb z0.b, p0, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.clastb.nxv16i8(<vscale x 16 x i1> %pg,
                                                                  <vscale x 16 x i8> %a,
                                                                  <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @clastb_i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: clastb_i16:
; CHECK: clastb z0.h, p0, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.clastb.nxv8i16(<vscale x 8 x i1> %pg,
                                                                  <vscale x 8 x i16> %a,
                                                                  <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @clastb_i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: clastb_i32:
; CHECK: clastb z0.s, p0, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.clastb.nxv4i32(<vscale x 4 x i1> %pg,
                                                                  <vscale x 4 x i32> %a,
                                                                  <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @clastb_i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: clastb_i64:
; CHECK: clastb z0.d, p0, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.clastb.nxv2i64(<vscale x 2 x i1> %pg,
                                                                  <vscale x 2 x i64> %a,
                                                                  <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

define <vscale x 8 x half> @clastb_f16(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a, <vscale x 8 x half> %b) {
; CHECK-LABEL: clastb_f16:
; CHECK: clastb z0.h, p0, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.clastb.nxv8f16(<vscale x 8 x i1> %pg,
                                                                   <vscale x 8 x half> %a,
                                                                   <vscale x 8 x half> %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @clastb_f32(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: clastb_f32:
; CHECK: clastb z0.s, p0, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.clastb.nxv4f32(<vscale x 4 x i1> %pg,
                                                                    <vscale x 4 x float> %a,
                                                                    <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @clastb_f64(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> %b) {
; CHECK-LABEL: clastb_f64:
; CHECK: clastb z0.d, p0, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.clastb.nxv2f64(<vscale x 2 x i1> %pg,
                                                                     <vscale x 2 x double> %a,
                                                                     <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %out
}

;
; CLASTB (Scalar)
;

define i8 @clastb_n_i8(<vscale x 16 x i1> %pg, i8 %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: clastb_n_i8:
; CHECK: clastb w0, p0, w0, z0.b
; CHECK-NEXT: ret
  %out = call i8 @llvm.aarch64.sve.clastb.n.nxv16i8(<vscale x 16 x i1> %pg,
                                                    i8 %a,
                                                    <vscale x 16 x i8> %b)
  ret i8 %out
}

define i16 @clastb_n_i16(<vscale x 8 x i1> %pg, i16 %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: clastb_n_i16:
; CHECK: clastb w0, p0, w0, z0.h
; CHECK-NEXT: ret
  %out = call i16 @llvm.aarch64.sve.clastb.n.nxv8i16(<vscale x 8 x i1> %pg,
                                                     i16 %a,
                                                     <vscale x 8 x i16> %b)
  ret i16 %out
}

define i32 @clastb_n_i32(<vscale x 4 x i1> %pg, i32 %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: clastb_n_i32:
; CHECK: clastb w0, p0, w0, z0.s
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.clastb.n.nxv4i32(<vscale x 4 x i1> %pg,
                                                     i32 %a,
                                                     <vscale x 4 x i32> %b)
  ret i32 %out
}

define i64 @clastb_n_i64(<vscale x 2 x i1> %pg, i64 %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: clastb_n_i64:
; CHECK: clastb x0, p0, x0, z0.d
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.clastb.n.nxv2i64(<vscale x 2 x i1> %pg,
                                                     i64 %a,
                                                     <vscale x 2 x i64> %b)
  ret i64 %out
}

define half @clastb_n_f16(<vscale x 8 x i1> %pg, half %a, <vscale x 8 x half> %b) {
; CHECK-LABEL: clastb_n_f16:
; CHECK: clastb h0, p0, h0, z1.h
; CHECK-NEXT: ret
  %out = call half @llvm.aarch64.sve.clastb.n.nxv8f16(<vscale x 8 x i1> %pg,
                                                      half %a,
                                                      <vscale x 8 x half> %b)
  ret half %out
}

define float @clastb_n_f32(<vscale x 4 x i1> %pg, float %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: clastb_n_f32:
; CHECK: clastb s0, p0, s0, z1.s
; CHECK-NEXT: ret
  %out = call float @llvm.aarch64.sve.clastb.n.nxv4f32(<vscale x 4 x i1> %pg,
                                                       float %a,
                                                       <vscale x 4 x float> %b)
  ret float %out
}

define double @clastb_n_f64(<vscale x 2 x i1> %pg, double %a, <vscale x 2 x double> %b) {
; CHECK-LABEL: clastb_n_f64:
; CHECK: clastb d0, p0, d0, z1.d
; CHECK-NEXT: ret
  %out = call double @llvm.aarch64.sve.clastb.n.nxv2f64(<vscale x 2 x i1> %pg,
                                                        double %a,
                                                        <vscale x 2 x double> %b)
  ret double %out
}

;
; EXT
;

define <vscale x 16 x i8> @ext_i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: ext_i8:
; CHECK: ext z0.b, z0.b, z1.b, #255
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.ext.nxv16i8(<vscale x 16 x i8> %a,
                                                               <vscale x 16 x i8> %b,
                                                               i32 255)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @ext_i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: ext_i16:
; CHECK: ext z0.b, z0.b, z1.b, #0
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.ext.nxv8i16(<vscale x 8 x i16> %a,
                                                               <vscale x 8 x i16> %b,
                                                               i32 0)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @ext_i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: ext_i32:
; CHECK: ext z0.b, z0.b, z1.b, #4
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.ext.nxv4i32(<vscale x 4 x i32> %a,
                                                               <vscale x 4 x i32> %b,
                                                               i32 1)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @ext_i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: ext_i64:
; CHECK: ext z0.b, z0.b, z1.b, #16
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.ext.nxv2i64(<vscale x 2 x i64> %a,
                                                               <vscale x 2 x i64> %b,
                                                               i32 2)
  ret <vscale x 2 x i64> %out
}

define <vscale x 8 x half> @ext_f16(<vscale x 8 x half> %a, <vscale x 8 x half> %b) {
; CHECK-LABEL: ext_f16:
; CHECK: ext z0.b, z0.b, z1.b, #6
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.ext.nxv8f16(<vscale x 8 x half> %a,
                                                                <vscale x 8 x half> %b,
                                                                i32 3)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @ext_f32(<vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: ext_f32:
; CHECK: ext z0.b, z0.b, z1.b, #16
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.ext.nxv4f32(<vscale x 4 x float> %a,
                                                                 <vscale x 4 x float> %b,
                                                                 i32 4)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @ext_f64(<vscale x 2 x double> %a, <vscale x 2 x double> %b) {
; CHECK-LABEL: ext_f64:
; CHECK: ext z0.b, z0.b, z1.b, #40
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.ext.nxv2f64(<vscale x 2 x double> %a,
                                                                  <vscale x 2 x double> %b,
                                                                  i32 5)
  ret <vscale x 2 x double> %out
}

;
; LASTA
;

define i8 @lasta_i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a) {
; CHECK-LABEL: lasta_i8
; CHECK: lasta w0, p0, z0.b
; CHECK-NEXT: ret
  %res = call i8 @llvm.aarch64.sve.lasta.nxv16i8(<vscale x 16 x i1> %pg,
                                                 <vscale x 16 x i8> %a)
  ret i8 %res
}

define i16 @lasta_i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a) {
; CHECK-LABEL: lasta_i16
; CHECK: lasta w0, p0, z0.h
; CHECK-NEXT: ret
  %res = call i16 @llvm.aarch64.sve.lasta.nxv8i16(<vscale x 8 x i1> %pg,
                                                  <vscale x 8 x i16> %a)
  ret i16 %res
}

define i32 @lasta_i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a) {
; CHECK-LABEL: lasta_i32
; CHECK: lasta w0, p0, z0.s
; CHECK-NEXT: ret
  %res = call i32 @llvm.aarch64.sve.lasta.nxv4i32(<vscale x 4 x i1> %pg,
                                                  <vscale x 4 x i32> %a)
  ret i32 %res
}

define i64 @lasta_i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a) {
; CHECK-LABEL:  lasta_i64
; CHECK: lasta x0, p0, z0.d
; CHECK-NEXT: ret
  %res = call i64 @llvm.aarch64.sve.lasta.nxv2i64(<vscale x 2 x i1> %pg,
                                                  <vscale x 2 x i64> %a)
  ret i64 %res
}

define half @lasta_f16(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a) {
; CHECK-LABEL: lasta_f16
; CHECK: lasta h0, p0, z0.h
; CHECK-NEXT: ret
  %res = call half @llvm.aarch64.sve.lasta.nxv8f16(<vscale x 8 x i1> %pg,
                                                   <vscale x 8 x half> %a)
  ret half %res
}

define float @lasta_f32(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a) {
; CHECK-LABEL: lasta_f32
; CHECK: lasta s0, p0, z0.s
; CHECK-NEXT: ret
  %res = call float @llvm.aarch64.sve.lasta.nxv4f32(<vscale x 4 x i1> %pg,
                                                    <vscale x 4 x float> %a)
  ret float %res
}

define float @lasta_f32_v2(<vscale x 2 x i1> %pg, <vscale x 2 x float> %a) {
; CHECK-LABEL: lasta_f32_v2
; CHECK: lasta s0, p0, z0.s
; CHECK-NEXT: ret
  %res = call float @llvm.aarch64.sve.lasta.nxv2f32(<vscale x 2 x i1> %pg,
                                                    <vscale x 2 x float> %a)
  ret float %res
}

define double @lasta_f64(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a) {
; CHECK-LABEL:  lasta_f64
; CHECK: lasta d0, p0, z0.d
; CHECK-NEXT: ret
  %res = call double @llvm.aarch64.sve.lasta.nxv2f64(<vscale x 2 x i1> %pg,
                                                     <vscale x 2 x double> %a)
  ret double %res
}

;
; LASTB
;

define i8 @lastb_i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a) {
; CHECK-LABEL: lastb_i8
; CHECK: lastb w0, p0, z0.b
; CHECK-NEXT: ret
  %res = call i8 @llvm.aarch64.sve.lastb.nxv16i8(<vscale x 16 x i1> %pg,
                                                 <vscale x 16 x i8> %a)
  ret i8 %res
}

define i16 @lastb_i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a) {
; CHECK-LABEL: lastb_i16
; CHECK: lastb w0, p0, z0.h
; CHECK-NEXT: ret
  %res = call i16 @llvm.aarch64.sve.lastb.nxv8i16(<vscale x 8 x i1> %pg,
                                                  <vscale x 8 x i16> %a)
  ret i16 %res
}

define i32 @lastb_i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a) {
; CHECK-LABEL: lastb_i32
; CHECK: lastb w0, p0, z0.s
; CHECK-NEXT: ret
  %res = call i32 @llvm.aarch64.sve.lastb.nxv4i32(<vscale x 4 x i1> %pg,
                                                  <vscale x 4 x i32> %a)
  ret i32 %res
}

define i64 @lastb_i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a) {
; CHECK-LABEL:  lastb_i64
; CHECK: lastb x0, p0, z0.d
; CHECK-NEXT: ret
  %res = call i64 @llvm.aarch64.sve.lastb.nxv2i64(<vscale x 2 x i1> %pg,
                                                  <vscale x 2 x i64> %a)
  ret i64 %res
}

define half @lastb_f16(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a) {
; CHECK-LABEL: lastb_f16
; CHECK: lastb h0, p0, z0.h
; CHECK-NEXT: ret
  %res = call half @llvm.aarch64.sve.lastb.nxv8f16(<vscale x 8 x i1> %pg,
                                                   <vscale x 8 x half> %a)
  ret half %res
}

define float @lastb_f32(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a) {
; CHECK-LABEL: lastb_f32
; CHECK: lastb s0, p0, z0.s
; CHECK-NEXT: ret
  %res = call float @llvm.aarch64.sve.lastb.nxv4f32(<vscale x 4 x i1> %pg,
                                                    <vscale x 4 x float> %a)
  ret float %res
}

define float @lastb_f32_v2(<vscale x 2 x i1> %pg, <vscale x 2 x float> %a) {
; CHECK-LABEL: lastb_f32_v2
; CHECK: lastb s0, p0, z0.s
; CHECK-NEXT: ret
  %res = call float @llvm.aarch64.sve.lastb.nxv2f32(<vscale x 2 x i1> %pg,
                                                    <vscale x 2 x float> %a)
  ret float %res
}

define double @lastb_f64(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a) {
; CHECK-LABEL:  lastb_f64
; CHECK: lastb d0, p0, z0.d
; CHECK-NEXT: ret
  %res = call double @llvm.aarch64.sve.lastb.nxv2f64(<vscale x 2 x i1> %pg,
                                                     <vscale x 2 x double> %a)
  ret double %res
}

;
; COMPACT
;

define <vscale x 4 x i32> @compact_i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a) {
; CHECK-LABEL: compact_i32:
; CHECK: compact z0.s, p0, z0.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.compact.nxv4i32(<vscale x 4 x i1> %pg,
                                                                   <vscale x 4 x i32> %a)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @compact_i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a) {
; CHECK-LABEL: compact_i64:
; CHECK: compact z0.d, p0, z0.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.compact.nxv2i64(<vscale x 2 x i1> %pg,
                                                                   <vscale x 2 x i64> %a)
  ret <vscale x 2 x i64> %out
}

define <vscale x 4 x float> @compact_f32(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a) {
; CHECK-LABEL: compact_f32:
; CHECK: compact z0.s, p0, z0.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.compact.nxv4f32(<vscale x 4 x i1> %pg,
                                                                     <vscale x 4 x float> %a)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @compact_f64(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a) {
; CHECK-LABEL: compact_f64:
; CHECK: compact z0.d, p0, z0.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.compact.nxv2f64(<vscale x 2 x i1> %pg,
                                                                      <vscale x 2 x double> %a)
  ret <vscale x 2 x double> %out
}

;
; REV
;

define <vscale x 16 x i1> @rev_b8( <vscale x 16 x i1> %a) {
; CHECK-LABEL: rev_b8
; CHECK: rev p0.b, p0.b
; CHECK-NEXT: ret
  %res = call <vscale x 16 x i1> @llvm.aarch64.sve.rev.nxv16i1(<vscale x 16 x i1> %a)
  ret <vscale x 16 x i1> %res
}

define <vscale x 8 x i1> @rev_b16(<vscale x 8 x i1> %a) {
; CHECK-LABEL: rev_b16
; CHECK: rev p0.h, p0.h
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i1> @llvm.aarch64.sve.rev.nxv8i1(<vscale x 8 x i1> %a)
  ret <vscale x 8 x i1> %res
}

define <vscale x 4 x i1> @rev_b32(<vscale x 4 x i1> %a) {
; CHECK-LABEL: rev_b32
; CHECK: rev p0.s, p0.s
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i1> @llvm.aarch64.sve.rev.nxv4i1(<vscale x 4 x i1> %a)
  ret <vscale x 4 x i1> %res
}

define <vscale x 2 x i1> @rev_b64(<vscale x 2 x i1> %a) {
; CHECK-LABEL:  rev_b64
; CHECK: rev p0.d, p0.d
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i1> @llvm.aarch64.sve.rev.nxv2i1(<vscale x 2 x i1> %a)
  ret <vscale x 2 x i1> %res
}

define <vscale x 16 x i8> @rev_i8( <vscale x 16 x i8> %a) {
; CHECK-LABEL: rev_i8
; CHECK: rev z0.b, z0.b
; CHECK-NEXT: ret
  %res = call <vscale x 16 x i8> @llvm.aarch64.sve.rev.nxv16i8(<vscale x 16 x i8> %a)
  ret <vscale x 16 x i8> %res
}

define <vscale x 8 x i16> @rev_i16(<vscale x 8 x i16> %a) {
; CHECK-LABEL: rev_i16
; CHECK: rev z0.h, z0.h
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i16> @llvm.aarch64.sve.rev.nxv8i16(<vscale x 8 x i16> %a)
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @rev_i32(<vscale x 4 x i32> %a) {
; CHECK-LABEL: rev_i32
; CHECK: rev z0.s, z0.s
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.rev.nxv4i32(<vscale x 4 x i32> %a)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @rev_i64(<vscale x 2 x i64> %a) {
; CHECK-LABEL:  rev_i64
; CHECK: rev z0.d, z0.d
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.rev.nxv2i64(<vscale x 2 x i64> %a)
  ret <vscale x 2 x i64> %res
}

define <vscale x 8 x half> @rev_f16(<vscale x 8 x half> %a) {
; CHECK-LABEL: rev_f16
; CHECK: rev z0.h, z0.h
; CHECK-NEXT: ret
  %res = call <vscale x 8 x half> @llvm.aarch64.sve.rev.nxv8f16(<vscale x 8 x half> %a)
  ret <vscale x 8 x half> %res
}

define <vscale x 4 x float> @rev_f32(<vscale x 4 x float> %a) {
; CHECK-LABEL: rev_f32
; CHECK: rev z0.s, z0.s
; CHECK-NEXT: ret
  %res = call <vscale x 4 x float> @llvm.aarch64.sve.rev.nxv4f32(<vscale x 4 x float> %a)
  ret <vscale x 4 x float> %res
}

define <vscale x 2 x double> @rev_f64(<vscale x 2 x double> %a) {
; CHECK-LABEL:  rev_f64
; CHECK: rev z0.d, z0.d
; CHECK-NEXT: ret
  %res = call <vscale x 2 x double> @llvm.aarch64.sve.rev.nxv2f64(<vscale x 2 x double> %a)
  ret <vscale x 2 x double> %res
}

;
; SPLICE
;

define <vscale x 16 x i8> @splice_i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: splice_i8:
; CHECK: splice z0.b, p0, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.splice.nxv16i8(<vscale x 16 x i1> %pg,
                                                                  <vscale x 16 x i8> %a,
                                                                  <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @splice_i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: splice_i16:
; CHECK: splice z0.h, p0, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.splice.nxv8i16(<vscale x 8 x i1> %pg,
                                                                  <vscale x 8 x i16> %a,
                                                                  <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @splice_i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: splice_i32:
; CHECK: splice z0.s, p0, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.splice.nxv4i32(<vscale x 4 x i1> %pg,
                                                                  <vscale x 4 x i32> %a,
                                                                  <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @splice_i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: splice_i64:
; CHECK: splice z0.d, p0, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.splice.nxv2i64(<vscale x 2 x i1> %pg,
                                                                  <vscale x 2 x i64> %a,
                                                                  <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

define <vscale x 8 x half> @splice_f16(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a, <vscale x 8 x half> %b) {
; CHECK-LABEL: splice_f16:
; CHECK: splice z0.h, p0, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.splice.nxv8f16(<vscale x 8 x i1> %pg,
                                                                   <vscale x 8 x half> %a,
                                                                   <vscale x 8 x half> %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @splice_f32(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: splice_f32:
; CHECK: splice z0.s, p0, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.splice.nxv4f32(<vscale x 4 x i1> %pg,
                                                                    <vscale x 4 x float> %a,
                                                                    <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @splice_f64(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> %b) {
; CHECK-LABEL: splice_f64:
; CHECK: splice z0.d, p0, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.splice.nxv2f64(<vscale x 2 x i1> %pg,
                                                                     <vscale x 2 x double> %a,
                                                                     <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %out
}

;
; SUNPKHI
;

define <vscale x 8 x i16> @sunpkhi_i16(<vscale x 16 x i8> %a) {
; CHECK-LABEL: sunpkhi_i16
; CHECK: sunpkhi z0.h, z0.b
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i16> @llvm.aarch64.sve.sunpkhi.nxv8i16(<vscale x 16 x i8> %a)
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @sunpkhi_i32(<vscale x 8 x i16> %a) {
; CHECK-LABEL: sunpkhi_i32
; CHECK: sunpkhi z0.s, z0.h
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.sunpkhi.nxv4i32(<vscale x 8 x i16> %a)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @sunpkhi_i64(<vscale x 4 x i32> %a) {
; CHECK-LABEL:  sunpkhi_i64
; CHECK: sunpkhi z0.d, z0.s
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.sunpkhi.nxv2i64(<vscale x 4 x i32> %a)
  ret <vscale x 2 x i64> %res
}

;
; SUNPKLO
;

define <vscale x 8 x i16> @sunpklo_i16(<vscale x 16 x i8> %a) {
; CHECK-LABEL: sunpklo_i16
; CHECK: sunpklo z0.h, z0.b
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i16> @llvm.aarch64.sve.sunpklo.nxv8i16(<vscale x 16 x i8> %a)
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @sunpklo_i32(<vscale x 8 x i16> %a) {
; CHECK-LABEL: sunpklo_i32
; CHECK: sunpklo z0.s, z0.h
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.sunpklo.nxv4i32(<vscale x 8 x i16> %a)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @sunpklo_i64(<vscale x 4 x i32> %a) {
; CHECK-LABEL:  sunpklo_i64
; CHECK: sunpklo z0.d, z0.s
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.sunpklo.nxv2i64(<vscale x 4 x i32> %a)
  ret <vscale x 2 x i64> %res
}

;
; TBL
;

define <vscale x 16 x i8> @tbl_i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: tbl_i8:
; CHECK: tbl z0.b, { z0.b }, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.tbl.nxv16i8(<vscale x 16 x i8> %a,
                                                               <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @tbl_i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: tbl_i16:
; CHECK: tbl z0.h, { z0.h }, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.tbl.nxv8i16(<vscale x 8 x i16> %a,
                                                               <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @tbl_i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: tbl_i32:
; CHECK: tbl z0.s, { z0.s }, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.tbl.nxv4i32(<vscale x 4 x i32> %a,
                                                               <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @tbl_i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: tbl_i64:
; CHECK: tbl z0.d, { z0.d }, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.tbl.nxv2i64(<vscale x 2 x i64> %a,
                                                               <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

define <vscale x 8 x half> @tbl_f16(<vscale x 8 x half> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: tbl_f16:
; CHECK: tbl z0.h, { z0.h }, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.tbl.nxv8f16(<vscale x 8 x half> %a,
                                                                <vscale x 8 x i16> %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @tbl_f32(<vscale x 4 x float> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: tbl_f32:
; CHECK: tbl z0.s, { z0.s }, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.tbl.nxv4f32(<vscale x 4 x float> %a,
                                                                 <vscale x 4 x i32> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @tbl_f64(<vscale x 2 x double> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: tbl_f64:
; CHECK: tbl z0.d, { z0.d }, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.tbl.nxv2f64(<vscale x 2 x double> %a,
                                                                  <vscale x 2 x i64> %b)
  ret <vscale x 2 x double> %out
}

;
; UUNPKHI
;

define <vscale x 8 x i16> @uunpkhi_i16(<vscale x 16 x i8> %a) {
; CHECK-LABEL: uunpkhi_i16
; CHECK: uunpkhi z0.h, z0.b
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i16> @llvm.aarch64.sve.uunpkhi.nxv8i16(<vscale x 16 x i8> %a)
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @uunpkhi_i32(<vscale x 8 x i16> %a) {
; CHECK-LABEL: uunpkhi_i32
; CHECK: uunpkhi z0.s, z0.h
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.uunpkhi.nxv4i32(<vscale x 8 x i16> %a)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @uunpkhi_i64(<vscale x 4 x i32> %a) {
; CHECK-LABEL:  uunpkhi_i64
; CHECK: uunpkhi z0.d, z0.s
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.uunpkhi.nxv2i64(<vscale x 4 x i32> %a)
  ret <vscale x 2 x i64> %res
}

;
; UUNPKLO
;

define <vscale x 8 x i16> @uunpklo_i16(<vscale x 16 x i8> %a) {
; CHECK-LABEL: uunpklo_i16
; CHECK: uunpklo z0.h, z0.b
; CHECK-NEXT: ret
  %res = call <vscale x 8 x i16> @llvm.aarch64.sve.uunpklo.nxv8i16(<vscale x 16 x i8> %a)
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @uunpklo_i32(<vscale x 8 x i16> %a) {
; CHECK-LABEL: uunpklo_i32
; CHECK: uunpklo z0.s, z0.h
; CHECK-NEXT: ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.uunpklo.nxv4i32(<vscale x 8 x i16> %a)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @uunpklo_i64(<vscale x 4 x i32> %a) {
; CHECK-LABEL:  uunpklo_i64
; CHECK: uunpklo z0.d, z0.s
; CHECK-NEXT: ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.uunpklo.nxv2i64(<vscale x 4 x i32> %a)
  ret <vscale x 2 x i64> %res
}

;
; TRN1
;

define <vscale x 16 x i1> @trn1_b8(<vscale x 16 x i1> %a, <vscale x 16 x i1> %b) {
; CHECK-LABEL: trn1_b8:
; CHECK: trn1 p0.b, p0.b, p1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.trn1.nxv16i1(<vscale x 16 x i1> %a,
                                                                <vscale x 16 x i1> %b)
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @trn1_b16(<vscale x 8 x i1> %a, <vscale x 8 x i1> %b) {
; CHECK-LABEL: trn1_b16:
; CHECK: trn1 p0.h, p0.h, p1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.trn1.nxv8i1(<vscale x 8 x i1> %a,
                                                              <vscale x 8 x i1> %b)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @trn1_b32(<vscale x 4 x i1> %a, <vscale x 4 x i1> %b) {
; CHECK-LABEL: trn1_b32:
; CHECK: trn1 p0.s, p0.s, p1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.trn1.nxv4i1(<vscale x 4 x i1> %a,
                                                              <vscale x 4 x i1> %b)
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @trn1_b64(<vscale x 2 x i1> %a, <vscale x 2 x i1> %b) {
; CHECK-LABEL: trn1_b64:
; CHECK: trn1 p0.d, p0.d, p1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i1> @llvm.aarch64.sve.trn1.nxv2i1(<vscale x 2 x i1> %a,
                                                              <vscale x 2 x i1> %b)
  ret <vscale x 2 x i1> %out
}

define <vscale x 16 x i8> @trn1_i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: trn1_i8:
; CHECK: trn1 z0.b, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.trn1.nxv16i8(<vscale x 16 x i8> %a,
                                                                <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @trn1_i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: trn1_i16:
; CHECK: trn1 z0.h, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.trn1.nxv8i16(<vscale x 8 x i16> %a,
                                                                <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @trn1_i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: trn1_i32:
; CHECK: trn1 z0.s, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.trn1.nxv4i32(<vscale x 4 x i32> %a,
                                                                <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @trn1_i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: trn1_i64:
; CHECK: trn1 z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.trn1.nxv2i64(<vscale x 2 x i64> %a,
                                                                <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

define <vscale x 4 x half> @trn1_f16_v4(<vscale x 4 x half> %a, <vscale x 4 x half> %b) {
; CHECK-LABEL: trn1_f16_v4:
; CHECK: trn1 z0.s, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x half> @llvm.aarch64.sve.trn1.nxv4f16(<vscale x 4 x half> %a,
                                                                 <vscale x 4 x half> %b)
  ret <vscale x 4 x half> %out
}

define <vscale x 8 x half> @trn1_f16(<vscale x 8 x half> %a, <vscale x 8 x half> %b) {
; CHECK-LABEL: trn1_f16:
; CHECK: trn1 z0.h, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.trn1.nxv8f16(<vscale x 8 x half> %a,
                                                                 <vscale x 8 x half> %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @trn1_f32(<vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: trn1_f32:
; CHECK: trn1 z0.s, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.trn1.nxv4f32(<vscale x 4 x float> %a,
                                                                  <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @trn1_f64(<vscale x 2 x double> %a, <vscale x 2 x double> %b) {
; CHECK-LABEL: trn1_f64:
; CHECK: trn1 z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.trn1.nxv2f64(<vscale x 2 x double> %a,
                                                                   <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %out
}

;
; TRN2
;

define <vscale x 16 x i1> @trn2_b8(<vscale x 16 x i1> %a, <vscale x 16 x i1> %b) {
; CHECK-LABEL: trn2_b8:
; CHECK: trn2 p0.b, p0.b, p1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.trn2.nxv16i1(<vscale x 16 x i1> %a,
                                                                <vscale x 16 x i1> %b)
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @trn2_b16(<vscale x 8 x i1> %a, <vscale x 8 x i1> %b) {
; CHECK-LABEL: trn2_b16:
; CHECK: trn2 p0.h, p0.h, p1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.trn2.nxv8i1(<vscale x 8 x i1> %a,
                                                              <vscale x 8 x i1> %b)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @trn2_b32(<vscale x 4 x i1> %a, <vscale x 4 x i1> %b) {
; CHECK-LABEL: trn2_b32:
; CHECK: trn2 p0.s, p0.s, p1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.trn2.nxv4i1(<vscale x 4 x i1> %a,
                                                              <vscale x 4 x i1> %b)
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @trn2_b64(<vscale x 2 x i1> %a, <vscale x 2 x i1> %b) {
; CHECK-LABEL: trn2_b64:
; CHECK: trn2 p0.d, p0.d, p1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i1> @llvm.aarch64.sve.trn2.nxv2i1(<vscale x 2 x i1> %a,
                                                              <vscale x 2 x i1> %b)
  ret <vscale x 2 x i1> %out
}

define <vscale x 16 x i8> @trn2_i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: trn2_i8:
; CHECK: trn2 z0.b, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.trn2.nxv16i8(<vscale x 16 x i8> %a,
                                                                <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @trn2_i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: trn2_i16:
; CHECK: trn2 z0.h, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.trn2.nxv8i16(<vscale x 8 x i16> %a,
                                                                <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @trn2_i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: trn2_i32:
; CHECK: trn2 z0.s, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.trn2.nxv4i32(<vscale x 4 x i32> %a,
                                                                <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @trn2_i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: trn2_i64:
; CHECK: trn2 z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.trn2.nxv2i64(<vscale x 2 x i64> %a,
                                                                <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

define <vscale x 4 x half> @trn2_f16_v4(<vscale x 4 x half> %a, <vscale x 4 x half> %b) {
; CHECK-LABEL: trn2_f16_v4:
; CHECK: trn2 z0.s, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x half> @llvm.aarch64.sve.trn2.nxv4f16(<vscale x 4 x half> %a,
                                                                 <vscale x 4 x half> %b)
  ret <vscale x 4 x half> %out
}

define <vscale x 8 x half> @trn2_f16(<vscale x 8 x half> %a, <vscale x 8 x half> %b) {
; CHECK-LABEL: trn2_f16:
; CHECK: trn2 z0.h, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.trn2.nxv8f16(<vscale x 8 x half> %a,
                                                                 <vscale x 8 x half> %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @trn2_f32(<vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: trn2_f32:
; CHECK: trn2 z0.s, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.trn2.nxv4f32(<vscale x 4 x float> %a,
                                                                  <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @trn2_f64(<vscale x 2 x double> %a, <vscale x 2 x double> %b) {
; CHECK-LABEL: trn2_f64:
; CHECK: trn2 z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.trn2.nxv2f64(<vscale x 2 x double> %a,
                                                                   <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %out
}

;
; UZP1
;

define <vscale x 16 x i1> @uzp1_b8(<vscale x 16 x i1> %a, <vscale x 16 x i1> %b) {
; CHECK-LABEL: uzp1_b8:
; CHECK: uzp1 p0.b, p0.b, p1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.uzp1.nxv16i1(<vscale x 16 x i1> %a,
                                                                <vscale x 16 x i1> %b)
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @uzp1_b16(<vscale x 8 x i1> %a, <vscale x 8 x i1> %b) {
; CHECK-LABEL: uzp1_b16:
; CHECK: uzp1 p0.h, p0.h, p1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.uzp1.nxv8i1(<vscale x 8 x i1> %a,
                                                              <vscale x 8 x i1> %b)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @uzp1_b32(<vscale x 4 x i1> %a, <vscale x 4 x i1> %b) {
; CHECK-LABEL: uzp1_b32:
; CHECK: uzp1 p0.s, p0.s, p1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.uzp1.nxv4i1(<vscale x 4 x i1> %a,
                                                              <vscale x 4 x i1> %b)
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @uzp1_b64(<vscale x 2 x i1> %a, <vscale x 2 x i1> %b) {
; CHECK-LABEL: uzp1_b64:
; CHECK: uzp1 p0.d, p0.d, p1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i1> @llvm.aarch64.sve.uzp1.nxv2i1(<vscale x 2 x i1> %a,
                                                              <vscale x 2 x i1> %b)
  ret <vscale x 2 x i1> %out
}

define <vscale x 16 x i8> @uzp1_i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: uzp1_i8:
; CHECK: uzp1 z0.b, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.uzp1.nxv16i8(<vscale x 16 x i8> %a,
                                                                <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @uzp1_i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: uzp1_i16:
; CHECK: uzp1 z0.h, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.uzp1.nxv8i16(<vscale x 8 x i16> %a,
                                                                <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @uzp1_i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: uzp1_i32:
; CHECK: uzp1 z0.s, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.uzp1.nxv4i32(<vscale x 4 x i32> %a,
                                                                <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @uzp1_i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: uzp1_i64:
; CHECK: uzp1 z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.uzp1.nxv2i64(<vscale x 2 x i64> %a,
                                                                <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

define <vscale x 4 x half> @uzp1_f16_v4(<vscale x 4 x half> %a, <vscale x 4 x half> %b) {
; CHECK-LABEL: uzp1_f16_v4:
; CHECK: uzp1 z0.s, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x half> @llvm.aarch64.sve.uzp1.nxv4f16(<vscale x 4 x half> %a,
                                                                 <vscale x 4 x half> %b)
  ret <vscale x 4 x half> %out
}

define <vscale x 8 x half> @uzp1_f16(<vscale x 8 x half> %a, <vscale x 8 x half> %b) {
; CHECK-LABEL: uzp1_f16:
; CHECK: uzp1 z0.h, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.uzp1.nxv8f16(<vscale x 8 x half> %a,
                                                                 <vscale x 8 x half> %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @uzp1_f32(<vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: uzp1_f32:
; CHECK: uzp1 z0.s, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.uzp1.nxv4f32(<vscale x 4 x float> %a,
                                                                  <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @uzp1_f64(<vscale x 2 x double> %a, <vscale x 2 x double> %b) {
; CHECK-LABEL: uzp1_f64:
; CHECK: uzp1 z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.uzp1.nxv2f64(<vscale x 2 x double> %a,
                                                                   <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %out
}

;
; UZP2
;

define <vscale x 16 x i1> @uzp2_b8(<vscale x 16 x i1> %a, <vscale x 16 x i1> %b) {
; CHECK-LABEL: uzp2_b8:
; CHECK: uzp2 p0.b, p0.b, p1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.uzp2.nxv16i1(<vscale x 16 x i1> %a,
                                                                <vscale x 16 x i1> %b)
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @uzp2_b16(<vscale x 8 x i1> %a, <vscale x 8 x i1> %b) {
; CHECK-LABEL: uzp2_b16:
; CHECK: uzp2 p0.h, p0.h, p1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.uzp2.nxv8i1(<vscale x 8 x i1> %a,
                                                              <vscale x 8 x i1> %b)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @uzp2_b32(<vscale x 4 x i1> %a, <vscale x 4 x i1> %b) {
; CHECK-LABEL: uzp2_b32:
; CHECK: uzp2 p0.s, p0.s, p1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.uzp2.nxv4i1(<vscale x 4 x i1> %a,
                                                              <vscale x 4 x i1> %b)
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @uzp2_b64(<vscale x 2 x i1> %a, <vscale x 2 x i1> %b) {
; CHECK-LABEL: uzp2_b64:
; CHECK: uzp2 p0.d, p0.d, p1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i1> @llvm.aarch64.sve.uzp2.nxv2i1(<vscale x 2 x i1> %a,
                                                              <vscale x 2 x i1> %b)
  ret <vscale x 2 x i1> %out
}

define <vscale x 16 x i8> @uzp2_i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: uzp2_i8:
; CHECK: uzp2 z0.b, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.uzp2.nxv16i8(<vscale x 16 x i8> %a,
                                                                <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @uzp2_i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: uzp2_i16:
; CHECK: uzp2 z0.h, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.uzp2.nxv8i16(<vscale x 8 x i16> %a,
                                                                <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @uzp2_i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: uzp2_i32:
; CHECK: uzp2 z0.s, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.uzp2.nxv4i32(<vscale x 4 x i32> %a,
                                                                <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @uzp2_i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: uzp2_i64:
; CHECK: uzp2 z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.uzp2.nxv2i64(<vscale x 2 x i64> %a,
                                                                <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

define <vscale x 4 x half> @uzp2_f16_v4(<vscale x 4 x half> %a, <vscale x 4 x half> %b) {
; CHECK-LABEL: uzp2_f16_v4:
; CHECK: uzp2 z0.s, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x half> @llvm.aarch64.sve.uzp2.nxv4f16(<vscale x 4 x half> %a,
                                                                 <vscale x 4 x half> %b)
  ret <vscale x 4 x half> %out
}

define <vscale x 8 x half> @uzp2_f16(<vscale x 8 x half> %a, <vscale x 8 x half> %b) {
; CHECK-LABEL: uzp2_f16:
; CHECK: uzp2 z0.h, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.uzp2.nxv8f16(<vscale x 8 x half> %a,
                                                                 <vscale x 8 x half> %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @uzp2_f32(<vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: uzp2_f32:
; CHECK: uzp2 z0.s, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.uzp2.nxv4f32(<vscale x 4 x float> %a,
                                                                  <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @uzp2_f64(<vscale x 2 x double> %a, <vscale x 2 x double> %b) {
; CHECK-LABEL: uzp2_f64:
; CHECK: uzp2 z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.uzp2.nxv2f64(<vscale x 2 x double> %a,
                                                                   <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %out
}

;
; ZIP1
;

define <vscale x 16 x i1> @zip1_b8(<vscale x 16 x i1> %a, <vscale x 16 x i1> %b) {
; CHECK-LABEL: zip1_b8:
; CHECK: zip1 p0.b, p0.b, p1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.zip1.nxv16i1(<vscale x 16 x i1> %a,
                                                                <vscale x 16 x i1> %b)
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @zip1_b16(<vscale x 8 x i1> %a, <vscale x 8 x i1> %b) {
; CHECK-LABEL: zip1_b16:
; CHECK: zip1 p0.h, p0.h, p1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.zip1.nxv8i1(<vscale x 8 x i1> %a,
                                                              <vscale x 8 x i1> %b)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @zip1_b32(<vscale x 4 x i1> %a, <vscale x 4 x i1> %b) {
; CHECK-LABEL: zip1_b32:
; CHECK: zip1 p0.s, p0.s, p1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.zip1.nxv4i1(<vscale x 4 x i1> %a,
                                                              <vscale x 4 x i1> %b)
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @zip1_b64(<vscale x 2 x i1> %a, <vscale x 2 x i1> %b) {
; CHECK-LABEL: zip1_b64:
; CHECK: zip1 p0.d, p0.d, p1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i1> @llvm.aarch64.sve.zip1.nxv2i1(<vscale x 2 x i1> %a,
                                                              <vscale x 2 x i1> %b)
  ret <vscale x 2 x i1> %out
}

define <vscale x 16 x i8> @zip1_i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: zip1_i8:
; CHECK: zip1 z0.b, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.zip1.nxv16i8(<vscale x 16 x i8> %a,
                                                                <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @zip1_i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: zip1_i16:
; CHECK: zip1 z0.h, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.zip1.nxv8i16(<vscale x 8 x i16> %a,
                                                                <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @zip1_i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: zip1_i32:
; CHECK: zip1 z0.s, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.zip1.nxv4i32(<vscale x 4 x i32> %a,
                                                                <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @zip1_i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: zip1_i64:
; CHECK: zip1 z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.zip1.nxv2i64(<vscale x 2 x i64> %a,
                                                                <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

define <vscale x 4 x half> @zip1_f16_v4(<vscale x 4 x half> %a, <vscale x 4 x half> %b) {
; CHECK-LABEL: zip1_f16_v4:
; CHECK: zip1 z0.s, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x half> @llvm.aarch64.sve.zip1.nxv4f16(<vscale x 4 x half> %a,
                                                                 <vscale x 4 x half> %b)
  ret <vscale x 4 x half> %out
}

define <vscale x 8 x half> @zip1_f16(<vscale x 8 x half> %a, <vscale x 8 x half> %b) {
; CHECK-LABEL: zip1_f16:
; CHECK: zip1 z0.h, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.zip1.nxv8f16(<vscale x 8 x half> %a,
                                                                 <vscale x 8 x half> %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @zip1_f32(<vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: zip1_f32:
; CHECK: zip1 z0.s, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.zip1.nxv4f32(<vscale x 4 x float> %a,
                                                                  <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @zip1_f64(<vscale x 2 x double> %a, <vscale x 2 x double> %b) {
; CHECK-LABEL: zip1_f64:
; CHECK: zip1 z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.zip1.nxv2f64(<vscale x 2 x double> %a,
                                                                   <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %out
}

;
; ZIP2
;

define <vscale x 16 x i1> @zip2_b8(<vscale x 16 x i1> %a, <vscale x 16 x i1> %b) {
; CHECK-LABEL: zip2_b8:
; CHECK: zip2 p0.b, p0.b, p1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.zip2.nxv16i1(<vscale x 16 x i1> %a,
                                                                <vscale x 16 x i1> %b)
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @zip2_b16(<vscale x 8 x i1> %a, <vscale x 8 x i1> %b) {
; CHECK-LABEL: zip2_b16:
; CHECK: zip2 p0.h, p0.h, p1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.zip2.nxv8i1(<vscale x 8 x i1> %a,
                                                              <vscale x 8 x i1> %b)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @zip2_b32(<vscale x 4 x i1> %a, <vscale x 4 x i1> %b) {
; CHECK-LABEL: zip2_b32:
; CHECK: zip2 p0.s, p0.s, p1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.zip2.nxv4i1(<vscale x 4 x i1> %a,
                                                              <vscale x 4 x i1> %b)
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @zip2_b64(<vscale x 2 x i1> %a, <vscale x 2 x i1> %b) {
; CHECK-LABEL: zip2_b64:
; CHECK: zip2 p0.d, p0.d, p1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i1> @llvm.aarch64.sve.zip2.nxv2i1(<vscale x 2 x i1> %a,
                                                              <vscale x 2 x i1> %b)
  ret <vscale x 2 x i1> %out
}

define <vscale x 16 x i8> @zip2_i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: zip2_i8:
; CHECK: zip2 z0.b, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.zip2.nxv16i8(<vscale x 16 x i8> %a,
                                                                <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @zip2_i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: zip2_i16:
; CHECK: zip2 z0.h, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.zip2.nxv8i16(<vscale x 8 x i16> %a,
                                                                <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @zip2_i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: zip2_i32:
; CHECK: zip2 z0.s, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.zip2.nxv4i32(<vscale x 4 x i32> %a,
                                                                <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @zip2_i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: zip2_i64:
; CHECK: zip2 z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.zip2.nxv2i64(<vscale x 2 x i64> %a,
                                                                <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

define <vscale x 4 x half> @zip2_f16_v4(<vscale x 4 x half> %a, <vscale x 4 x half> %b) {
; CHECK-LABEL: zip2_f16_v4:
; CHECK: zip2 z0.s, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x half> @llvm.aarch64.sve.zip2.nxv4f16(<vscale x 4 x half> %a,
                                                                 <vscale x 4 x half> %b)
  ret <vscale x 4 x half> %out
}

define <vscale x 8 x half> @zip2_f16(<vscale x 8 x half> %a, <vscale x 8 x half> %b) {
; CHECK-LABEL: zip2_f16:
; CHECK: zip2 z0.h, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.zip2.nxv8f16(<vscale x 8 x half> %a,
                                                                 <vscale x 8 x half> %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @zip2_f32(<vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: zip2_f32:
; CHECK: zip2 z0.s, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.zip2.nxv4f32(<vscale x 4 x float> %a,
                                                                  <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @zip2_f64(<vscale x 2 x double> %a, <vscale x 2 x double> %b) {
; CHECK-LABEL: zip2_f64:
; CHECK: zip2 z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.zip2.nxv2f64(<vscale x 2 x double> %a,
                                                                   <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %out
}

declare <vscale x 16 x i8> @llvm.aarch64.sve.clasta.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.clasta.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.clasta.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.clasta.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 2 x i64>)
declare <vscale x 8 x half> @llvm.aarch64.sve.clasta.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.clasta.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.clasta.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>)

declare i8 @llvm.aarch64.sve.clasta.n.nxv16i8(<vscale x 16 x i1>, i8, <vscale x 16 x i8>)
declare i16 @llvm.aarch64.sve.clasta.n.nxv8i16(<vscale x 8 x i1>, i16, <vscale x 8 x i16>)
declare i32 @llvm.aarch64.sve.clasta.n.nxv4i32(<vscale x 4 x i1>, i32, <vscale x 4 x i32>)
declare i64 @llvm.aarch64.sve.clasta.n.nxv2i64(<vscale x 2 x i1>, i64, <vscale x 2 x i64>)
declare half @llvm.aarch64.sve.clasta.n.nxv8f16(<vscale x 8 x i1>, half, <vscale x 8 x half>)
declare float @llvm.aarch64.sve.clasta.n.nxv4f32(<vscale x 4 x i1>, float, <vscale x 4 x float>)
declare double @llvm.aarch64.sve.clasta.n.nxv2f64(<vscale x 2 x i1>, double, <vscale x 2 x double>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.clastb.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.clastb.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.clastb.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.clastb.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 2 x i64>)
declare <vscale x 8 x half> @llvm.aarch64.sve.clastb.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.clastb.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.clastb.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>)

declare i8 @llvm.aarch64.sve.clastb.n.nxv16i8(<vscale x 16 x i1>, i8, <vscale x 16 x i8>)
declare i16 @llvm.aarch64.sve.clastb.n.nxv8i16(<vscale x 8 x i1>, i16, <vscale x 8 x i16>)
declare i32 @llvm.aarch64.sve.clastb.n.nxv4i32(<vscale x 4 x i1>, i32, <vscale x 4 x i32>)
declare i64 @llvm.aarch64.sve.clastb.n.nxv2i64(<vscale x 2 x i1>, i64, <vscale x 2 x i64>)
declare half @llvm.aarch64.sve.clastb.n.nxv8f16(<vscale x 8 x i1>, half, <vscale x 8 x half>)
declare float @llvm.aarch64.sve.clastb.n.nxv4f32(<vscale x 4 x i1>, float, <vscale x 4 x float>)
declare double @llvm.aarch64.sve.clastb.n.nxv2f64(<vscale x 2 x i1>, double, <vscale x 2 x double>)

declare <vscale x 4 x i32> @llvm.aarch64.sve.compact.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.compact.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>)
declare <vscale x 4 x float> @llvm.aarch64.sve.compact.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.compact.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.ext.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>, i32)
declare <vscale x 8 x i16> @llvm.aarch64.sve.ext.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.ext.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.ext.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>, i32)
declare <vscale x 8 x half> @llvm.aarch64.sve.ext.nxv8f16(<vscale x 8 x half>, <vscale x 8 x half>, i32)
declare <vscale x 4 x float> @llvm.aarch64.sve.ext.nxv4f32(<vscale x 4 x float>, <vscale x 4 x float>, i32)
declare <vscale x 2 x double> @llvm.aarch64.sve.ext.nxv2f64(<vscale x 2 x double>, <vscale x 2 x double>, i32)

declare i8 @llvm.aarch64.sve.lasta.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>)
declare i16 @llvm.aarch64.sve.lasta.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>)
declare i32 @llvm.aarch64.sve.lasta.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>)
declare i64 @llvm.aarch64.sve.lasta.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>)
declare half @llvm.aarch64.sve.lasta.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>)
declare float @llvm.aarch64.sve.lasta.nxv2f32(<vscale x 2 x i1>, <vscale x 2 x float>)
declare float @llvm.aarch64.sve.lasta.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>)
declare double @llvm.aarch64.sve.lasta.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>)

declare i8 @llvm.aarch64.sve.lastb.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>)
declare i16 @llvm.aarch64.sve.lastb.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>)
declare i32 @llvm.aarch64.sve.lastb.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>)
declare i64 @llvm.aarch64.sve.lastb.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>)
declare half @llvm.aarch64.sve.lastb.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>)
declare float @llvm.aarch64.sve.lastb.nxv2f32(<vscale x 2 x i1>, <vscale x 2 x float>)
declare float @llvm.aarch64.sve.lastb.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>)
declare double @llvm.aarch64.sve.lastb.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>)

declare <vscale x 16 x i1> @llvm.aarch64.sve.rev.nxv16i1(<vscale x 16 x i1>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.rev.nxv8i1(<vscale x 8 x i1>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.rev.nxv4i1(<vscale x 4 x i1>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.rev.nxv2i1(<vscale x 2 x i1>)
declare <vscale x 16 x i8> @llvm.aarch64.sve.rev.nxv16i8(<vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.rev.nxv8i16(<vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.rev.nxv4i32(<vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.rev.nxv2i64(<vscale x 2 x i64>)
declare <vscale x 8 x half> @llvm.aarch64.sve.rev.nxv8f16(<vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.rev.nxv4f32(<vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.rev.nxv2f64(<vscale x 2 x double>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.splice.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.splice.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.splice.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.splice.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 2 x i64>)
declare <vscale x 8 x half> @llvm.aarch64.sve.splice.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.splice.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.splice.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>)

declare <vscale x 8 x i16> @llvm.aarch64.sve.sunpkhi.nxv8i16(<vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sunpkhi.nxv4i32(<vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sunpkhi.nxv2i64(<vscale x 4 x i32>)

declare <vscale x 8 x i16> @llvm.aarch64.sve.sunpklo.nxv8i16(<vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sunpklo.nxv4i32(<vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sunpklo.nxv2i64(<vscale x 4 x i32>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.tbl.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.tbl.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.tbl.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.tbl.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>)
declare <vscale x 8 x half> @llvm.aarch64.sve.tbl.nxv8f16(<vscale x 8 x half>, <vscale x 8 x i16>)
declare <vscale x 4 x float> @llvm.aarch64.sve.tbl.nxv4f32(<vscale x 4 x float>, <vscale x 4 x i32>)
declare <vscale x 2 x double> @llvm.aarch64.sve.tbl.nxv2f64(<vscale x 2 x double>, <vscale x 2 x i64>)

declare <vscale x 8 x i16> @llvm.aarch64.sve.uunpkhi.nxv8i16(<vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.uunpkhi.nxv4i32(<vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.uunpkhi.nxv2i64(<vscale x 4 x i32>)

declare <vscale x 8 x i16> @llvm.aarch64.sve.uunpklo.nxv8i16(<vscale x 16 x i8>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.uunpklo.nxv4i32(<vscale x 8 x i16>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.uunpklo.nxv2i64(<vscale x 4 x i32>)

declare <vscale x 16 x i1> @llvm.aarch64.sve.trn1.nxv16i1(<vscale x 16 x i1>, <vscale x 16 x i1>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.trn1.nxv8i1(<vscale x 8 x i1>, <vscale x 8 x i1>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.trn1.nxv4i1(<vscale x 4 x i1>, <vscale x 4 x i1>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.trn1.nxv2i1(<vscale x 2 x i1>, <vscale x 2 x i1>)
declare <vscale x 16 x i8> @llvm.aarch64.sve.trn1.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.trn1.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.trn1.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.trn1.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>)
declare <vscale x 4 x half> @llvm.aarch64.sve.trn1.nxv4f16(<vscale x 4 x half>, <vscale x 4 x half>)
declare <vscale x 8 x half> @llvm.aarch64.sve.trn1.nxv8f16(<vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.trn1.nxv4f32(<vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.trn1.nxv2f64(<vscale x 2 x double>, <vscale x 2 x double>)

declare <vscale x 16 x i1> @llvm.aarch64.sve.trn2.nxv16i1(<vscale x 16 x i1>, <vscale x 16 x i1>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.trn2.nxv8i1(<vscale x 8 x i1>, <vscale x 8 x i1>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.trn2.nxv4i1(<vscale x 4 x i1>, <vscale x 4 x i1>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.trn2.nxv2i1(<vscale x 2 x i1>, <vscale x 2 x i1>)
declare <vscale x 16 x i8> @llvm.aarch64.sve.trn2.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.trn2.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.trn2.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.trn2.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>)
declare <vscale x 4 x half> @llvm.aarch64.sve.trn2.nxv4f16(<vscale x 4 x half>, <vscale x 4 x half>)
declare <vscale x 8 x half> @llvm.aarch64.sve.trn2.nxv8f16(<vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.trn2.nxv4f32(<vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.trn2.nxv2f64(<vscale x 2 x double>, <vscale x 2 x double>)

declare <vscale x 16 x i1> @llvm.aarch64.sve.uzp1.nxv16i1(<vscale x 16 x i1>, <vscale x 16 x i1>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.uzp1.nxv8i1(<vscale x 8 x i1>, <vscale x 8 x i1>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.uzp1.nxv4i1(<vscale x 4 x i1>, <vscale x 4 x i1>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.uzp1.nxv2i1(<vscale x 2 x i1>, <vscale x 2 x i1>)
declare <vscale x 16 x i8> @llvm.aarch64.sve.uzp1.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.uzp1.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.uzp1.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.uzp1.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>)
declare <vscale x 4 x half> @llvm.aarch64.sve.uzp1.nxv4f16(<vscale x 4 x half>, <vscale x 4 x half>)
declare <vscale x 8 x half> @llvm.aarch64.sve.uzp1.nxv8f16(<vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.uzp1.nxv4f32(<vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.uzp1.nxv2f64(<vscale x 2 x double>, <vscale x 2 x double>)

declare <vscale x 16 x i1> @llvm.aarch64.sve.uzp2.nxv16i1(<vscale x 16 x i1>, <vscale x 16 x i1>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.uzp2.nxv8i1(<vscale x 8 x i1>, <vscale x 8 x i1>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.uzp2.nxv4i1(<vscale x 4 x i1>, <vscale x 4 x i1>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.uzp2.nxv2i1(<vscale x 2 x i1>, <vscale x 2 x i1>)
declare <vscale x 16 x i8> @llvm.aarch64.sve.uzp2.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.uzp2.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.uzp2.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.uzp2.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>)
declare <vscale x 4 x half> @llvm.aarch64.sve.uzp2.nxv4f16(<vscale x 4 x half>, <vscale x 4 x half>)
declare <vscale x 8 x half> @llvm.aarch64.sve.uzp2.nxv8f16(<vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.uzp2.nxv4f32(<vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.uzp2.nxv2f64(<vscale x 2 x double>, <vscale x 2 x double>)

declare <vscale x 16 x i1> @llvm.aarch64.sve.zip1.nxv16i1(<vscale x 16 x i1>, <vscale x 16 x i1>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.zip1.nxv8i1(<vscale x 8 x i1>, <vscale x 8 x i1>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.zip1.nxv4i1(<vscale x 4 x i1>, <vscale x 4 x i1>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.zip1.nxv2i1(<vscale x 2 x i1>, <vscale x 2 x i1>)
declare <vscale x 16 x i8> @llvm.aarch64.sve.zip1.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.zip1.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.zip1.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.zip1.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>)
declare <vscale x 4 x half> @llvm.aarch64.sve.zip1.nxv4f16(<vscale x 4 x half>, <vscale x 4 x half>)
declare <vscale x 8 x half> @llvm.aarch64.sve.zip1.nxv8f16(<vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.zip1.nxv4f32(<vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.zip1.nxv2f64(<vscale x 2 x double>, <vscale x 2 x double>)

declare <vscale x 16 x i1> @llvm.aarch64.sve.zip2.nxv16i1(<vscale x 16 x i1>, <vscale x 16 x i1>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.zip2.nxv8i1(<vscale x 8 x i1>, <vscale x 8 x i1>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.zip2.nxv4i1(<vscale x 4 x i1>, <vscale x 4 x i1>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.zip2.nxv2i1(<vscale x 2 x i1>, <vscale x 2 x i1>)
declare <vscale x 16 x i8> @llvm.aarch64.sve.zip2.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.zip2.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.zip2.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.zip2.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>)
declare <vscale x 4 x half> @llvm.aarch64.sve.zip2.nxv4f16(<vscale x 4 x half>, <vscale x 4 x half>)
declare <vscale x 8 x half> @llvm.aarch64.sve.zip2.nxv8f16(<vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.zip2.nxv4f32(<vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.zip2.nxv2f64(<vscale x 2 x double>, <vscale x 2 x double>)
