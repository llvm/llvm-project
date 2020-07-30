; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; WARN-NOT: warning

;
; DUP
;

define <vscale x 16 x i8> @dup_i8(<vscale x 16 x i8> %a, <vscale x 16 x i1> %pg, i8 %b) {
; CHECK-LABEL: dup_i8:
; CHECK: mov z0.b, p0/m, w0
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.dup.nxv16i8(<vscale x 16 x i8> %a,
                                                               <vscale x 16 x i1> %pg,
                                                               i8 %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @dup_i16(<vscale x 8 x i16> %a, <vscale x 8 x i1> %pg, i16 %b) {
; CHECK-LABEL: dup_i16:
; CHECK: mov z0.h, p0/m, w0
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.dup.nxv8i16(<vscale x 8 x i16> %a,
                                                               <vscale x 8 x i1> %pg,
                                                               i16 %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @dup_i32(<vscale x 4 x i32> %a, <vscale x 4 x i1> %pg, i32 %b) {
; CHECK-LABEL: dup_i32:
; CHECK: mov z0.s, p0/m, w0
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.dup.nxv4i32(<vscale x 4 x i32> %a,
                                                               <vscale x 4 x i1> %pg,
                                                               i32 %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @dup_i64(<vscale x 2 x i64> %a, <vscale x 2 x i1> %pg, i64 %b) {
; CHECK-LABEL: dup_i64:
; CHECK: mov z0.d, p0/m, x0
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.nxv2i64(<vscale x 2 x i64> %a,
                                                               <vscale x 2 x i1> %pg,
                                                               i64 %b)
  ret <vscale x 2 x i64> %out
}

define <vscale x 8 x half> @dup_f16(<vscale x 8 x half> %a, <vscale x 8 x i1> %pg, half %b) {
; CHECK-LABEL: dup_f16:
; CHECK: mov z0.h, p0/m, h1
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.dup.nxv8f16(<vscale x 8 x half> %a,
                                                                <vscale x 8 x i1> %pg,
                                                                half %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 8 x bfloat> @dup_bf16(<vscale x 8 x bfloat> %a, <vscale x 8 x i1> %pg, bfloat %b) #0 {
; CHECK-LABEL: dup_bf16:
; CHECK: mov z0.h, p0/m, h1
; CHECK-NEXT: ret
  %out = call <vscale x 8 x bfloat> @llvm.aarch64.sve.dup.nxv8bf16(<vscale x 8 x bfloat> %a,
                                                                   <vscale x 8 x i1> %pg,
                                                                   bfloat %b)
  ret <vscale x 8 x bfloat> %out
}

define <vscale x 4 x float> @dup_f32(<vscale x 4 x float> %a, <vscale x 4 x i1> %pg, float %b) {
; CHECK-LABEL: dup_f32:
; CHECK: mov z0.s, p0/m, s1
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.dup.nxv4f32(<vscale x 4 x float> %a,
                                                                 <vscale x 4 x i1> %pg,
                                                                 float %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @dup_f64(<vscale x 2 x double> %a, <vscale x 2 x i1> %pg, double %b) {
; CHECK-LABEL: dup_f64:
; CHECK: mov z0.d, p0/m, d1
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.dup.nxv2f64(<vscale x 2 x double> %a,
                                                                  <vscale x 2 x i1> %pg,
                                                                  double %b)
  ret <vscale x 2 x double> %out
}

define <vscale x 8 x bfloat> @test_svdup_n_bf16_z(<vscale x 8 x i1> %pg, bfloat %op) #0 {
; CHECK-LABEL: test_svdup_n_bf16_z:
; CHECK: mov z1.h, #0
; CHECK: mov z1.h, p0/m, h0
; CHECK: mov z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 8 x bfloat> @llvm.aarch64.sve.dup.nxv8bf16(<vscale x 8 x bfloat> zeroinitializer, <vscale x 8 x i1> %pg, bfloat %op)
  ret <vscale x 8 x bfloat> %out
}

define <vscale x 8 x bfloat> @test_svdup_n_bf16_m(<vscale x 8 x bfloat> %inactive, <vscale x 8 x i1> %pg, bfloat %op) #0 {
; CHECK-LABEL: test_svdup_n_bf16_m:
; CHECK: mov z0.h, p0/m, h1
; CHECK-NEXT: ret
  %out = call <vscale x 8 x bfloat> @llvm.aarch64.sve.dup.nxv8bf16(<vscale x 8 x bfloat> %inactive, <vscale x 8 x i1> %pg, bfloat %op)
  ret <vscale x 8 x bfloat> %out
}


define <vscale x 8 x bfloat> @test_svdup_n_bf16_x(<vscale x 8 x i1> %pg, bfloat %op) #0 {
; CHECK-LABEL: test_svdup_n_bf16_x:
; CHECK: mov z0.h, p0/m, h0
; CHECK-NEXT: ret
  %out = call <vscale x 8 x bfloat> @llvm.aarch64.sve.dup.nxv8bf16(<vscale x 8 x bfloat> undef, <vscale x 8 x i1> %pg, bfloat %op)
  ret <vscale x 8 x bfloat> %out
}

declare <vscale x 16 x i8> @llvm.aarch64.sve.dup.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i1>, i8)
declare <vscale x 8 x i16> @llvm.aarch64.sve.dup.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i1>, i16)
declare <vscale x 4 x i32> @llvm.aarch64.sve.dup.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i1>, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.dup.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i1>, i64)
declare <vscale x 8 x half> @llvm.aarch64.sve.dup.nxv8f16(<vscale x 8 x half>, <vscale x 8 x i1>, half)
declare <vscale x 8 x bfloat> @llvm.aarch64.sve.dup.nxv8bf16(<vscale x 8 x bfloat>, <vscale x 8 x i1>, bfloat)
declare <vscale x 4 x float> @llvm.aarch64.sve.dup.nxv4f32(<vscale x 4 x float>, <vscale x 4 x i1>, float)
declare <vscale x 2 x double> @llvm.aarch64.sve.dup.nxv2f64(<vscale x 2 x double>, <vscale x 2 x i1>, double)

; +bf16 is required for the bfloat version.
attributes #0 = { "target-features"="+sve,+bf16" }
