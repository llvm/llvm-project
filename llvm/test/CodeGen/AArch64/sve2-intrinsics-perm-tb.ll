; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve2 < %s | FileCheck %s

;
; TBL2
;

define <vscale x 16 x i8> @tbl2_b(<vscale x 16 x i8> %a, <vscale x 16 x i8> %unused,
                                  <vscale x 16 x i8> %b, <vscale x 16 x i8> %c) {
; CHECK-LABEL: tbl2_b:
; CHECK: mov z1.d, z0.d
; CHECK-NEXT: tbl z0.b, { z1.b, z2.b }, z3.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.tbl2.nxv16i8(<vscale x 16 x i8> %a,
                                                                <vscale x 16 x i8> %b,
                                                                <vscale x 16 x i8> %c)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @tbl2_h(<vscale x 8 x i16> %a, <vscale x 16 x i8> %unused,
                                  <vscale x 8 x i16> %b, <vscale x 8 x i16> %c) {
; CHECK-LABEL: tbl2_h:
; CHECK: mov z1.d, z0.d
; CHECK-NEXT: tbl z0.h, { z1.h, z2.h }, z3.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.tbl2.nxv8i16(<vscale x 8 x i16> %a,
                                                                <vscale x 8 x i16> %b,
                                                                <vscale x 8 x i16> %c)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @tbl2_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %unused,
                                  <vscale x 4 x i32> %b, <vscale x 4 x i32> %c) {
; CHECK-LABEL: tbl2_s:
; CHECK: mov z1.d, z0.d
; CHECK-NEXT: tbl z0.s, { z1.s, z2.s }, z3.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.tbl2.nxv4i32(<vscale x 4 x i32> %a,
                                                                <vscale x 4 x i32> %b,
                                                                <vscale x 4 x i32> %c)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @tbl2_d(<vscale x 2 x i64> %a, <vscale x 2 x i64> %unused,
                                  <vscale x 2 x i64> %b, <vscale x 2 x i64> %c) {
; CHECK-LABEL: tbl2_d:
; CHECK: mov z1.d, z0.d
; CHECK-NEXT: tbl z0.d, { z1.d, z2.d }, z3.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.tbl2.nxv2i64(<vscale x 2 x i64> %a,
                                                                <vscale x 2 x i64> %b,
                                                                <vscale x 2 x i64> %c)
  ret <vscale x 2 x i64> %out
}

define <vscale x 8 x half> @tbl2_fh(<vscale x 8 x half> %a, <vscale x 8 x half> %unused,
                                    <vscale x 8 x half> %b, <vscale x 8 x i16> %c) {
; CHECK-LABEL: tbl2_fh:
; CHECK: mov z1.d, z0.d
; CHECK-NEXT: tbl z0.h, { z1.h, z2.h }, z3.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.tbl2.nxv8f16(<vscale x 8 x half> %a,
                                                                 <vscale x 8 x half> %b,
                                                                 <vscale x 8 x i16> %c)
  ret <vscale x 8 x half> %out
}

define <vscale x 8 x bfloat> @tbl2_bf16(<vscale x 8 x bfloat> %a, <vscale x 8 x bfloat> %unused,
                                        <vscale x 8 x bfloat> %b, <vscale x 8 x i16> %c) #0 {
; CHECK-LABEL: tbl2_bf16:
; CHECK: mov z1.d, z0.d
; CHECK-NEXT: tbl z0.h, { z1.h, z2.h }, z3.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x bfloat> @llvm.aarch64.sve.tbl2.nxv8bf16(<vscale x 8 x bfloat> %a,
                                                                    <vscale x 8 x bfloat> %b,
                                                                    <vscale x 8 x i16> %c)
  ret <vscale x 8 x bfloat> %out
}

define <vscale x 4 x float> @tbl2_fs(<vscale x 4 x float> %a, <vscale x 4 x float> %unused,
                                     <vscale x 4 x float> %b, <vscale x 4 x i32> %c) {
; CHECK-LABEL: tbl2_fs:
; CHECK: mov z1.d, z0.d
; CHECK-NEXT: tbl z0.s, { z1.s, z2.s }, z3.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.tbl2.nxv4f32(<vscale x 4 x float> %a,
                                                                  <vscale x 4 x float> %b,
                                                                  <vscale x 4 x i32> %c)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @tbl2_fd(<vscale x 2 x double> %a, <vscale x 2 x double> %unused,
                                      <vscale x 2 x double> %b, <vscale x 2 x i64> %c) {
; CHECK-LABEL: tbl2_fd:
; CHECK: mov z1.d, z0.d
; CHECK-NEXT: tbl z0.d, { z1.d, z2.d }, z3.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.tbl2.nxv2f64(<vscale x 2 x double> %a,
                                                                   <vscale x 2 x double> %b,
                                                                   <vscale x 2 x i64> %c)
  ret <vscale x 2 x double> %out
}

;
; TBX
;

define <vscale x 16 x i8> @tbx_b(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b, <vscale x 16 x i8> %c) {
; CHECK-LABEL: tbx_b:
; CHECK: tbx z0.b, z1.b, z2.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.tbx.nxv16i8(<vscale x 16 x i8> %a,
                                                               <vscale x 16 x i8> %b,
                                                               <vscale x 16 x i8> %c)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @tbx_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b, <vscale x 8 x i16> %c) {
; CHECK-LABEL: tbx_h:
; CHECK: tbx z0.h, z1.h, z2.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.tbx.nxv8i16(<vscale x 8 x i16> %a,
                                                               <vscale x 8 x i16> %b,
                                                               <vscale x 8 x i16> %c)
  ret <vscale x 8 x i16> %out
}

define <vscale x 8 x half> @ftbx_h(<vscale x 8 x half> %a, <vscale x 8 x half> %b, <vscale x 8 x i16> %c) {
; CHECK-LABEL: ftbx_h:
; CHECK: tbx z0.h, z1.h, z2.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.tbx.nxv8f16(<vscale x 8 x half> %a,
                                                                <vscale x 8 x half> %b,
                                                                <vscale x 8 x i16> %c)
  ret <vscale x 8 x half> %out
}

define <vscale x 8 x bfloat> @ftbx_h_bf16(<vscale x 8 x bfloat> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x i16> %c) #0 {
; CHECK-LABEL: ftbx_h_bf16:
; CHECK: tbx z0.h, z1.h, z2.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x bfloat> @llvm.aarch64.sve.tbx.nxv8bf16(<vscale x 8 x bfloat> %a,
                                                                   <vscale x 8 x bfloat> %b,
                                                                   <vscale x 8 x i16> %c)
  ret <vscale x 8 x bfloat> %out
}

define <vscale x 4 x i32> @tbx_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b, <vscale x 4 x i32> %c) {
; CHECK-LABEL: tbx_s:
; CHECK: tbx z0.s, z1.s, z2.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.tbx.nxv4i32(<vscale x 4 x i32> %a,
                                                               <vscale x 4 x i32> %b,
                                                               <vscale x 4 x i32> %c)
  ret <vscale x 4 x i32> %out
}

define <vscale x 4 x float> @ftbx_s(<vscale x 4 x float> %a, <vscale x 4 x float> %b, <vscale x 4 x i32> %c) {
; CHECK-LABEL: ftbx_s:
; CHECK: tbx z0.s, z1.s, z2.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.tbx.nxv4f32(<vscale x 4 x float> %a,
                                                                 <vscale x 4 x float> %b,
                                                                 <vscale x 4 x i32> %c)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x i64> @tbx_d(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b, <vscale x 2 x i64> %c) {
; CHECK-LABEL: tbx_d:
; CHECK: tbx z0.d, z1.d, z2.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.tbx.nxv2i64(<vscale x 2 x i64> %a,
                                                               <vscale x 2 x i64> %b,
                                                               <vscale x 2 x i64> %c)
  ret <vscale x 2 x i64> %out
}

define <vscale x 2 x double> @ftbx_d(<vscale x 2 x double> %a, <vscale x 2 x double> %b, <vscale x 2 x i64> %c) {
; CHECK-LABEL: ftbx_d:
; CHECK: tbx z0.d, z1.d, z2.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.tbx.nxv2f64(<vscale x 2 x double> %a,
                                                                  <vscale x 2 x double> %b,
                                                                  <vscale x 2 x i64> %c)
  ret <vscale x 2 x double> %out
}

declare <vscale x 16 x i8> @llvm.aarch64.sve.tbl2.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.tbl2.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.tbl2.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.tbl2.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 8 x half> @llvm.aarch64.sve.tbl2.nxv8f16(<vscale x 8 x half>, <vscale x 8 x half>, <vscale x 8 x i16>)
declare <vscale x 4 x float> @llvm.aarch64.sve.tbl2.nxv4f32(<vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x i32>)
declare <vscale x 2 x double> @llvm.aarch64.sve.tbl2.nxv2f64(<vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x i64>)

declare <vscale x 8 x bfloat> @llvm.aarch64.sve.tbl2.nxv8bf16(<vscale x 8 x bfloat>, <vscale x 8 x bfloat>, <vscale x 8 x i16>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.tbx.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.tbx.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.tbx.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.tbx.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 8 x half> @llvm.aarch64.sve.tbx.nxv8f16(<vscale x 8 x half>, <vscale x 8 x half>, <vscale x 8 x i16>)
declare <vscale x 4 x float> @llvm.aarch64.sve.tbx.nxv4f32(<vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x i32>)
declare <vscale x 2 x double> @llvm.aarch64.sve.tbx.nxv2f64(<vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x i64>)

declare <vscale x 8 x bfloat> @llvm.aarch64.sve.tbx.nxv8bf16(<vscale x 8 x bfloat>, <vscale x 8 x bfloat>, <vscale x 8 x i16>)

; +bf16 is required for the bfloat version.
attributes #0 = { "target-features"="+sve,+bf16" }
