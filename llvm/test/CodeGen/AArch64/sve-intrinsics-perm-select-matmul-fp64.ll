; RUN: llc -mtriple=aarch64-none-linux-gnu -mattr=+sve,+f64mm -asm-verbose=0 < %s -o - | FileCheck %s

;
; TRN1Q
;

define <vscale x 16 x i8> @trn1_i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) nounwind {
; CHECK-LABEL: trn1_i8:
; CHECK-NEXT:  trn1 z0.q, z0.q, z1.q
; CHECK-NEXT:  ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.trn1q.nxv16i8(<vscale x 16 x i8> %a,
                                                                 <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @trn1_i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) nounwind {
; CHECK-LABEL: trn1_i16:
; CHECK-NEXT:  trn1 z0.q, z0.q, z1.q
; CHECK-NEXT:  ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.trn1q.nxv8i16(<vscale x 8 x i16> %a,
                                                                 <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @trn1_i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) nounwind {
; CHECK-LABEL: trn1_i32:
; CHECK-NEXT:  trn1 z0.q, z0.q, z1.q
; CHECK-NEXT:  ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.trn1q.nxv4i32(<vscale x 4 x i32> %a,
                                                                 <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @trn1_i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) nounwind {
; CHECK-LABEL: trn1_i64:
; CHECK-NEXT:  trn1 z0.q, z0.q, z1.q
; CHECK-NEXT:  ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.trn1q.nxv2i64(<vscale x 2 x i64> %a,
                                                                 <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

define <vscale x 8 x half> @trn1_f16(<vscale x 8 x half> %a, <vscale x 8 x half> %b) nounwind {
; CHECK-LABEL: trn1_f16:
; CHECK-NEXT:  trn1 z0.q, z0.q, z1.q
; CHECK-NEXT:  ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.trn1q.nxv8f16(<vscale x 8 x half> %a,
                                                                  <vscale x 8 x half> %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 8 x bfloat> @trn1_bf16(<vscale x 8 x bfloat> %a, <vscale x 8 x bfloat> %b) nounwind #0 {
; CHECK-LABEL: trn1_bf16:
; CHECK-NEXT:  trn1 z0.q, z0.q, z1.q
; CHECK-NEXT:  ret
  %out = call <vscale x 8 x bfloat> @llvm.aarch64.sve.trn1q.nxv8bf16(<vscale x 8 x bfloat> %a,
                                                                     <vscale x 8 x bfloat> %b)
  ret <vscale x 8 x bfloat> %out
}

define <vscale x 4 x float> @trn1_f32(<vscale x 4 x float> %a, <vscale x 4 x float> %b) nounwind {
; CHECK-LABEL: trn1_f32:
; CHECK-NEXT:  trn1 z0.q, z0.q, z1.q
; CHECK-NEXT:  ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.trn1q.nxv4f32(<vscale x 4 x float> %a,
                                                                   <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @trn1_f64(<vscale x 2 x double> %a, <vscale x 2 x double> %b) nounwind {
; CHECK-LABEL: trn1_f64:
; CHECK-NEXT:  trn1 z0.q, z0.q, z1.q
; CHECK-NEXT:  ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.trn1q.nxv2f64(<vscale x 2 x double> %a,
                                                                    <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %out
}

;
; TRN2Q
;

define <vscale x 16 x i8> @trn2_i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) nounwind {
; CHECK-LABEL: trn2_i8:
; CHECK-NEXT:  trn2 z0.q, z0.q, z1.q
; CHECK-NEXT:  ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.trn2q.nxv16i8(<vscale x 16 x i8> %a,
                                                                 <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @trn2_i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) nounwind {
; CHECK-LABEL: trn2_i16:
; CHECK-NEXT:  trn2 z0.q, z0.q, z1.q
; CHECK-NEXT:  ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.trn2q.nxv8i16(<vscale x 8 x i16> %a,
                                                                 <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @trn2_i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) nounwind {
; CHECK-LABEL: trn2_i32:
; CHECK-NEXT:  trn2 z0.q, z0.q, z1.q
; CHECK-NEXT:  ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.trn2q.nxv4i32(<vscale x 4 x i32> %a,
                                                                 <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @trn2_i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) nounwind {
; CHECK-LABEL: trn2_i64:
; CHECK-NEXT:  trn2 z0.q, z0.q, z1.q
; CHECK-NEXT:  ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.trn2q.nxv2i64(<vscale x 2 x i64> %a,
                                                                 <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

define <vscale x 8 x half> @trn2_f16(<vscale x 8 x half> %a, <vscale x 8 x half> %b) nounwind {
; CHECK-LABEL: trn2_f16:
; CHECK-NEXT:  trn2 z0.q, z0.q, z1.q
; CHECK-NEXT:  ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.trn2q.nxv8f16(<vscale x 8 x half> %a,
                                                                  <vscale x 8 x half> %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 8 x bfloat> @trn2_bf16(<vscale x 8 x bfloat> %a, <vscale x 8 x bfloat> %b) nounwind #0 {
; CHECK-LABEL: trn2_bf16:
; CHECK-NEXT:  trn2 z0.q, z0.q, z1.q
; CHECK-NEXT:  ret
  %out = call <vscale x 8 x bfloat> @llvm.aarch64.sve.trn2q.nxv8bf16(<vscale x 8 x bfloat> %a,
                                                                     <vscale x 8 x bfloat> %b)
  ret <vscale x 8 x bfloat> %out
}

define <vscale x 4 x float> @trn2_f32(<vscale x 4 x float> %a, <vscale x 4 x float> %b) nounwind {
; CHECK-LABEL: trn2_f32:
; CHECK-NEXT:  trn2 z0.q, z0.q, z1.q
; CHECK-NEXT:  ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.trn2q.nxv4f32(<vscale x 4 x float> %a,
                                                                   <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @trn2_f64(<vscale x 2 x double> %a, <vscale x 2 x double> %b) nounwind {
; CHECK-LABEL: trn2_f64:
; CHECK-NEXT:  trn2 z0.q, z0.q, z1.q
; CHECK-NEXT:  ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.trn2q.nxv2f64(<vscale x 2 x double> %a,
                                                                    <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %out
}

;
; UZP1Q
;

define <vscale x 16 x i8> @uzp1_i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) nounwind {
; CHECK-LABEL: uzp1_i8:
; CHECK-NEXT:  uzp1 z0.q, z0.q, z1.q
; CHECK-NEXT:  ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.uzp1q.nxv16i8(<vscale x 16 x i8> %a,
                                                                 <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @uzp1_i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) nounwind {
; CHECK-LABEL: uzp1_i16:
; CHECK-NEXT:  uzp1 z0.q, z0.q, z1.q
; CHECK-NEXT:  ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.uzp1q.nxv8i16(<vscale x 8 x i16> %a,
                                                                 <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @uzp1_i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) nounwind {
; CHECK-LABEL: uzp1_i32:
; CHECK-NEXT:  uzp1 z0.q, z0.q, z1.q
; CHECK-NEXT:  ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.uzp1q.nxv4i32(<vscale x 4 x i32> %a,
                                                                 <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @uzp1_i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) nounwind {
; CHECK-LABEL: uzp1_i64:
; CHECK-NEXT:  uzp1 z0.q, z0.q, z1.q
; CHECK-NEXT:  ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.uzp1q.nxv2i64(<vscale x 2 x i64> %a,
                                                                 <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

define <vscale x 8 x half> @uzp1_f16(<vscale x 8 x half> %a, <vscale x 8 x half> %b) nounwind {
; CHECK-LABEL: uzp1_f16:
; CHECK-NEXT:  uzp1 z0.q, z0.q, z1.q
; CHECK-NEXT:  ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.uzp1q.nxv8f16(<vscale x 8 x half> %a,
                                                                  <vscale x 8 x half> %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 8 x bfloat> @uzp1_bf16(<vscale x 8 x bfloat> %a, <vscale x 8 x bfloat> %b) nounwind #0 {
; CHECK-LABEL: uzp1_bf16:
; CHECK-NEXT:  uzp1 z0.q, z0.q, z1.q
; CHECK-NEXT:  ret
  %out = call <vscale x 8 x bfloat> @llvm.aarch64.sve.uzp1q.nxv8bf16(<vscale x 8 x bfloat> %a,
                                                                     <vscale x 8 x bfloat> %b)
  ret <vscale x 8 x bfloat> %out
}

define <vscale x 4 x float> @uzp1_f32(<vscale x 4 x float> %a, <vscale x 4 x float> %b) nounwind {
; CHECK-LABEL: uzp1_f32:
; CHECK-NEXT:  uzp1 z0.q, z0.q, z1.q
; CHECK-NEXT:  ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.uzp1q.nxv4f32(<vscale x 4 x float> %a,
                                                                   <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @uzp1_f64(<vscale x 2 x double> %a, <vscale x 2 x double> %b) nounwind {
; CHECK-LABEL: uzp1_f64:
; CHECK-NEXT:  uzp1 z0.q, z0.q, z1.q
; CHECK-NEXT:  ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.uzp1q.nxv2f64(<vscale x 2 x double> %a,
                                                                    <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %out
}

;
; UZP2Q
;

define <vscale x 16 x i8> @uzp2_i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) nounwind {
; CHECK-LABEL: uzp2_i8:
; CHECK-NEXT:  uzp2 z0.q, z0.q, z1.q
; CHECK-NEXT:  ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.uzp2q.nxv16i8(<vscale x 16 x i8> %a,
                                                                 <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @uzp2_i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) nounwind {
; CHECK-LABEL: uzp2_i16:
; CHECK-NEXT:  uzp2 z0.q, z0.q, z1.q
; CHECK-NEXT:  ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.uzp2q.nxv8i16(<vscale x 8 x i16> %a,
                                                                 <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @uzp2_i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) nounwind {
; CHECK-LABEL: uzp2_i32:
; CHECK-NEXT:  uzp2 z0.q, z0.q, z1.q
; CHECK-NEXT:  ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.uzp2q.nxv4i32(<vscale x 4 x i32> %a,
                                                                 <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @uzp2_i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) nounwind {
; CHECK-LABEL: uzp2_i64:
; CHECK-NEXT:  uzp2 z0.q, z0.q, z1.q
; CHECK-NEXT:  ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.uzp2q.nxv2i64(<vscale x 2 x i64> %a,
                                                                 <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

define <vscale x 8 x half> @uzp2_f16(<vscale x 8 x half> %a, <vscale x 8 x half> %b) nounwind {
; CHECK-LABEL: uzp2_f16:
; CHECK-NEXT:  uzp2 z0.q, z0.q, z1.q
; CHECK-NEXT:  ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.uzp2q.nxv8f16(<vscale x 8 x half> %a,
                                                                  <vscale x 8 x half> %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 8 x bfloat> @uzp2_bf16(<vscale x 8 x bfloat> %a, <vscale x 8 x bfloat> %b) nounwind #0 {
; CHECK-LABEL: uzp2_bf16:
; CHECK-NEXT:  uzp2 z0.q, z0.q, z1.q
; CHECK-NEXT:  ret
  %out = call <vscale x 8 x bfloat> @llvm.aarch64.sve.uzp2q.nxv8bf16(<vscale x 8 x bfloat> %a,
                                                                     <vscale x 8 x bfloat> %b)
  ret <vscale x 8 x bfloat> %out
}

define <vscale x 4 x float> @uzp2_f32(<vscale x 4 x float> %a, <vscale x 4 x float> %b) nounwind {
; CHECK-LABEL: uzp2_f32:
; CHECK-NEXT:  uzp2 z0.q, z0.q, z1.q
; CHECK-NEXT:  ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.uzp2q.nxv4f32(<vscale x 4 x float> %a,
                                                                   <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @uzp2_f64(<vscale x 2 x double> %a, <vscale x 2 x double> %b) nounwind {
; CHECK-LABEL: uzp2_f64:
; CHECK-NEXT:  uzp2 z0.q, z0.q, z1.q
; CHECK-NEXT:  ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.uzp2q.nxv2f64(<vscale x 2 x double> %a,
                                                                    <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %out
}

;
; ZIP1Q
;

define <vscale x 16 x i8> @zip1_i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) nounwind {
; CHECK-LABEL: zip1_i8:
; CHECK-NEXT:  zip1 z0.q, z0.q, z1.q
; CHECK-NEXT:  ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.zip1q.nxv16i8(<vscale x 16 x i8> %a,
                                                                <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @zip1_i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) nounwind {
; CHECK-LABEL: zip1_i16:
; CHECK-NEXT:  zip1 z0.q, z0.q, z1.q
; CHECK-NEXT:  ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.zip1q.nxv8i16(<vscale x 8 x i16> %a,
                                                                 <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @zip1_i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) nounwind {
; CHECK-LABEL: zip1_i32:
; CHECK-NEXT:  zip1 z0.q, z0.q, z1.q
; CHECK-NEXT:  ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.zip1q.nxv4i32(<vscale x 4 x i32> %a,
                                                                 <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @zip1_i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) nounwind {
; CHECK-LABEL: zip1_i64:
; CHECK-NEXT:  zip1 z0.q, z0.q, z1.q
; CHECK-NEXT:  ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.zip1q.nxv2i64(<vscale x 2 x i64> %a,
                                                                 <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

define <vscale x 8 x half> @zip1_f16(<vscale x 8 x half> %a, <vscale x 8 x half> %b) nounwind {
; CHECK-LABEL: zip1_f16:
; CHECK-NEXT:  zip1 z0.q, z0.q, z1.q
; CHECK-NEXT:  ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.zip1q.nxv8f16(<vscale x 8 x half> %a,
                                                                  <vscale x 8 x half> %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 8 x bfloat> @zip1_bf16(<vscale x 8 x bfloat> %a, <vscale x 8 x bfloat> %b) nounwind #0 {
; CHECK-LABEL: zip1_bf16:
; CHECK-NEXT:  zip1 z0.q, z0.q, z1.q
; CHECK-NEXT:  ret
  %out = call <vscale x 8 x bfloat> @llvm.aarch64.sve.zip1q.nxv8bf16(<vscale x 8 x bfloat> %a,
                                                                     <vscale x 8 x bfloat> %b)
  ret <vscale x 8 x bfloat> %out
}

define <vscale x 4 x float> @zip1_f32(<vscale x 4 x float> %a, <vscale x 4 x float> %b) nounwind {
; CHECK-LABEL: zip1_f32:
; CHECK-NEXT:  zip1 z0.q, z0.q, z1.q
; CHECK-NEXT:  ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.zip1q.nxv4f32(<vscale x 4 x float> %a,
                                                                   <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @zip1_f64(<vscale x 2 x double> %a, <vscale x 2 x double> %b) nounwind {
; CHECK-LABEL: zip1_f64:
; CHECK-NEXT:  zip1 z0.q, z0.q, z1.q
; CHECK-NEXT:  ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.zip1q.nxv2f64(<vscale x 2 x double> %a,
                                                                    <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %out
}

;
; ZIP2Q
;

define <vscale x 16 x i8> @zip2_i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) nounwind {
; CHECK-LABEL: zip2_i8:
; CHECK-NEXT:  zip2 z0.q, z0.q, z1.q
; CHECK-NEXT:  ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.zip2q.nxv16i8(<vscale x 16 x i8> %a,
                                                                 <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @zip2_i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) nounwind {
; CHECK-LABEL: zip2_i16:
; CHECK-NEXT:  zip2 z0.q, z0.q, z1.q
; CHECK-NEXT:  ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.zip2q.nxv8i16(<vscale x 8 x i16> %a,
                                                                 <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @zip2_i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) nounwind {
; CHECK-LABEL: zip2_i32:
; CHECK-NEXT:  zip2 z0.q, z0.q, z1.q
; CHECK-NEXT:  ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.zip2q.nxv4i32(<vscale x 4 x i32> %a,
                                                                 <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @zip2_i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) nounwind {
; CHECK-LABEL: zip2_i64:
; CHECK-NEXT:  zip2 z0.q, z0.q, z1.q
; CHECK-NEXT:  ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.zip2q.nxv2i64(<vscale x 2 x i64> %a,
                                                                 <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

define <vscale x 8 x half> @zip2_f16(<vscale x 8 x half> %a, <vscale x 8 x half> %b) nounwind {
; CHECK-LABEL: zip2_f16:
; CHECK-NEXT:  zip2 z0.q, z0.q, z1.q
; CHECK-NEXT:  ret
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.zip2q.nxv8f16(<vscale x 8 x half> %a,
                                                                  <vscale x 8 x half> %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 8 x bfloat> @zip2_bf16(<vscale x 8 x bfloat> %a, <vscale x 8 x bfloat> %b) nounwind #0 {
; CHECK-LABEL: zip2_bf16:
; CHECK-NEXT:  zip2 z0.q, z0.q, z1.q
; CHECK-NEXT:  ret
  %out = call <vscale x 8 x bfloat> @llvm.aarch64.sve.zip2q.nxv8bf16(<vscale x 8 x bfloat> %a,
                                                                     <vscale x 8 x bfloat> %b)
  ret <vscale x 8 x bfloat> %out
}

define <vscale x 4 x float> @zip2_f32(<vscale x 4 x float> %a, <vscale x 4 x float> %b) nounwind {
; CHECK-LABEL: zip2_f32:
; CHECK-NEXT:  zip2 z0.q, z0.q, z1.q
; CHECK-NEXT:  ret
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.zip2q.nxv4f32(<vscale x 4 x float> %a,
                                                                   <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @zip2_f64(<vscale x 2 x double> %a, <vscale x 2 x double> %b) nounwind {
; CHECK-LABEL: zip2_f64:
; CHECK-NEXT:  zip2 z0.q, z0.q, z1.q
; CHECK-NEXT:  ret
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.zip2q.nxv2f64(<vscale x 2 x double> %a,
                                                                    <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %out
}


declare <vscale x 2 x double> @llvm.aarch64.sve.trn1q.nxv2f64(<vscale x 2 x double>, <vscale x 2 x double>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.trn1q.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>)
declare <vscale x 4 x float> @llvm.aarch64.sve.trn1q.nxv4f32(<vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.trn1q.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 8 x bfloat> @llvm.aarch64.sve.trn1q.nxv8bf16(<vscale x 8 x bfloat>, <vscale x 8 x bfloat>)
declare <vscale x 8 x half> @llvm.aarch64.sve.trn1q.nxv8f16(<vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.trn1q.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 16 x i8> @llvm.aarch64.sve.trn1q.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>)

declare <vscale x 2 x double> @llvm.aarch64.sve.trn2q.nxv2f64(<vscale x 2 x double>, <vscale x 2 x double>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.trn2q.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>)
declare <vscale x 4 x float> @llvm.aarch64.sve.trn2q.nxv4f32(<vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.trn2q.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 8 x bfloat> @llvm.aarch64.sve.trn2q.nxv8bf16(<vscale x 8 x bfloat>, <vscale x 8 x bfloat>)
declare <vscale x 8 x half> @llvm.aarch64.sve.trn2q.nxv8f16(<vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.trn2q.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 16 x i8> @llvm.aarch64.sve.trn2q.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>)

declare <vscale x 2 x double> @llvm.aarch64.sve.uzp1q.nxv2f64(<vscale x 2 x double>, <vscale x 2 x double>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.uzp1q.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>)
declare <vscale x 4 x float> @llvm.aarch64.sve.uzp1q.nxv4f32(<vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.uzp1q.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 8 x bfloat> @llvm.aarch64.sve.uzp1q.nxv8bf16(<vscale x 8 x bfloat>, <vscale x 8 x bfloat>)
declare <vscale x 8 x half> @llvm.aarch64.sve.uzp1q.nxv8f16(<vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.uzp1q.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 16 x i8> @llvm.aarch64.sve.uzp1q.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>)

declare <vscale x 2 x double> @llvm.aarch64.sve.uzp2q.nxv2f64(<vscale x 2 x double>, <vscale x 2 x double>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.uzp2q.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>)
declare <vscale x 4 x float> @llvm.aarch64.sve.uzp2q.nxv4f32(<vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.uzp2q.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 8 x bfloat> @llvm.aarch64.sve.uzp2q.nxv8bf16(<vscale x 8 x bfloat>, <vscale x 8 x bfloat>)
declare <vscale x 8 x half> @llvm.aarch64.sve.uzp2q.nxv8f16(<vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.uzp2q.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 16 x i8> @llvm.aarch64.sve.uzp2q.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>)

declare <vscale x 2 x double> @llvm.aarch64.sve.zip1q.nxv2f64(<vscale x 2 x double>, <vscale x 2 x double>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.zip1q.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>)
declare <vscale x 4 x float> @llvm.aarch64.sve.zip1q.nxv4f32(<vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.zip1q.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 8 x bfloat> @llvm.aarch64.sve.zip1q.nxv8bf16(<vscale x 8 x bfloat>, <vscale x 8 x bfloat>)
declare <vscale x 8 x half> @llvm.aarch64.sve.zip1q.nxv8f16(<vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.zip1q.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 16 x i8> @llvm.aarch64.sve.zip1q.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>)

declare <vscale x 2 x double> @llvm.aarch64.sve.zip2q.nxv2f64(<vscale x 2 x double>, <vscale x 2 x double>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.zip2q.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>)
declare <vscale x 4 x float> @llvm.aarch64.sve.zip2q.nxv4f32(<vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.zip2q.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 8 x bfloat> @llvm.aarch64.sve.zip2q.nxv8bf16(<vscale x 8 x bfloat>, <vscale x 8 x bfloat>)
declare <vscale x 8 x half> @llvm.aarch64.sve.zip2q.nxv8f16(<vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.zip2q.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 16 x i8> @llvm.aarch64.sve.zip2q.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>)

; +bf16 is required for the bfloat version.
attributes #0 = { "target-features"="+sve,+fp64mm,+bf16" }
