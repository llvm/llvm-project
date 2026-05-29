; sitofp iN->bfloat is decomposed via float intermediate + bit shuffling.
; sitofp i8->float is widened through i32.
; RUN: llc -mtriple=air %s -o - | FileCheck %s

; CHECK-LABEL: define bfloat @to_bf16
; CHECK: sitofp i32 %x to float
; CHECK: bitcast float {{.*}} to i32
; CHECK: lshr i32 {{.*}}, 16
; CHECK: trunc i32 {{.*}} to i16
; CHECK: bitcast i16 {{.*}} to bfloat

; CHECK-LABEL: define float @to_f32_from_i8
; CHECK: sext i8 %x to i32
; CHECK: sitofp i32 {{.*}} to float

define bfloat @to_bf16(i32 %x) {
  %r = sitofp i32 %x to bfloat
  ret bfloat %r
}

define float @to_f32_from_i8(i8 %x) {
  %r = sitofp i8 %x to float
  ret float %r
}
