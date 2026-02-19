; RUN: rm -rf %t && split-file %s %t
; RUN: llvm-as < %t/missing-arg.ll | llvm-dis | FileCheck -check-prefix=MISSING-ARG %s
; RUN: llvm-as < %t/void-return.ll | llvm-dis | FileCheck -check-prefix=VOID %s
; RUN: llvm-as < %t/bfloat.ll | llvm-dis | FileCheck -check-prefix=BFLOAT %s
; RUN: llvm-as < %t/half.ll | llvm-dis | FileCheck -check-prefix=HALF %s
; RUN: llvm-as < %t/vector.ll | llvm-dis | FileCheck -check-prefix=VECTOR %s

;--- missing-arg.ll

define i16 @convert_to_fp16__missing_arg() {
  ; MISSING-ARG: %result = call i16 @llvm.convert.to.fp16.f32()
  %result = call i16 @llvm.convert.to.fp16.f32()
  ret i16 %result
}

define float @convert_from_fp16__f32_missing_arg() {
; MISSING-ARG: %result = call float @llvm.convert.from.fp16.f32()
  %result = call float @llvm.convert.from.fp16.f32()
  ret float %result
}

declare i16 @llvm.convert.to.fp16.f32()
declare float @llvm.convert.from.fp16.f32()


;--- void-return.ll

define void @convert_to_fp16__f32(float %src) {
; VOID: call void @llvm.convert.to.fp16.f32(float %src)
  call void @llvm.convert.to.fp16.f32(float %src)
  ret void
}

define void @convert_from_fp16__f32(i16 %src) {
; VOID: call void @llvm.convert.from.fp16.f32(i16 %src)
  call void @llvm.convert.from.fp16.f32(i16 %src)
  ret void
}

declare void @llvm.convert.to.fp16.f32(float)
declare void @llvm.convert.from.fp16.f32(i16)

;--- bfloat.ll

; Not well formed but the verifier never enforced this.
define i16 @convert_to_fp16__bf16(bfloat %src) {
; BFLOAT: %result = call i16 @llvm.convert.to.fp16.bf16(bfloat %src)
  %result = call i16 @llvm.convert.to.fp16.bf16(bfloat %src)
  ret i16 %result
}

; Not well formed but the verifier never enforced this.
define bfloat @convert_from_fp16__bf16(i16 %src) {
; BFLOAT: %result = call bfloat @llvm.convert.from.fp16.bf16(i16 %src)
  %result = call bfloat @llvm.convert.from.fp16.bf16(i16 %src)
  ret bfloat %result
}

declare i16 @llvm.convert.to.fp16.bf16(bfloat)
declare bfloat @llvm.convert.from.fp16.bf16(i16)

;--- half.ll

define i16 @convert_to_fp16__f16(half %src) {
; HALF: %result = call i16 @llvm.convert.to.fp16.f16(half %src)
  %result = call i16 @llvm.convert.to.fp16.f16(half %src)
  ret i16 %result
}

define half @convert_from_fp16__f16(i16 %src) {
; HALF: %result = call half @llvm.convert.from.fp16.f16(i16 %src)
  %result = call half @llvm.convert.from.fp16.f16(i16 %src)
  ret half %result
}

declare i16 @llvm.convert.to.fp16.f16(half)
declare half @llvm.convert.from.fp16.f16(i16)

;--- vector.ll

; These were not declared as supporting vectors.
define <2 x i16> @convert_to_fp16__v2f32(<2 x float> %src) {
; VECTOR: %result = call <2 x i16> @llvm.convert.to.fp16.v2f32(<2 x float> %src)
  %result = call <2 x i16> @llvm.convert.to.fp16.v2f32(<2 x float> %src)
  ret <2 x i16> %result
}

define <2 x float> @convert_from_fp16__v2f32(<2 x i16> %src) {
; VECTOR: %result = call <2 x float> @llvm.convert.from.fp16.v2f32(<2 x i16> %src)
  %result = call <2 x float> @llvm.convert.from.fp16.v2f32(<2 x i16> %src)
  ret <2 x float> %result
}

declare <2 x i16> @llvm.convert.to.fp16.v2f32(<2 x float>)
declare <2 x float> @llvm.convert.from.fp16.v2f32(<2 x i16>)
