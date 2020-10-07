; RUN: llc < %s -mtriple=aarch64-linux--gnu -aarch64-neon-syntax=generic | FileCheck %s

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"

declare i8 @llvm.vector.reduce.smax.v16i8(<16 x i8>)
declare i16 @llvm.vector.reduce.smax.v8i16(<8 x i16>)
declare i32 @llvm.vector.reduce.smax.v4i32(<4 x i32>)
declare i8 @llvm.vector.reduce.umax.v16i8(<16 x i8>)
declare i16 @llvm.vector.reduce.umax.v8i16(<8 x i16>)
declare i32 @llvm.vector.reduce.umax.v4i32(<4 x i32>)

declare i8 @llvm.vector.reduce.smin.v16i8(<16 x i8>)
declare i16 @llvm.vector.reduce.smin.v8i16(<8 x i16>)
declare i32 @llvm.vector.reduce.smin.v4i32(<4 x i32>)
declare i8 @llvm.vector.reduce.umin.v16i8(<16 x i8>)
declare i16 @llvm.vector.reduce.umin.v8i16(<8 x i16>)
declare i32 @llvm.vector.reduce.umin.v4i32(<4 x i32>)

declare float @llvm.vector.reduce.fmax.v4f32(<4 x float>)
declare float @llvm.vector.reduce.fmin.v4f32(<4 x float>)

; CHECK-LABEL: smax_B
; CHECK: smaxv {{b[0-9]+}}, {{v[0-9]+}}.16b
define i8 @smax_B(<16 x i8>* nocapture readonly %arr)  {
  %arr.load = load <16 x i8>, <16 x i8>* %arr
  %r = call i8 @llvm.vector.reduce.smax.v16i8(<16 x i8> %arr.load)
  ret i8 %r
}

; CHECK-LABEL: smax_H
; CHECK: smaxv {{h[0-9]+}}, {{v[0-9]+}}.8h
define i16 @smax_H(<8 x i16>* nocapture readonly %arr) {
  %arr.load = load <8 x i16>, <8 x i16>* %arr
  %r = call i16 @llvm.vector.reduce.smax.v8i16(<8 x i16> %arr.load)
  ret i16 %r
}

; CHECK-LABEL: smax_S
; CHECK: smaxv {{s[0-9]+}}, {{v[0-9]+}}.4s
define i32 @smax_S(<4 x i32> * nocapture readonly %arr)  {
  %arr.load = load <4 x i32>, <4 x i32>* %arr
  %r = call i32 @llvm.vector.reduce.smax.v4i32(<4 x i32> %arr.load)
  ret i32 %r
}

; CHECK-LABEL: umax_B
; CHECK: umaxv {{b[0-9]+}}, {{v[0-9]+}}.16b
define i8 @umax_B(<16 x i8>* nocapture readonly %arr)  {
  %arr.load = load <16 x i8>, <16 x i8>* %arr
  %r = call i8 @llvm.vector.reduce.umax.v16i8(<16 x i8> %arr.load)
  ret i8 %r
}

; CHECK-LABEL: umax_H
; CHECK: umaxv {{h[0-9]+}}, {{v[0-9]+}}.8h
define i16 @umax_H(<8 x i16>* nocapture readonly %arr)  {
  %arr.load = load <8 x i16>, <8 x i16>* %arr
  %r = call i16 @llvm.vector.reduce.umax.v8i16(<8 x i16> %arr.load)
  ret i16 %r
}

; CHECK-LABEL: umax_S
; CHECK: umaxv {{s[0-9]+}}, {{v[0-9]+}}.4s
define i32 @umax_S(<4 x i32>* nocapture readonly %arr) {
  %arr.load = load <4 x i32>, <4 x i32>* %arr
  %r = call i32 @llvm.vector.reduce.umax.v4i32(<4 x i32> %arr.load)
  ret i32 %r
}

; CHECK-LABEL: smin_B
; CHECK: sminv {{b[0-9]+}}, {{v[0-9]+}}.16b
define i8 @smin_B(<16 x i8>* nocapture readonly %arr) {
  %arr.load = load <16 x i8>, <16 x i8>* %arr
  %r = call i8 @llvm.vector.reduce.smin.v16i8(<16 x i8> %arr.load)
  ret i8 %r
}

; CHECK-LABEL: smin_H
; CHECK: sminv {{h[0-9]+}}, {{v[0-9]+}}.8h
define i16 @smin_H(<8 x i16>* nocapture readonly %arr) {
  %arr.load = load <8 x i16>, <8 x i16>* %arr
  %r = call i16 @llvm.vector.reduce.smin.v8i16(<8 x i16> %arr.load)
  ret i16 %r
}

; CHECK-LABEL: smin_S
; CHECK: sminv {{s[0-9]+}}, {{v[0-9]+}}.4s
define i32 @smin_S(<4 x i32>* nocapture readonly %arr) {
  %arr.load = load <4 x i32>, <4 x i32>* %arr
  %r = call i32 @llvm.vector.reduce.smin.v4i32(<4 x i32> %arr.load)
  ret i32 %r
}

; CHECK-LABEL: umin_B
; CHECK: uminv {{b[0-9]+}}, {{v[0-9]+}}.16b
define i8 @umin_B(<16 x i8>* nocapture readonly %arr)  {
  %arr.load = load <16 x i8>, <16 x i8>* %arr
  %r = call i8 @llvm.vector.reduce.umin.v16i8(<16 x i8> %arr.load)
  ret i8 %r
}

; CHECK-LABEL: umin_H
; CHECK: uminv {{h[0-9]+}}, {{v[0-9]+}}.8h
define i16 @umin_H(<8 x i16>* nocapture readonly %arr)  {
  %arr.load = load <8 x i16>, <8 x i16>* %arr
  %r = call i16 @llvm.vector.reduce.umin.v8i16(<8 x i16> %arr.load)
  ret i16 %r
}

; CHECK-LABEL: umin_S
; CHECK: uminv {{s[0-9]+}}, {{v[0-9]+}}.4s
define i32 @umin_S(<4 x i32>* nocapture readonly %arr) {
  %arr.load = load <4 x i32>, <4 x i32>* %arr
  %r = call i32 @llvm.vector.reduce.umin.v4i32(<4 x i32> %arr.load)
  ret i32 %r
}

; CHECK-LABEL: fmaxnm_S
; CHECK: fmaxnmv
define float @fmaxnm_S(<4 x float>* nocapture readonly %arr) {
  %arr.load  = load <4 x float>, <4 x float>* %arr
  %r = call nnan float @llvm.vector.reduce.fmax.v4f32(<4 x float> %arr.load)
  ret float %r
}

; CHECK-LABEL: fminnm_S
; CHECK: fminnmv
define float @fminnm_S(<4 x float>* nocapture readonly %arr) {
  %arr.load  = load <4 x float>, <4 x float>* %arr
  %r = call nnan float @llvm.vector.reduce.fmin.v4f32(<4 x float> %arr.load)
  ret float %r
}

declare i16 @llvm.vector.reduce.umax.v16i16(<16 x i16>)

define i16 @oversized_umax_256(<16 x i16>* nocapture readonly %arr)  {
; CHECK-LABEL: oversized_umax_256
; CHECK: umax [[V0:v[0-9]+]].8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
; CHECK: umaxv {{h[0-9]+}}, [[V0]]
  %arr.load = load <16 x i16>, <16 x i16>* %arr
  %r = call i16 @llvm.vector.reduce.umax.v16i16(<16 x i16> %arr.load)
  ret i16 %r
}

declare i32 @llvm.vector.reduce.umax.v16i32(<16 x i32>)

define i32 @oversized_umax_512(<16 x i32>* nocapture readonly %arr)  {
; CHECK-LABEL: oversized_umax_512
; CHECK: umax v
; CHECK-NEXT: umax v
; CHECK-NEXT: umax [[V0:v[0-9]+]].4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
; CHECK-NEXT: umaxv {{s[0-9]+}}, [[V0]]
  %arr.load = load <16 x i32>, <16 x i32>* %arr
  %r = call i32 @llvm.vector.reduce.umax.v16i32(<16 x i32> %arr.load)
  ret i32 %r
}

declare i16 @llvm.vector.reduce.umin.v16i16(<16 x i16>)

define i16 @oversized_umin_256(<16 x i16>* nocapture readonly %arr)  {
; CHECK-LABEL: oversized_umin_256
; CHECK: umin [[V0:v[0-9]+]].8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
; CHECK: uminv {{h[0-9]+}}, [[V0]]
  %arr.load = load <16 x i16>, <16 x i16>* %arr
  %r = call i16 @llvm.vector.reduce.umin.v16i16(<16 x i16> %arr.load)
  ret i16 %r
}

declare i32 @llvm.vector.reduce.umin.v16i32(<16 x i32>)

define i32 @oversized_umin_512(<16 x i32>* nocapture readonly %arr)  {
; CHECK-LABEL: oversized_umin_512
; CHECK: umin v
; CHECK-NEXT: umin v
; CHECK-NEXT: umin [[V0:v[0-9]+]].4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
; CHECK-NEXT: uminv {{s[0-9]+}}, [[V0]]
  %arr.load = load <16 x i32>, <16 x i32>* %arr
  %r = call i32 @llvm.vector.reduce.umin.v16i32(<16 x i32> %arr.load)
  ret i32 %r
}

declare i16 @llvm.vector.reduce.smax.v16i16(<16 x i16>)

define i16 @oversized_smax_256(<16 x i16>* nocapture readonly %arr)  {
; CHECK-LABEL: oversized_smax_256
; CHECK: smax [[V0:v[0-9]+]].8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
; CHECK: smaxv {{h[0-9]+}}, [[V0]]
  %arr.load = load <16 x i16>, <16 x i16>* %arr
  %r = call i16 @llvm.vector.reduce.smax.v16i16(<16 x i16> %arr.load)
  ret i16 %r
}

declare i32 @llvm.vector.reduce.smax.v16i32(<16 x i32>)

define i32 @oversized_smax_512(<16 x i32>* nocapture readonly %arr)  {
; CHECK-LABEL: oversized_smax_512
; CHECK: smax v
; CHECK-NEXT: smax v
; CHECK-NEXT: smax [[V0:v[0-9]+]].4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
; CHECK-NEXT: smaxv {{s[0-9]+}}, [[V0]]
  %arr.load = load <16 x i32>, <16 x i32>* %arr
  %r = call i32 @llvm.vector.reduce.smax.v16i32(<16 x i32> %arr.load)
  ret i32 %r
}

declare i16 @llvm.vector.reduce.smin.v16i16(<16 x i16>)

define i16 @oversized_smin_256(<16 x i16>* nocapture readonly %arr)  {
; CHECK-LABEL: oversized_smin_256
; CHECK: smin [[V0:v[0-9]+]].8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
; CHECK: sminv {{h[0-9]+}}, [[V0]]
  %arr.load = load <16 x i16>, <16 x i16>* %arr
  %r = call i16 @llvm.vector.reduce.smin.v16i16(<16 x i16> %arr.load)
  ret i16 %r
}

declare i32 @llvm.vector.reduce.smin.v16i32(<16 x i32>)

define i32 @oversized_smin_512(<16 x i32>* nocapture readonly %arr)  {
; CHECK-LABEL: oversized_smin_512
; CHECK: smin v
; CHECK-NEXT: smin v
; CHECK-NEXT: smin [[V0:v[0-9]+]].4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
; CHECK-NEXT: sminv {{s[0-9]+}}, [[V0]]
  %arr.load = load <16 x i32>, <16 x i32>* %arr
  %r = call i32 @llvm.vector.reduce.smin.v16i32(<16 x i32> %arr.load)
  ret i32 %r
}
