; RUN: opt < %s -mtriple=aarch64--linux-gnu -cost-model -analyze | FileCheck %s --check-prefix=COST
; RUN: llc < %s -mtriple=aarch64--linux-gnu | FileCheck %s --check-prefix=CODE

; COST-LABEL: add.i8.v8i8
; COST:       Found an estimated cost of 1 for instruction: %r = call i8 @llvm.experimental.vector.reduce.add.i8.v8i8(<8 x i8> %v)
; CODE-LABEL: add.i8.v8i8
; CODE:       addv b0, v0.8b
define i8 @add.i8.v8i8(<8 x i8> %v) {
  %r = call i8 @llvm.experimental.vector.reduce.add.i8.v8i8(<8 x i8> %v)
  ret i8 %r
}

; COST-LABEL: add.i8.v16i8
; COST:       Found an estimated cost of 1 for instruction: %r = call i8 @llvm.experimental.vector.reduce.add.i8.v16i8(<16 x i8> %v)
; CODE-LABEL: add.i8.v16i8
; CODE:       addv b0, v0.16b
define i8 @add.i8.v16i8(<16 x i8> %v) {
  %r = call i8 @llvm.experimental.vector.reduce.add.i8.v16i8(<16 x i8> %v)
  ret i8 %r
}

; COST-LABEL: add.i16.v4i16
; COST:       Found an estimated cost of 1 for instruction: %r = call i16 @llvm.experimental.vector.reduce.add.i16.v4i16(<4 x i16> %v)
; CODE-LABEL: add.i16.v4i16
; CODE:       addv h0, v0.4h
define i16 @add.i16.v4i16(<4 x i16> %v) {
  %r = call i16 @llvm.experimental.vector.reduce.add.i16.v4i16(<4 x i16> %v)
  ret i16 %r
}

; COST-LABEL: add.i16.v8i16
; COST:       Found an estimated cost of 1 for instruction: %r = call i16 @llvm.experimental.vector.reduce.add.i16.v8i16(<8 x i16> %v)
; CODE-LABEL: add.i16.v8i16
; CODE:       addv h0, v0.8h
define i16 @add.i16.v8i16(<8 x i16> %v) {
  %r = call i16 @llvm.experimental.vector.reduce.add.i16.v8i16(<8 x i16> %v)
  ret i16 %r
}

; COST-LABEL: add.i32.v4i32
; COST:       Found an estimated cost of 1 for instruction: %r = call i32 @llvm.experimental.vector.reduce.add.i32.v4i32(<4 x i32> %v)
; CODE-LABEL: add.i32.v4i32
; CODE:       addv s0, v0.4s
define i32 @add.i32.v4i32(<4 x i32> %v) {
  %r = call i32 @llvm.experimental.vector.reduce.add.i32.v4i32(<4 x i32> %v)
  ret i32 %r
}

; COST-LABEL: umin.i8.v8i8
; COST:       Found an estimated cost of 157 for instruction: %r = call i8 @llvm.experimental.vector.reduce.umin.i8.v8i8(<8 x i8> %v)
; CODE-LABEL: umin.i8.v8i8
; CODE:       uminv b0, v0.8b
define i8 @umin.i8.v8i8(<8 x i8> %v) {
  %r = call i8 @llvm.experimental.vector.reduce.umin.i8.v8i8(<8 x i8> %v)
  ret i8 %r
}

; COST-LABEL: umin.i8.v16i8
; COST:       Found an estimated cost of 388 for instruction: %r = call i8 @llvm.experimental.vector.reduce.umin.i8.v16i8(<16 x i8> %v)
; CODE-LABEL: umin.i8.v16i8
; CODE:       uminv b0, v0.16b
define i8 @umin.i8.v16i8(<16 x i8> %v) {
  %r = call i8 @llvm.experimental.vector.reduce.umin.i8.v16i8(<16 x i8> %v)
  ret i8 %r
}

; COST-LABEL: umin.i16.v4i16
; COST:       Found an estimated cost of 58 for instruction: %r = call i16 @llvm.experimental.vector.reduce.umin.i16.v4i16(<4 x i16> %v)
; CODE-LABEL: umin.i16.v4i16
; CODE:       uminv h0, v0.4h
define i16 @umin.i16.v4i16(<4 x i16> %v) {
  %r = call i16 @llvm.experimental.vector.reduce.umin.i16.v4i16(<4 x i16> %v)
  ret i16 %r
}

; COST-LABEL: umin.i16.v8i16
; COST:       Found an estimated cost of 157 for instruction: %r = call i16 @llvm.experimental.vector.reduce.umin.i16.v8i16(<8 x i16> %v)
; CODE-LABEL: umin.i16.v8i16
; CODE:       uminv h0, v0.8h
define i16 @umin.i16.v8i16(<8 x i16> %v) {
  %r = call i16 @llvm.experimental.vector.reduce.umin.i16.v8i16(<8 x i16> %v)
  ret i16 %r
}

; COST-LABEL: umin.i32.v4i32
; COST:       Found an estimated cost of 58 for instruction: %r = call i32 @llvm.experimental.vector.reduce.umin.i32.v4i32(<4 x i32> %v)
; CODE-LABEL: umin.i32.v4i32
; CODE:       uminv s0, v0.4s
define i32 @umin.i32.v4i32(<4 x i32> %v) {
  %r = call i32 @llvm.experimental.vector.reduce.umin.i32.v4i32(<4 x i32> %v)
  ret i32 %r
}

; COST-LABEL: umax.i8.v8i8
; COST:       Found an estimated cost of 157 for instruction: %r = call i8 @llvm.experimental.vector.reduce.umax.i8.v8i8(<8 x i8> %v)
; CODE-LABEL: umax.i8.v8i8
; CODE:       umaxv b0, v0.8b
define i8 @umax.i8.v8i8(<8 x i8> %v) {
  %r = call i8 @llvm.experimental.vector.reduce.umax.i8.v8i8(<8 x i8> %v)
  ret i8 %r
}

; COST-LABEL: umax.i8.v16i8
; COST:       Found an estimated cost of 388 for instruction: %r = call i8 @llvm.experimental.vector.reduce.umax.i8.v16i8(<16 x i8> %v)
; CODE-LABEL: umax.i8.v16i8
; CODE:       umaxv b0, v0.16b
define i8 @umax.i8.v16i8(<16 x i8> %v) {
  %r = call i8 @llvm.experimental.vector.reduce.umax.i8.v16i8(<16 x i8> %v)
  ret i8 %r
}

; COST-LABEL: umax.i16.v4i16
; COST:       Found an estimated cost of 58 for instruction: %r = call i16 @llvm.experimental.vector.reduce.umax.i16.v4i16(<4 x i16> %v)
; CODE-LABEL: umax.i16.v4i16
; CODE:       umaxv h0, v0.4h
define i16 @umax.i16.v4i16(<4 x i16> %v) {
  %r = call i16 @llvm.experimental.vector.reduce.umax.i16.v4i16(<4 x i16> %v)
  ret i16 %r
}

; COST-LABEL: umax.i16.v8i16
; COST:       Found an estimated cost of 157 for instruction: %r = call i16 @llvm.experimental.vector.reduce.umax.i16.v8i16(<8 x i16> %v)
; CODE-LABEL: umax.i16.v8i16
; CODE:       umaxv h0, v0.8h
define i16 @umax.i16.v8i16(<8 x i16> %v) {
  %r = call i16 @llvm.experimental.vector.reduce.umax.i16.v8i16(<8 x i16> %v)
  ret i16 %r
}

; COST-LABEL: umax.i32.v4i32
; COST:       Found an estimated cost of 58 for instruction: %r = call i32 @llvm.experimental.vector.reduce.umax.i32.v4i32(<4 x i32> %v)
; CODE-LABEL: umax.i32.v4i32
; CODE:       umaxv s0, v0.4s
define i32 @umax.i32.v4i32(<4 x i32> %v) {
  %r = call i32 @llvm.experimental.vector.reduce.umax.i32.v4i32(<4 x i32> %v)
  ret i32 %r
}

; COST-LABEL: smin.i8.v8i8
; COST:       Found an estimated cost of 157 for instruction: %r = call i8 @llvm.experimental.vector.reduce.smin.i8.v8i8(<8 x i8> %v)
; CODE-LABEL: smin.i8.v8i8
; CODE:       sminv b0, v0.8b
define i8 @smin.i8.v8i8(<8 x i8> %v) {
  %r = call i8 @llvm.experimental.vector.reduce.smin.i8.v8i8(<8 x i8> %v)
  ret i8 %r
}

; COST-LABEL: smin.i8.v16i8
; COST:       Found an estimated cost of 388 for instruction: %r = call i8 @llvm.experimental.vector.reduce.smin.i8.v16i8(<16 x i8> %v)
; CODE-LABEL: smin.i8.v16i8
; CODE:       sminv b0, v0.16b
define i8 @smin.i8.v16i8(<16 x i8> %v) {
  %r = call i8 @llvm.experimental.vector.reduce.smin.i8.v16i8(<16 x i8> %v)
  ret i8 %r
}

; COST-LABEL: smin.i16.v4i16
; COST:       Found an estimated cost of 58 for instruction: %r = call i16 @llvm.experimental.vector.reduce.smin.i16.v4i16(<4 x i16> %v)
; CODE-LABEL: smin.i16.v4i16
; CODE:       sminv h0, v0.4h
define i16 @smin.i16.v4i16(<4 x i16> %v) {
  %r = call i16 @llvm.experimental.vector.reduce.smin.i16.v4i16(<4 x i16> %v)
  ret i16 %r
}

; COST-LABEL: smin.i16.v8i16
; COST:       Found an estimated cost of 157 for instruction: %r = call i16 @llvm.experimental.vector.reduce.smin.i16.v8i16(<8 x i16> %v)
; CODE-LABEL: smin.i16.v8i16
; CODE:       sminv h0, v0.8h
define i16 @smin.i16.v8i16(<8 x i16> %v) {
  %r = call i16 @llvm.experimental.vector.reduce.smin.i16.v8i16(<8 x i16> %v)
  ret i16 %r
}

; COST-LABEL: smin.i32.v4i32
; COST:       Found an estimated cost of 58 for instruction: %r = call i32 @llvm.experimental.vector.reduce.smin.i32.v4i32(<4 x i32> %v)
; CODE-LABEL: smin.i32.v4i32
; CODE:       sminv s0, v0.4s
define i32 @smin.i32.v4i32(<4 x i32> %v) {
  %r = call i32 @llvm.experimental.vector.reduce.smin.i32.v4i32(<4 x i32> %v)
  ret i32 %r
}

; COST-LABEL: smax.i8.v8i8
; COST:       Found an estimated cost of 157 for instruction: %r = call i8 @llvm.experimental.vector.reduce.smax.i8.v8i8(<8 x i8> %v)
; CODE-LABEL: smax.i8.v8i8
; CODE:       smaxv b0, v0.8b
define i8 @smax.i8.v8i8(<8 x i8> %v) {
  %r = call i8 @llvm.experimental.vector.reduce.smax.i8.v8i8(<8 x i8> %v)
  ret i8 %r
}

; COST-LABEL: smax.i8.v16i8
; COST:       Found an estimated cost of 388 for instruction: %r = call i8 @llvm.experimental.vector.reduce.smax.i8.v16i8(<16 x i8> %v)
; CODE-LABEL: smax.i8.v16i8
; CODE:       smaxv b0, v0.16b
define i8 @smax.i8.v16i8(<16 x i8> %v) {
  %r = call i8 @llvm.experimental.vector.reduce.smax.i8.v16i8(<16 x i8> %v)
  ret i8 %r
}

; COST-LABEL: smax.i16.v4i16
; COST:       Found an estimated cost of 58 for instruction: %r = call i16 @llvm.experimental.vector.reduce.smax.i16.v4i16(<4 x i16> %v)
; CODE-LABEL: smax.i16.v4i16
; CODE:       smaxv h0, v0.4h
define i16 @smax.i16.v4i16(<4 x i16> %v) {
  %r = call i16 @llvm.experimental.vector.reduce.smax.i16.v4i16(<4 x i16> %v)
  ret i16 %r
}

; COST-LABEL: smax.i16.v8i16
; COST:       Found an estimated cost of 157 for instruction: %r = call i16 @llvm.experimental.vector.reduce.smax.i16.v8i16(<8 x i16> %v)
; CODE-LABEL: smax.i16.v8i16
; CODE:       smaxv h0, v0.8h
define i16 @smax.i16.v8i16(<8 x i16> %v) {
  %r = call i16 @llvm.experimental.vector.reduce.smax.i16.v8i16(<8 x i16> %v)
  ret i16 %r
}

; COST-LABEL: smax.i32.v4i32
; COST:       Found an estimated cost of 58 for instruction: %r = call i32 @llvm.experimental.vector.reduce.smax.i32.v4i32(<4 x i32> %v)
; CODE-LABEL: smax.i32.v4i32
; CODE:       smaxv s0, v0.4s
define i32 @smax.i32.v4i32(<4 x i32> %v) {
  %r = call i32 @llvm.experimental.vector.reduce.smax.i32.v4i32(<4 x i32> %v)
  ret i32 %r
}

; COST-LABEL: fmin.f32.v4f32
; COST:       Found an estimated cost of 58 for instruction: %r = call nnan float @llvm.experimental.vector.reduce.fmin.f32.v4f32(<4 x float> %v)
; CODE-LABEL: fmin.f32.v4f32
; CODE:       fminnmv s0, v0.4s
define float @fmin.f32.v4f32(<4 x float> %v) {
  %r = call nnan float @llvm.experimental.vector.reduce.fmin.f32.v4f32(<4 x float> %v)
  ret float %r
}

; COST-LABEL: fmax.f32.v4f32
; COST:       Found an estimated cost of 58 for instruction: %r = call nnan float @llvm.experimental.vector.reduce.fmax.f32.v4f32(<4 x float> %v)
; CODE-LABEL: fmax.f32.v4f32
; CODE:       fmaxnmv s0, v0.4s
define float @fmax.f32.v4f32(<4 x float> %v) {
  %r = call nnan float @llvm.experimental.vector.reduce.fmax.f32.v4f32(<4 x float> %v)
  ret float %r
}

declare i8 @llvm.experimental.vector.reduce.add.i8.v8i8(<8 x i8>)
declare i8 @llvm.experimental.vector.reduce.add.i8.v16i8(<16 x i8>)
declare i16 @llvm.experimental.vector.reduce.add.i16.v4i16(<4 x i16>)
declare i16 @llvm.experimental.vector.reduce.add.i16.v8i16(<8 x i16>)
declare i32 @llvm.experimental.vector.reduce.add.i32.v4i32(<4 x i32>)

declare i8 @llvm.experimental.vector.reduce.umin.i8.v8i8(<8 x i8>)
declare i8 @llvm.experimental.vector.reduce.umin.i8.v16i8(<16 x i8>)
declare i16 @llvm.experimental.vector.reduce.umin.i16.v4i16(<4 x i16>)
declare i16 @llvm.experimental.vector.reduce.umin.i16.v8i16(<8 x i16>)
declare i32 @llvm.experimental.vector.reduce.umin.i32.v4i32(<4 x i32>)

declare i8 @llvm.experimental.vector.reduce.umax.i8.v8i8(<8 x i8>)
declare i8 @llvm.experimental.vector.reduce.umax.i8.v16i8(<16 x i8>)
declare i16 @llvm.experimental.vector.reduce.umax.i16.v4i16(<4 x i16>)
declare i16 @llvm.experimental.vector.reduce.umax.i16.v8i16(<8 x i16>)
declare i32 @llvm.experimental.vector.reduce.umax.i32.v4i32(<4 x i32>)

declare i8 @llvm.experimental.vector.reduce.smin.i8.v8i8(<8 x i8>)
declare i8 @llvm.experimental.vector.reduce.smin.i8.v16i8(<16 x i8>)
declare i16 @llvm.experimental.vector.reduce.smin.i16.v4i16(<4 x i16>)
declare i16 @llvm.experimental.vector.reduce.smin.i16.v8i16(<8 x i16>)
declare i32 @llvm.experimental.vector.reduce.smin.i32.v4i32(<4 x i32>)

declare i8 @llvm.experimental.vector.reduce.smax.i8.v8i8(<8 x i8>)
declare i8 @llvm.experimental.vector.reduce.smax.i8.v16i8(<16 x i8>)
declare i16 @llvm.experimental.vector.reduce.smax.i16.v4i16(<4 x i16>)
declare i16 @llvm.experimental.vector.reduce.smax.i16.v8i16(<8 x i16>)
declare i32 @llvm.experimental.vector.reduce.smax.i32.v4i32(<4 x i32>)

declare float @llvm.experimental.vector.reduce.fmin.f32.v4f32(<4 x float>)

declare float @llvm.experimental.vector.reduce.fmax.f32.v4f32(<4 x float>)
