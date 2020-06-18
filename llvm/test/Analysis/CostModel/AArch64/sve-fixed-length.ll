; RUN: opt < %s -cost-model -analyze | FileCheck %s -D#VBITS=128
; RUN: opt < %s -cost-model -analyze -aarch64-sve-vector-bits-min=128 | FileCheck %s -D#VBITS=128
; RUN: opt < %s -cost-model -analyze -aarch64-sve-vector-bits-min=256 | FileCheck %s -D#VBITS=256
; RUN: opt < %s -cost-model -analyze -aarch64-sve-vector-bits-min=384 | FileCheck %s -D#VBITS=256
; RUN: opt < %s -cost-model -analyze -aarch64-sve-vector-bits-min=512 | FileCheck %s -D#VBITS=512
; RUN: opt < %s -cost-model -analyze -aarch64-sve-vector-bits-min=640 | FileCheck %s -D#VBITS=512
; RUN: opt < %s -cost-model -analyze -aarch64-sve-vector-bits-min=768 | FileCheck %s -D#VBITS=512
; RUN: opt < %s -cost-model -analyze -aarch64-sve-vector-bits-min=896 | FileCheck %s -D#VBITS=512
; RUN: opt < %s -cost-model -analyze -aarch64-sve-vector-bits-min=1024 | FileCheck %s -D#VBITS=1024
; RUN: opt < %s -cost-model -analyze -aarch64-sve-vector-bits-min=1152 | FileCheck %s -D#VBITS=1024
; RUN: opt < %s -cost-model -analyze -aarch64-sve-vector-bits-min=1280 | FileCheck %s -D#VBITS=1024
; RUN: opt < %s -cost-model -analyze -aarch64-sve-vector-bits-min=1408 | FileCheck %s -D#VBITS=1024
; RUN: opt < %s -cost-model -analyze -aarch64-sve-vector-bits-min=1536 | FileCheck %s -D#VBITS=1024
; RUN: opt < %s -cost-model -analyze -aarch64-sve-vector-bits-min=1664 | FileCheck %s -D#VBITS=1024
; RUN: opt < %s -cost-model -analyze -aarch64-sve-vector-bits-min=1792 | FileCheck %s -D#VBITS=1024
; RUN: opt < %s -cost-model -analyze -aarch64-sve-vector-bits-min=1920 | FileCheck %s -D#VBITS=1024
; RUN: opt < %s -cost-model -analyze -aarch64-sve-vector-bits-min=2048 | FileCheck %s -D#VBITS=2048

; VBITS represents the useful bit size of a vector register from the code
; generator's point of view. It is clamped to power-of-2 values because
; only power-of-2 vector lengths are considered legal, regardless of the
; user specified vector length.

target triple = "aarch64-unknown-linux-gnu"

; Ensure the cost of legalisation is removed as the vector length grows.
; NOTE: Assumes BaseCost_add=1, BaseCost_fadd=2.
define void @add() #0 {
; CHECK-LABEL: Printing analysis 'Cost Model Analysis' for function 'add':
; CHECK: cost of [[#div(127,VBITS)+1]] for instruction:   %add128 = add <4 x i32> undef, undef
; CHECK: cost of [[#div(255,VBITS)+1]] for instruction:   %add256 = add <8 x i32> undef, undef
; CHECK: cost of [[#div(511,VBITS)+1]] for instruction:   %add512 = add <16 x i32> undef, undef
; CHECK: cost of [[#div(1023,VBITS)+1]] for instruction:   %add1024 = add <32 x i32> undef, undef
; CHECK: cost of [[#div(2047,VBITS)+1]] for instruction:   %add2048 = add <64 x i32> undef, undef
  %add128 = add <4 x i32> undef, undef
  %add256 = add <8 x i32> undef, undef
  %add512 = add <16 x i32> undef, undef
  %add1024 = add <32 x i32> undef, undef
  %add2048 = add <64 x i32> undef, undef

; Using a single vector length, ensure all element types are recognised.
; CHECK: cost of [[#div(511,VBITS)+1]] for instruction:   %add512.i8 = add <64 x i8> undef, undef
; CHECK: cost of [[#div(511,VBITS)+1]] for instruction:   %add512.i16 = add <32 x i16> undef, undef
; CHECK: cost of [[#div(511,VBITS)+1]] for instruction:   %add512.i32 = add <16 x i32> undef, undef
; CHECK: cost of [[#div(511,VBITS)+1]] for instruction:   %add512.i64 = add <8 x i64> undef, undef
; CHECK: cost of [[#mul(div(511,VBITS)+1,2)]] for instruction:   %add512.f16 = fadd <32 x half> undef, undef
; CHECK: cost of [[#mul(div(511,VBITS)+1,2)]] for instruction:   %add512.f32 = fadd <16 x float> undef, undef
; CHECK: cost of [[#mul(div(511,VBITS)+1,2)]] for instruction:   %add512.f64 = fadd <8 x double> undef, undef
  %add512.i8 = add <64 x i8> undef, undef
  %add512.i16 = add <32 x i16> undef, undef
  %add512.i32 = add <16 x i32> undef, undef
  %add512.i64 = add <8 x i64> undef, undef
  %add512.f16 = fadd <32 x half> undef, undef
  %add512.f32 = fadd <16 x float> undef, undef
  %add512.f64 = fadd <8 x double> undef, undef

  ret void
}

attributes #0 = { "target-features"="+sve" }
