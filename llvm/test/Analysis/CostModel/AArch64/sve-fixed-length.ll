; RUN: opt < %s -passes="print<cost-model>" 2>&1 -disable-output | FileCheck %s -D#VBITS=128
; RUN: opt < %s -passes="print<cost-model>" 2>&1 -disable-output -aarch64-sve-vector-bits-min=128 | FileCheck %s -D#VBITS=128
; RUN: opt < %s -passes="print<cost-model>" 2>&1 -disable-output -aarch64-sve-vector-bits-min=256 | FileCheck %s -D#VBITS=256
; RUN: opt < %s -passes="print<cost-model>" 2>&1 -disable-output -aarch64-sve-vector-bits-min=384 | FileCheck %s -D#VBITS=256
; RUN: opt < %s -passes="print<cost-model>" 2>&1 -disable-output -aarch64-sve-vector-bits-min=512 | FileCheck %s -D#VBITS=512
; RUN: opt < %s -passes="print<cost-model>" 2>&1 -disable-output -aarch64-sve-vector-bits-min=640 | FileCheck %s -D#VBITS=512
; RUN: opt < %s -passes="print<cost-model>" 2>&1 -disable-output -aarch64-sve-vector-bits-min=768 | FileCheck %s -D#VBITS=512
; RUN: opt < %s -passes="print<cost-model>" 2>&1 -disable-output -aarch64-sve-vector-bits-min=896 | FileCheck %s -D#VBITS=512
; RUN: opt < %s -passes="print<cost-model>" 2>&1 -disable-output -aarch64-sve-vector-bits-min=1024 | FileCheck %s -D#VBITS=1024
; RUN: opt < %s -passes="print<cost-model>" 2>&1 -disable-output -aarch64-sve-vector-bits-min=1152 | FileCheck %s -D#VBITS=1024
; RUN: opt < %s -passes="print<cost-model>" 2>&1 -disable-output -aarch64-sve-vector-bits-min=1280 | FileCheck %s -D#VBITS=1024
; RUN: opt < %s -passes="print<cost-model>" 2>&1 -disable-output -aarch64-sve-vector-bits-min=1408 | FileCheck %s -D#VBITS=1024
; RUN: opt < %s -passes="print<cost-model>" 2>&1 -disable-output -aarch64-sve-vector-bits-min=1536 | FileCheck %s -D#VBITS=1024
; RUN: opt < %s -passes="print<cost-model>" 2>&1 -disable-output -aarch64-sve-vector-bits-min=1664 | FileCheck %s -D#VBITS=1024
; RUN: opt < %s -passes="print<cost-model>" 2>&1 -disable-output -aarch64-sve-vector-bits-min=1792 | FileCheck %s -D#VBITS=1024
; RUN: opt < %s -passes="print<cost-model>" 2>&1 -disable-output -aarch64-sve-vector-bits-min=1920 | FileCheck %s -D#VBITS=1024
; RUN: opt < %s -passes="print<cost-model>" 2>&1 -disable-output -aarch64-sve-vector-bits-min=2048 | FileCheck %s -D#VBITS=2048

; VBITS represents the useful bit size of a vector register from the code
; generator's point of view. It is clamped to power-of-2 values because
; only power-of-2 vector lengths are considered legal, regardless of the
; user specified vector length.

target triple = "aarch64-unknown-linux-gnu"
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"

; Ensure the cost of legalisation is removed as the vector length grows.
; NOTE: Assumes BaseCost_add=1, BaseCost_fadd=2.
define void @add() #0 {
; CHECK-LABEL: function 'add'
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

; Assuming base_cost = 2
; Assuming legalization_cost = (vec_len-1/VBITS)+1
; Assuming extra cost of 8 for i8.
; Assuming extra cost of 4 for i16.
; The hard-coded expected cost is based on VBITS=128
define void @sdiv() #0 {
; CHECK-LABEL: function 'sdiv'

; CHECK: cost of 5 for instruction:  %sdiv16.i8   = sdiv <2 x i8> undef, undef
  %sdiv16.i8   = sdiv <2 x i8> undef, undef

; CHECK: cost of 8 for instruction:  %sdiv32.i8   = sdiv <4 x i8> undef, undef
  %sdiv32.i8   = sdiv <4 x i8> undef, undef

; CHECK: cost of 5 for instruction:  %sdiv32.i16   = sdiv <2 x i16> undef, undef
  %sdiv32.i16  = sdiv <2 x i16> undef, undef

; CHECK: cost of 8 for instruction:  %sdiv64.i8   = sdiv <8 x i8> undef, undef
  %sdiv64.i8   = sdiv <8 x i8> undef, undef

; CHECK: cost of 5 for instruction:  %sdiv64.i16   = sdiv <4 x i16> undef, undef
  %sdiv64.i16  = sdiv <4 x i16> undef, undef

; CHECK: cost of 1 for instruction:  %sdiv64.i32   = sdiv <2 x i32> undef, undef
  %sdiv64.i32  = sdiv <2 x i32> undef, undef

; CHECK: cost of [[#mul(mul(div(128-1, VBITS)+1, 2), 8)]] for instruction:  %sdiv128.i8   = sdiv <16 x i8> undef, undef
  %sdiv128.i8 = sdiv <16 x i8> undef, undef

; CHECK: cost of [[#mul(mul(div(128-1, VBITS)+1, 2), 4)]] for instruction:  %sdiv128.i16   = sdiv <8 x i16> undef, undef
  %sdiv128.i16 = sdiv <8 x i16> undef, undef

; CHECK: cost of [[#mul(div(128-1, VBITS)+1, 2)]] for instruction:  %sdiv128.i64   = sdiv <2 x i64> undef, undef
  %sdiv128.i64 = sdiv <2 x i64> undef, undef

; CHECK: cost of [[#mul(mul(div(512-1, VBITS)+1, 2), 8)]] for instruction:  %sdiv512.i8   = sdiv <64 x i8> undef, undef
  %sdiv512.i8  = sdiv <64 x i8> undef, undef

; CHECK: cost of [[#mul(mul(div(512-1, VBITS)+1, 2), 4)]] for instruction:  %sdiv512.i16   = sdiv <32 x i16> undef, undef
  %sdiv512.i16 = sdiv <32 x i16> undef, undef

; CHECK: cost of [[#mul(div(512-1, VBITS)+1, 2)]] for instruction:  %sdiv512.i32   = sdiv <16 x i32> undef, undef
  %sdiv512.i32 = sdiv <16 x i32> undef, undef

; CHECK: cost of [[#mul(div(512-1, VBITS)+1, 2)]] for instruction:  %sdiv512.i64   = sdiv <8 x i64> undef, undef
  %sdiv512.i64 = sdiv <8 x i64> undef, undef

  ret void
}

; Assuming base_cost = 2
; Assuming legalization_cost = (vec_len-1/VBITS)+1
; Assuming extra cost of 8 for i8.
; Assuming extra cost of 4 for i16.
; The hard-coded expected cost is based on VBITS=128
define void @udiv() #0 {
; CHECK-LABEL: function 'udiv'

; CHECK: cost of 5 for instruction:  %udiv16.i8   = udiv <2 x i8> undef, undef
  %udiv16.i8   = udiv <2 x i8> undef, undef

; CHECK: cost of 8 for instruction:  %udiv32.i8   = udiv <4 x i8> undef, undef
  %udiv32.i8   = udiv <4 x i8> undef, undef

; CHECK: cost of 5 for instruction:  %udiv32.i16   = udiv <2 x i16> undef, undef
  %udiv32.i16  = udiv <2 x i16> undef, undef

; CHECK: cost of 8 for instruction:  %udiv64.i8   = udiv <8 x i8> undef, undef
  %udiv64.i8   = udiv <8 x i8> undef, undef

; CHECK: cost of 5 for instruction:  %udiv64.i16   = udiv <4 x i16> undef, undef
  %udiv64.i16  = udiv <4 x i16> undef, undef

; CHECK: cost of 1 for instruction:  %udiv64.i32   = udiv <2 x i32> undef, undef
  %udiv64.i32  = udiv <2 x i32> undef, undef

; CHECK: cost of [[#mul(mul(div(128-1, VBITS)+1, 2), 8)]] for instruction:  %udiv128.i8   = udiv <16 x i8> undef, undef
  %udiv128.i8 = udiv <16 x i8> undef, undef

; CHECK: cost of [[#mul(mul(div(128-1, VBITS)+1, 2), 4)]] for instruction:  %udiv128.i16   = udiv <8 x i16> undef, undef
  %udiv128.i16 = udiv <8 x i16> undef, undef

; CHECK: cost of [[#mul(div(128-1, VBITS)+1, 2)]] for instruction:  %udiv128.i64   = udiv <2 x i64> undef, undef
  %udiv128.i64 = udiv <2 x i64> undef, undef

; CHECK: cost of [[#mul(mul(div(512-1, VBITS)+1, 2), 8)]] for instruction:  %udiv512.i8   = udiv <64 x i8> undef, undef
  %udiv512.i8  = udiv <64 x i8> undef, undef

; CHECK: cost of [[#mul(mul(div(512-1, VBITS)+1, 2), 4)]] for instruction:  %udiv512.i16   = udiv <32 x i16> undef, undef
  %udiv512.i16 = udiv <32 x i16> undef, undef

; CHECK: cost of [[#mul(div(512-1, VBITS)+1, 2)]] for instruction:  %udiv512.i32   = udiv <16 x i32> undef, undef
  %udiv512.i32 = udiv <16 x i32> undef, undef

; CHECK: cost of [[#mul(div(512-1, VBITS)+1, 2)]] for instruction:  %udiv512.i64   = udiv <8 x i64> undef, undef
  %udiv512.i64 = udiv <8 x i64> undef, undef

  ret void
}

; The hard-coded expected cost is based on VBITS=128
define void @mul() #0 {
; CHECK: cost of [[#div(128-1, VBITS)+1]] for instruction:  %mul128.i64  = mul <2 x i64> undef, undef
  %mul128.i64 = mul <2 x i64> undef, undef

; CHECK: cost of [[#div(512-1, VBITS)+1]] for instruction:  %mul512.i64 = mul <8 x i64> undef, undef
  %mul512.i64 = mul <8 x i64> undef, undef

   ret void
 }

attributes #0 = { "target-features"="+sve" }
