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

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

define void @fixed_sve_vls() #0 {
; CHECK-LABEL: 'fixed_sve_vls'
; CHECK:  Cost Model: Found an estimated cost of [[#div(2047,VBITS)+1]] for instruction: %v256i8 = call <256 x i8> @llvm.masked.load.v256i8.p0(ptr align 8 undef, <256 x i1> undef, <256 x i8> undef)
; CHECK:  Cost Model: Found an estimated cost of [[#div(4091,VBITS)+1]] for instruction: %v256i16 = call <256 x i16> @llvm.masked.load.v256i16.p0(ptr align 8 undef, <256 x i1> undef, <256 x i16> undef)
; CHECK:  Cost Model: Found an estimated cost of [[#div(511,VBITS)+1]] for instruction: %v16i32 = call <16 x i32> @llvm.masked.load.v16i32.p0(ptr align 8 undef, <16 x i1> undef, <16 x i32> undef)
; CHECK:  Cost Model: Found an estimated cost of [[#div(1023,VBITS)+1]] for instruction: %v16i64 = call <16 x i64> @llvm.masked.load.v16i64.p0(ptr align 8 undef, <16 x i1> undef, <16 x i64> undef)
; CHECK:  Cost Model: Found an estimated cost of [[#div(8191,VBITS)+1]] for instruction: %v512f16 = call <512 x half> @llvm.masked.load.v512f16.p0(ptr align 8 undef, <512 x i1> undef, <512 x half> undef)
; CHECK:  Cost Model: Found an estimated cost of [[#div(8191,VBITS)+1]] for instruction: %v256f32 = call <256 x float> @llvm.masked.load.v256f32.p0(ptr align 8 undef, <256 x i1> undef, <256 x float> undef)
; CHECK:  Cost Model: Found an estimated cost of [[#div(8191,VBITS)+1]] for instruction: %v128f64 = call <128 x double> @llvm.masked.load.v128f64.p0(ptr align 8 undef, <128 x i1> undef, <128 x double> undef)
; CHECK:  Cost Model: Found an estimated cost of 0 for instruction: ret void
entry:
  %v256i8 = call <256 x i8> @llvm.masked.load.v256i8.p0(ptr undef, i32 8, <256 x i1> undef, <256 x i8> undef)
  %v256i16 = call <256 x i16> @llvm.masked.load.v256i16.p0(ptr undef, i32 8, <256 x i1> undef, <256 x i16> undef)
  %v16i32 = call <16 x i32> @llvm.masked.load.v16i32.p0(ptr undef, i32 8, <16 x i1> undef, <16 x i32> undef)
  %v16i64 = call <16 x i64> @llvm.masked.load.v16i64.p0(ptr undef, i32 8, <16 x i1> undef, <16 x i64> undef)

  %v512f16 = call <512 x half> @llvm.masked.load.v512f16.p0(ptr undef, i32 8, <512 x i1> undef, <512 x half> undef)
  %v256f32 = call <256 x float> @llvm.masked.load.v256f32.p0(ptr undef, i32 8, <256 x i1> undef, <256 x float> undef)
  %v128f64 = call <128 x double> @llvm.masked.load.v128f64.p0(ptr undef, i32 8, <128 x i1> undef, <128 x double> undef)

  ret void
}

declare <256 x i8> @llvm.masked.load.v256i8.p0(ptr, i32, <256 x i1>, <256 x i8>)
declare <256 x i16> @llvm.masked.load.v256i16.p0(ptr, i32, <256 x i1>, <256 x i16>)
declare <16 x i32> @llvm.masked.load.v16i32.p0(ptr, i32, <16 x i1>, <16 x i32>)
declare <16 x i64> @llvm.masked.load.v16i64.p0(ptr, i32, <16 x i1>, <16 x i64>)

declare <512 x half> @llvm.masked.load.v512f16.p0(ptr, i32, <512 x i1>, <512 x half>)
declare <256 x float> @llvm.masked.load.v256f32.p0(ptr, i32, <256 x i1>, <256 x float>)
declare <128 x double> @llvm.masked.load.v128f64.p0(ptr, i32, <128 x i1>, <128 x double>)

attributes #0 = { "target-features"="+sve" }
