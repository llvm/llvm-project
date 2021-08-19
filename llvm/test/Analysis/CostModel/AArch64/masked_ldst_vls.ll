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

target triple = "aarch64-unknown-linux-gnu"

define void @fixed_sve_vls() #0 {
; CHECK-LABEL: 'fixed_sve_vls'
; CHECK:  Cost Model: Found an estimated cost of [[#mul(div(2047,VBITS)+1,2)]] for instruction: %v256i8 = call <256 x i8> @llvm.masked.load.v256i8.p0v256i8(<256 x i8>* undef, i32 8, <256 x i1> undef, <256 x i8> undef)
; CHECK:  Cost Model: Found an estimated cost of [[#mul(div(4091,VBITS)+1,2)]] for instruction: %v256i16 = call <256 x i16> @llvm.masked.load.v256i16.p0v256i16(<256 x i16>* undef, i32 8, <256 x i1> undef, <256 x i16> undef)
; CHECK:  Cost Model: Found an estimated cost of [[#mul(div(511,VBITS)+1,2)]] for instruction: %v16i32 = call <16 x i32> @llvm.masked.load.v16i32.p0v16i32(<16 x i32>* undef, i32 8, <16 x i1> undef, <16 x i32> undef)
; CHECK:  Cost Model: Found an estimated cost of [[#mul(div(1023,VBITS)+1,2)]] for instruction: %v16i64 = call <16 x i64> @llvm.masked.load.v16i64.p0v16i64(<16 x i64>* undef, i32 8, <16 x i1> undef, <16 x i64> undef)
; CHECK:  Cost Model: Found an estimated cost of [[#mul(div(8191,VBITS)+1,2)]] for instruction: %v512f16 = call <512 x half> @llvm.masked.load.v512f16.p0v512f16(<512 x half>* undef, i32 8, <512 x i1> undef, <512 x half> undef)
; CHECK:  Cost Model: Found an estimated cost of [[#mul(div(8191,VBITS)+1,2)]] for instruction: %v256f32 = call <256 x float> @llvm.masked.load.v256f32.p0v256f32(<256 x float>* undef, i32 8, <256 x i1> undef, <256 x float> undef)
; CHECK:  Cost Model: Found an estimated cost of [[#mul(div(8191,VBITS)+1,2)]] for instruction: %v128f64 = call <128 x double> @llvm.masked.load.v128f64.p0v128f64(<128 x double>* undef, i32 8, <128 x i1> undef, <128 x double> undef)
; CHECK:  Cost Model: Found an estimated cost of 0 for instruction: ret void
entry:
  %v256i8 = call <256 x i8> @llvm.masked.load.v256i8.p0v256i8(<256 x i8> *undef, i32 8, <256 x i1> undef, <256 x i8> undef)
  %v256i16 = call <256 x i16> @llvm.masked.load.v256i16.p0v256i16(<256 x i16> *undef, i32 8, <256 x i1> undef, <256 x i16> undef)
  %v16i32 = call <16 x i32> @llvm.masked.load.v16i32.p0v16i32(<16 x i32> *undef, i32 8, <16 x i1> undef, <16 x i32> undef)
  %v16i64 = call <16 x i64> @llvm.masked.load.v16i64.p0v16i64(<16 x i64> *undef, i32 8, <16 x i1> undef, <16 x i64> undef)

  %v512f16 = call <512 x half> @llvm.masked.load.v512f16.p0v512f16(<512 x half> *undef, i32 8, <512 x i1> undef, <512 x half> undef)
  %v256f32 = call <256 x float> @llvm.masked.load.v256f32.p0v256f32(<256 x float> *undef, i32 8, <256 x i1> undef, <256 x float> undef)
  %v128f64 = call <128 x double> @llvm.masked.load.v128f64.p0v128f64(<128 x double> *undef, i32 8, <128 x i1> undef, <128 x double> undef)

  ret void
}

declare <256 x i8> @llvm.masked.load.v256i8.p0v256i8(<256 x i8>*, i32, <256 x i1>, <256 x i8>)
declare <256 x i16> @llvm.masked.load.v256i16.p0v256i16(<256 x i16>*, i32, <256 x i1>, <256 x i16>)
declare <16 x i32> @llvm.masked.load.v16i32.p0v16i32(<16 x i32>*, i32, <16 x i1>, <16 x i32>)
declare <16 x i64> @llvm.masked.load.v16i64.p0v16i64(<16 x i64>*, i32, <16 x i1>, <16 x i64>)

declare <512 x half> @llvm.masked.load.v512f16.p0v512f16(<512 x half>*, i32, <512 x i1>, <512 x half>)
declare <256 x float> @llvm.masked.load.v256f32.p0v256f32(<256 x float>*, i32, <256 x i1>, <256 x float>)
declare <128 x double> @llvm.masked.load.v128f64.p0v128f64(<128 x double>*, i32, <128 x i1>, <128 x double>)

attributes #0 = { "target-features"="+sve" }
