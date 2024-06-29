; RUN: opt < %s -mtriple=systemz-unknown -mcpu=z13 -passes="print<cost-model>" -cost-kind=throughput 2>&1 -disable-output | FileCheck %s

define void @reduce(ptr %src, ptr %dst) {
; CHECK-LABEL: 'reduce'
; CHECK:  Cost Model: Found an estimated cost of 2 for instruction: %R2_64 = call i64 @llvm.vector.reduce.add.v2i64(<2 x i64> %V2_64)
; CHECK:  Cost Model: Found an estimated cost of 3 for instruction: %R4_64 = call i64 @llvm.vector.reduce.add.v4i64(<4 x i64> %V4_64)
; CHECK:  Cost Model: Found an estimated cost of 5 for instruction: %R8_64 = call i64 @llvm.vector.reduce.add.v8i64(<8 x i64> %V8_64)
; CHECK:  Cost Model: Found an estimated cost of 9 for instruction: %R16_64 = call i64 @llvm.vector.reduce.add.v16i64(<16 x i64> %V16_64)
; CHECK:  Cost Model: Found an estimated cost of 2 for instruction: %R2_32 = call i32 @llvm.vector.reduce.add.v2i32(<2 x i32> %V2_32)
; CHECK:  Cost Model: Found an estimated cost of 2 for instruction: %R4_32 = call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> %V4_32)
; CHECK:  Cost Model: Found an estimated cost of 3 for instruction: %R8_32 = call i32 @llvm.vector.reduce.add.v8i32(<8 x i32> %V8_32)
; CHECK:  Cost Model: Found an estimated cost of 5 for instruction: %R16_32 = call i32 @llvm.vector.reduce.add.v16i32(<16 x i32> %V16_32)
; CHECK:  Cost Model: Found an estimated cost of 3 for instruction: %R2_16 = call i16 @llvm.vector.reduce.add.v2i16(<2 x i16> %V2_16)
; CHECK:  Cost Model: Found an estimated cost of 3 for instruction: %R4_16 = call i16 @llvm.vector.reduce.add.v4i16(<4 x i16> %V4_16)
; CHECK:  Cost Model: Found an estimated cost of 3 for instruction: %R8_16 = call i16 @llvm.vector.reduce.add.v8i16(<8 x i16> %V8_16)
; CHECK:  Cost Model: Found an estimated cost of 4 for instruction: %R16_16 = call i16 @llvm.vector.reduce.add.v16i16(<16 x i16> %V16_16)
; CHECK:  Cost Model: Found an estimated cost of 3 for instruction: %R2_8 = call i8 @llvm.vector.reduce.add.v2i8(<2 x i8> %V2_8)
; CHECK:  Cost Model: Found an estimated cost of 3 for instruction: %R4_8 = call i8 @llvm.vector.reduce.add.v4i8(<4 x i8> %V4_8)
; CHECK:  Cost Model: Found an estimated cost of 3 for instruction: %R8_8 = call i8 @llvm.vector.reduce.add.v8i8(<8 x i8> %V8_8)
; CHECK:  Cost Model: Found an estimated cost of 3 for instruction: %R16_8 = call i8 @llvm.vector.reduce.add.v16i8(<16 x i8> %V16_8)
;
; CHECK:  Cost Model: Found an estimated cost of 10 for instruction: %R128_8 = call i8 @llvm.vector.reduce.add.v128i8(<128 x i8> %V128_8)
; CHECK:  Cost Model: Found an estimated cost of 20 for instruction: %R4_256 = call i256 @llvm.vector.reduce.add.v4i256(<4 x i256> %V4_256)

  ; REDUCEADD64

  %V2_64 = load <2 x i64>, ptr %src, align 8
  %R2_64 = call i64 @llvm.vector.reduce.add.v2i64(<2 x i64> %V2_64)
  store volatile i64 %R2_64, ptr %dst, align 4

  %V4_64 = load <4 x i64>, ptr %src, align 8
  %R4_64 = call i64 @llvm.vector.reduce.add.v4i64(<4 x i64> %V4_64)
  store volatile i64 %R4_64, ptr %dst, align 4

  %V8_64 = load <8 x i64>, ptr %src, align 8
  %R8_64 = call i64 @llvm.vector.reduce.add.v8i64(<8 x i64> %V8_64)
  store volatile i64 %R8_64, ptr %dst, align 4

  %V16_64 = load <16 x i64>, ptr %src, align 8
  %R16_64 = call i64 @llvm.vector.reduce.add.v16i64(<16 x i64> %V16_64)
  store volatile i64 %R16_64, ptr %dst, align 4

  ; REDUCEADD32

  %V2_32 = load <2 x i32>, ptr %src, align 8
  %R2_32 = call i32 @llvm.vector.reduce.add.v2i32(<2 x i32> %V2_32)
  store volatile i32 %R2_32, ptr %dst, align 4

  %V4_32 = load <4 x i32>, ptr %src, align 8
  %R4_32 = call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> %V4_32)
  store volatile i32 %R4_32, ptr %dst, align 4

  %V8_32 = load <8 x i32>, ptr %src, align 8
  %R8_32 = call i32 @llvm.vector.reduce.add.v8i32(<8 x i32> %V8_32)
  store volatile i32 %R8_32, ptr %dst, align 4

  %V16_32 = load <16 x i32>, ptr %src, align 8
  %R16_32 = call i32 @llvm.vector.reduce.add.v16i32(<16 x i32> %V16_32)
  store volatile i32 %R16_32, ptr %dst, align 4

  ; REDUCEADD16

  %V2_16 = load <2 x i16>, ptr %src, align 8
  %R2_16 = call i16 @llvm.vector.reduce.add.v2i16(<2 x i16> %V2_16)
  store volatile i16 %R2_16, ptr %dst, align 4

  %V4_16 = load <4 x i16>, ptr %src, align 8
  %R4_16 = call i16 @llvm.vector.reduce.add.v4i16(<4 x i16> %V4_16)
  store volatile i16 %R4_16, ptr %dst, align 4

  %V8_16 = load <8 x i16>, ptr %src, align 8
  %R8_16 = call i16 @llvm.vector.reduce.add.v8i16(<8 x i16> %V8_16)
  store volatile i16 %R8_16, ptr %dst, align 4

  %V16_16 = load <16 x i16>, ptr %src, align 8
  %R16_16 = call i16 @llvm.vector.reduce.add.v16i16(<16 x i16> %V16_16)
  store volatile i16 %R16_16, ptr %dst, align 4

  ; REDUCEADD8

  %V2_8 = load <2 x i8>, ptr %src, align 8
  %R2_8 = call i8 @llvm.vector.reduce.add.v2i8(<2 x i8> %V2_8)
  store volatile i8 %R2_8, ptr %dst, align 4

  %V4_8 = load <4 x i8>, ptr %src, align 8
  %R4_8 = call i8 @llvm.vector.reduce.add.v4i8(<4 x i8> %V4_8)
  store volatile i8 %R4_8, ptr %dst, align 4

  %V8_8 = load <8 x i8>, ptr %src, align 8
  %R8_8 = call i8 @llvm.vector.reduce.add.v8i8(<8 x i8> %V8_8)
  store volatile i8 %R8_8, ptr %dst, align 4

  %V16_8 = load <16 x i8>, ptr %src, align 8
  %R16_8 = call i8 @llvm.vector.reduce.add.v16i8(<16 x i8> %V16_8)
  store volatile i8 %R16_8, ptr %dst, align 4

  ; EXTREME VALUES

  %V128_8 = load <128 x i8>, ptr %src, align 8
  %R128_8 = call i8 @llvm.vector.reduce.add.v128i8(<128 x i8> %V128_8)
  store volatile i8 %R128_8, ptr %dst, align 4

  %V4_256 = load <4 x i256>, ptr %src, align 8
  %R4_256 = call i256 @llvm.vector.reduce.add.v4i256(<4 x i256> %V4_256)
  store volatile i256 %R4_256, ptr %dst, align 8

  ret void
}

declare i64 @llvm.vector.reduce.add.v2i64(<2 x i64>)
declare i64 @llvm.vector.reduce.add.v4i64(<4 x i64>)
declare i64 @llvm.vector.reduce.add.v8i64(<8 x i64>)
declare i64 @llvm.vector.reduce.add.v16i64(<16 x i64>)
declare i32 @llvm.vector.reduce.add.v2i32(<2 x i32>)
declare i32 @llvm.vector.reduce.add.v4i32(<4 x i32>)
declare i32 @llvm.vector.reduce.add.v8i32(<8 x i32>)
declare i32 @llvm.vector.reduce.add.v16i32(<16 x i32>)
declare i16 @llvm.vector.reduce.add.v2i16(<2 x i16>)
declare i16 @llvm.vector.reduce.add.v4i16(<4 x i16>)
declare i16 @llvm.vector.reduce.add.v8i16(<8 x i16>)
declare i16 @llvm.vector.reduce.add.v16i16(<16 x i16>)
declare i8 @llvm.vector.reduce.add.v2i8(<2 x i8>)
declare i8 @llvm.vector.reduce.add.v4i8(<4 x i8>)
declare i8 @llvm.vector.reduce.add.v8i8(<8 x i8>)
declare i8 @llvm.vector.reduce.add.v16i8(<16 x i8>)

declare i8 @llvm.vector.reduce.add.v128i8(<128 x i8>)
declare i256 @llvm.vector.reduce.add.v4i256(<4 x i256>)
