; RUN: opt < %s -mtriple=systemz-unknown -mcpu=z13 -passes="print<cost-model>" -cost-kind=throughput 2>&1 -disable-output | FileCheck %s

define void @reduce(ptr %src, ptr %dst) {
; CHECK-LABEL: 'reduce'
; CHECK:  Cost Model: Found an estimated cost of 3 for instruction: %R2 = call i32 @llvm.vector.reduce.add.v2i32(<2 x i32> %V2)
; CHECK:  Cost Model: Found an estimated cost of 5 for instruction: %R4 = call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> %V4)
; CHECK:  Cost Model: Found an estimated cost of 7 for instruction: %R8 = call i32 @llvm.vector.reduce.add.v8i32(<8 x i32> %V8)
; CHECK:  Cost Model: Found an estimated cost of 11 for instruction: %R16 = call i32 @llvm.vector.reduce.add.v16i32(<16 x i32> %V16)
;
  %V2 = load <2 x i32>, ptr %src, align 8
  %R2 = call i32 @llvm.vector.reduce.add.v2i32(<2 x i32> %V2)
  store volatile i32 %R2, ptr %dst, align 4

  %V4 = load <4 x i32>, ptr %src, align 8
  %R4 = call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> %V4)
  store volatile i32 %R4, ptr %dst, align 4

  %V8 = load <8 x i32>, ptr %src, align 8
  %R8 = call i32 @llvm.vector.reduce.add.v8i32(<8 x i32> %V8)
  store volatile i32 %R8, ptr %dst, align 4

  %V16 = load <16 x i32>, ptr %src, align 8
  %R16 = call i32 @llvm.vector.reduce.add.v16i32(<16 x i32> %V16)
  store volatile i32 %R16, ptr %dst, align 4

  ret void
}

declare i32 @llvm.vector.reduce.add.v2i32(<2 x i32>)
declare i32 @llvm.vector.reduce.add.v4i32(<4 x i32>)
declare i32 @llvm.vector.reduce.add.v8i32(<8 x i32>)
declare i32 @llvm.vector.reduce.add.v16i32(<16 x i32>)