; RUN: llc < %s -mtriple=arm64-eabi -mcpu=cyclone | FileCheck %s

define i32 @qadds(<4 x i32> %b, <4 x i32> %c) nounwind readnone optsize ssp {
; CHECK-LABEL: qadds:
; CHECK: sqadd s0, s0, s1
  %vecext = extractelement <4 x i32> %b, i32 0
  %vecext1 = extractelement <4 x i32> %c, i32 0
  %vqadd.i = tail call i32 @llvm.aarch64.neon.sqadd.i32(i32 %vecext, i32 %vecext1) nounwind
  ret i32 %vqadd.i
}

define i64 @qaddd(<2 x i64> %b, <2 x i64> %c) nounwind readnone optsize ssp {
; CHECK-LABEL: qaddd:
; CHECK: sqadd d0, d0, d1
  %vecext = extractelement <2 x i64> %b, i32 0
  %vecext1 = extractelement <2 x i64> %c, i32 0
  %vqadd.i = tail call i64 @llvm.aarch64.neon.sqadd.i64(i64 %vecext, i64 %vecext1) nounwind
  ret i64 %vqadd.i
}

define i32 @uqadds(<4 x i32> %b, <4 x i32> %c) nounwind readnone optsize ssp {
; CHECK-LABEL: uqadds:
; CHECK: uqadd s0, s0, s1
  %vecext = extractelement <4 x i32> %b, i32 0
  %vecext1 = extractelement <4 x i32> %c, i32 0
  %vqadd.i = tail call i32 @llvm.aarch64.neon.uqadd.i32(i32 %vecext, i32 %vecext1) nounwind
  ret i32 %vqadd.i
}

define i64 @uqaddd(<2 x i64> %b, <2 x i64> %c) nounwind readnone optsize ssp {
; CHECK-LABEL: uqaddd:
; CHECK: uqadd d0, d0, d1
  %vecext = extractelement <2 x i64> %b, i32 0
  %vecext1 = extractelement <2 x i64> %c, i32 0
  %vqadd.i = tail call i64 @llvm.aarch64.neon.uqadd.i64(i64 %vecext, i64 %vecext1) nounwind
  ret i64 %vqadd.i
}

declare i64 @llvm.aarch64.neon.uqadd.i64(i64, i64) nounwind readnone
declare i32 @llvm.aarch64.neon.uqadd.i32(i32, i32) nounwind readnone
declare i64 @llvm.aarch64.neon.sqadd.i64(i64, i64) nounwind readnone
declare i32 @llvm.aarch64.neon.sqadd.i32(i32, i32) nounwind readnone

define i32 @qsubs(<4 x i32> %b, <4 x i32> %c) nounwind readnone optsize ssp {
; CHECK-LABEL: qsubs:
; CHECK: sqsub s0, s0, s1
  %vecext = extractelement <4 x i32> %b, i32 0
  %vecext1 = extractelement <4 x i32> %c, i32 0
  %vqsub.i = tail call i32 @llvm.aarch64.neon.sqsub.i32(i32 %vecext, i32 %vecext1) nounwind
  ret i32 %vqsub.i
}

define i64 @qsubd(<2 x i64> %b, <2 x i64> %c) nounwind readnone optsize ssp {
; CHECK-LABEL: qsubd:
; CHECK: sqsub d0, d0, d1
  %vecext = extractelement <2 x i64> %b, i32 0
  %vecext1 = extractelement <2 x i64> %c, i32 0
  %vqsub.i = tail call i64 @llvm.aarch64.neon.sqsub.i64(i64 %vecext, i64 %vecext1) nounwind
  ret i64 %vqsub.i
}

define i32 @uqsubs(<4 x i32> %b, <4 x i32> %c) nounwind readnone optsize ssp {
; CHECK-LABEL: uqsubs:
; CHECK: uqsub s0, s0, s1
  %vecext = extractelement <4 x i32> %b, i32 0
  %vecext1 = extractelement <4 x i32> %c, i32 0
  %vqsub.i = tail call i32 @llvm.aarch64.neon.uqsub.i32(i32 %vecext, i32 %vecext1) nounwind
  ret i32 %vqsub.i
}

define i64 @uqsubd(<2 x i64> %b, <2 x i64> %c) nounwind readnone optsize ssp {
; CHECK-LABEL: uqsubd:
; CHECK: uqsub d0, d0, d1
  %vecext = extractelement <2 x i64> %b, i32 0
  %vecext1 = extractelement <2 x i64> %c, i32 0
  %vqsub.i = tail call i64 @llvm.aarch64.neon.uqsub.i64(i64 %vecext, i64 %vecext1) nounwind
  ret i64 %vqsub.i
}

declare i64 @llvm.aarch64.neon.uqsub.i64(i64, i64) nounwind readnone
declare i32 @llvm.aarch64.neon.uqsub.i32(i32, i32) nounwind readnone
declare i64 @llvm.aarch64.neon.sqsub.i64(i64, i64) nounwind readnone
declare i32 @llvm.aarch64.neon.sqsub.i32(i32, i32) nounwind readnone

define i32 @qabss(<4 x i32> %b, <4 x i32> %c) nounwind readnone {
; CHECK-LABEL: qabss:
; CHECK: sqabs s0, s0
; CHECK: ret
  %vecext = extractelement <4 x i32> %b, i32 0
  %vqabs.i = tail call i32 @llvm.aarch64.neon.sqabs.i32(i32 %vecext) nounwind
  ret i32 %vqabs.i
}

define i64 @qabsd(<2 x i64> %b, <2 x i64> %c) nounwind readnone {
; CHECK-LABEL: qabsd:
; CHECK: sqabs d0, d0
; CHECK: ret
  %vecext = extractelement <2 x i64> %b, i32 0
  %vqabs.i = tail call i64 @llvm.aarch64.neon.sqabs.i64(i64 %vecext) nounwind
  ret i64 %vqabs.i
}

define i32 @qnegs(<4 x i32> %b, <4 x i32> %c) nounwind readnone {
; CHECK-LABEL: qnegs:
; CHECK: sqneg s0, s0
; CHECK: ret
  %vecext = extractelement <4 x i32> %b, i32 0
  %vqneg.i = tail call i32 @llvm.aarch64.neon.sqneg.i32(i32 %vecext) nounwind
  ret i32 %vqneg.i
}

define i64 @qnegd(<2 x i64> %b, <2 x i64> %c) nounwind readnone {
; CHECK-LABEL: qnegd:
; CHECK: sqneg d0, d0
; CHECK: ret
  %vecext = extractelement <2 x i64> %b, i32 0
  %vqneg.i = tail call i64 @llvm.aarch64.neon.sqneg.i64(i64 %vecext) nounwind
  ret i64 %vqneg.i
}

declare i64 @llvm.aarch64.neon.sqneg.i64(i64) nounwind readnone
declare i32 @llvm.aarch64.neon.sqneg.i32(i32) nounwind readnone
declare i64 @llvm.aarch64.neon.sqabs.i64(i64) nounwind readnone
declare i32 @llvm.aarch64.neon.sqabs.i32(i32) nounwind readnone


define i32 @vqmovund(<2 x i64> %b) nounwind readnone {
; CHECK-LABEL: vqmovund:
; CHECK: sqxtun s0, d0
  %vecext = extractelement <2 x i64> %b, i32 0
  %vqmovun.i = tail call i32 @llvm.aarch64.neon.scalar.sqxtun.i32.i64(i64 %vecext) nounwind
  ret i32 %vqmovun.i
}

define i32 @vqmovnd_s(<2 x i64> %b) nounwind readnone {
; CHECK-LABEL: vqmovnd_s:
; CHECK: sqxtn s0, d0
  %vecext = extractelement <2 x i64> %b, i32 0
  %vqmovn.i = tail call i32 @llvm.aarch64.neon.scalar.sqxtn.i32.i64(i64 %vecext) nounwind
  ret i32 %vqmovn.i
}

define i32 @vqmovnd_u(<2 x i64> %b) nounwind readnone {
; CHECK-LABEL: vqmovnd_u:
; CHECK: uqxtn s0, d0
  %vecext = extractelement <2 x i64> %b, i32 0
  %vqmovn.i = tail call i32 @llvm.aarch64.neon.scalar.uqxtn.i32.i64(i64 %vecext) nounwind
  ret i32 %vqmovn.i
}

declare i32 @llvm.aarch64.neon.scalar.uqxtn.i32.i64(i64) nounwind readnone
declare i32 @llvm.aarch64.neon.scalar.sqxtn.i32.i64(i64) nounwind readnone
declare i32 @llvm.aarch64.neon.scalar.sqxtun.i32.i64(i64) nounwind readnone
