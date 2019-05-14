; RUN: llc < %s -mtriple=armv7-linux-gnueabihf -arm-atomic-cfg-tidy=0 -float-abi=hard -mcpu=cortex-a9 -O3 | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n32-S64"

; This test used to test vector spilling using vstmia/vldmia instructions, but
; the changes for PR:18825 prevent that spilling.

; VST1 and VLD1 are now used for spilling/restoring.
;
; TODO:
; I think more vldm should be generated, initial ones are used to load some
; elements and then a sequence of vldr are used:
; vldr  d15, [r1, #104]
; vldr  d13, [r2, #96]
; vldr  d9, [r1, #120]
; vldr  d11, [r2, #112]
; vldr  d14, [r1, #96]
; vldr  d12, [r2, #88]
; vldr  d8, [r1, #112]
; vldr  d10, [r2, #104]

; Also this patterns repeats several times which certainly seems like a vld1.64
; should be used to load the data:
; vldr  d16, [r1, #16]
; vldr  d17, [r1, #24]
; vst1.64 {d16, d17}, [lr:128]    @ 16-byte Spill

; CHECK: test:
; CHECK: vldmia r{{.*}}, {d{{.*}}, d{{.*}}}
; CHECK: vldmia r{{.*}}, {d{{.*}}, d{{.*}}}
define <16 x i64> @test(i64* %src0, i64* %src1) #0 {
entry:
  %addr.0 = getelementptr inbounds i64, i64* %src0, i32 0
  %el.0 = load i64, i64* %addr.0, align 8
  %addr.1 = getelementptr inbounds i64, i64* %src0, i32 1
  %el.1 = load i64, i64* %addr.1, align 8
  %addr.2 = getelementptr inbounds i64, i64* %src0, i32 2
  %el.2 = load i64, i64* %addr.2, align 8
  %addr.3 = getelementptr inbounds i64, i64* %src0, i32 3
  %el.3 = load i64, i64* %addr.3, align 8
  %addr.4 = getelementptr inbounds i64, i64* %src0, i32 4
  %el.4 = load i64, i64* %addr.4, align 8
  %addr.5 = getelementptr inbounds i64, i64* %src0, i32 5
  %el.5 = load i64, i64* %addr.5, align 8
  %addr.6 = getelementptr inbounds i64, i64* %src0, i32 6
  %el.6 = load i64, i64* %addr.6, align 8
  %addr.7 = getelementptr inbounds i64, i64* %src0, i32 7
  %el.7 = load i64, i64* %addr.7, align 8
  %addr.8 = getelementptr inbounds i64, i64* %src0, i32 8
  %el.8 = load i64, i64* %addr.8, align 8
  %addr.9 = getelementptr inbounds i64, i64* %src0, i32 9
  %el.9 = load i64, i64* %addr.9, align 8
  %addr.10 = getelementptr inbounds i64, i64* %src0, i32 10
  %el.10 = load i64, i64* %addr.10, align 8
  %addr.11 = getelementptr inbounds i64, i64* %src0, i32 11
  %el.11 = load i64, i64* %addr.11, align 8
  %addr.12 = getelementptr inbounds i64, i64* %src0, i32 12
  %el.12 = load i64, i64* %addr.12, align 8
  %addr.13 = getelementptr inbounds i64, i64* %src0, i32 13
  %el.13 = load i64, i64* %addr.13, align 8
  %addr.14 = getelementptr inbounds i64, i64* %src0, i32 14
  %el.14 = load i64, i64* %addr.14, align 8
  %addr.15 = getelementptr inbounds i64, i64* %src0, i32 15
  %el.15 = load i64, i64* %addr.15, align 8

  %addr.0.1 = getelementptr inbounds i64, i64* %src1, i32 0
  %el.0.1 = load i64, i64* %addr.0.1, align 8
  %addr.1.1 = getelementptr inbounds i64, i64* %src1, i32 1
  %el.1.1 = load i64, i64* %addr.1.1, align 8
  %addr.2.1 = getelementptr inbounds i64, i64* %src1, i32 2
  %el.2.1 = load i64, i64* %addr.2.1, align 8
  %addr.3.1 = getelementptr inbounds i64, i64* %src1, i32 3
  %el.3.1 = load i64, i64* %addr.3.1, align 8
  %addr.4.1 = getelementptr inbounds i64, i64* %src1, i32 4
  %el.4.1 = load i64, i64* %addr.4.1, align 8
  %addr.5.1 = getelementptr inbounds i64, i64* %src1, i32 5
  %el.5.1 = load i64, i64* %addr.5.1, align 8
  %addr.6.1 = getelementptr inbounds i64, i64* %src1, i32 6
  %el.6.1 = load i64, i64* %addr.6.1, align 8
  %addr.7.1 = getelementptr inbounds i64, i64* %src1, i32 7
  %el.7.1 = load i64, i64* %addr.7.1, align 8
  %addr.8.1 = getelementptr inbounds i64, i64* %src1, i32 8
  %el.8.1 = load i64, i64* %addr.8.1, align 8
  %addr.9.1 = getelementptr inbounds i64, i64* %src1, i32 9
  %el.9.1 = load i64, i64* %addr.9.1, align 8
  %addr.10.1 = getelementptr inbounds i64, i64* %src1, i32 10
  %el.10.1 = load i64, i64* %addr.10.1, align 8
  %addr.11.1 = getelementptr inbounds i64, i64* %src1, i32 11
  %el.11.1 = load i64, i64* %addr.11.1, align 8
  %addr.12.1 = getelementptr inbounds i64, i64* %src1, i32 12
  %el.12.1 = load i64, i64* %addr.12.1, align 8
  %addr.13.1 = getelementptr inbounds i64, i64* %src1, i32 13
  %el.13.1 = load i64, i64* %addr.13.1, align 8
  %addr.14.1 = getelementptr inbounds i64, i64* %src1, i32 14
  %el.14.1 = load i64, i64* %addr.14.1, align 8
  %addr.15.1 = getelementptr inbounds i64, i64* %src1, i32 15
  %el.15.1 = load i64, i64* %addr.15.1, align 8
  %vec.0 = insertelement <16 x i64> undef, i64 %el.0, i32 0
  %vec.1 = insertelement <16 x i64> %vec.0, i64 %el.1, i32 1
  %vec.2 = insertelement <16 x i64> %vec.1, i64 %el.2, i32 2
  %vec.3 = insertelement <16 x i64> %vec.2, i64 %el.3, i32 3
  %vec.4 = insertelement <16 x i64> %vec.3, i64 %el.4, i32 4
  %vec.5 = insertelement <16 x i64> %vec.4, i64 %el.5, i32 5
  %vec.6 = insertelement <16 x i64> %vec.5, i64 %el.6, i32 6
  %vec.7 = insertelement <16 x i64> %vec.6, i64 %el.7, i32 7
  %vec.8 = insertelement <16 x i64> %vec.7, i64 %el.8, i32 8
  %vec.9 = insertelement <16 x i64> %vec.8, i64 %el.9, i32 9
  %vec.10 = insertelement <16 x i64> %vec.9, i64 %el.10, i32 10
  %vec.11 = insertelement <16 x i64> %vec.10, i64 %el.11, i32 11
  %vec.12 = insertelement <16 x i64> %vec.11, i64 %el.12, i32 12
  %vec.13 = insertelement <16 x i64> %vec.12, i64 %el.13, i32 13
  %vec.14 = insertelement <16 x i64> %vec.13, i64 %el.14, i32 14
  %vec.15 = insertelement <16 x i64> %vec.14, i64 %el.15, i32 15
  call void @capture(i64* %src0, i64* %src1)
  %vec.0.1 = insertelement <16 x i64> undef, i64 %el.0.1, i32 0
  %vec.1.1 = insertelement <16 x i64> %vec.0.1, i64 %el.1.1, i32 1
  %vec.2.1 = insertelement <16 x i64> %vec.1.1, i64 %el.2.1, i32 2
  %vec.3.1 = insertelement <16 x i64> %vec.2.1, i64 %el.3.1, i32 3
  %vec.4.1 = insertelement <16 x i64> %vec.3.1, i64 %el.4.1, i32 4
  %vec.5.1 = insertelement <16 x i64> %vec.4.1, i64 %el.5.1, i32 5
  %vec.6.1 = insertelement <16 x i64> %vec.5.1, i64 %el.6.1, i32 6
  %vec.7.1 = insertelement <16 x i64> %vec.6.1, i64 %el.7.1, i32 7
  %vec.8.1 = insertelement <16 x i64> %vec.7.1, i64 %el.7.1, i32 8
  %vec.9.1 = insertelement <16 x i64> %vec.8.1, i64 %el.8.1, i32 9
  %vec.10.1 = insertelement <16 x i64> %vec.9.1, i64 %el.9.1, i32 10
  %vec.11.1 = insertelement <16 x i64> %vec.10.1, i64 %el.10.1, i32 11
  %vec.12.1 = insertelement <16 x i64> %vec.11.1, i64 %el.11.1, i32 12
  %vec.13.1 = insertelement <16 x i64> %vec.12.1, i64 %el.12.1, i32 13
  %vec.14.1 = insertelement <16 x i64> %vec.13.1, i64 %el.13.1, i32 14
  %vec.15.1 = insertelement <16 x i64> %vec.14.1, i64 %el.14.1, i32 15
  %res = add <16 x i64> %vec.15, %vec.15.1
  ret <16 x i64> %res
}

declare void @capture(i64*, i64*)

attributes #0 = { noredzone "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-frame-pointer-elim-non-leaf"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
