; RUN: llc < %s -mtriple=armv7-linux-gnueabihf -arm-atomic-cfg-tidy=0 -float-abi=hard -mcpu=cortex-a9 -O3 | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n32-S64"

; This test used to test vector spilling using vstmia/vldmia instructions, but
; the changes for PR:18825 prevent that spilling.

; CHECK: test:
; CHECK: vstmia
; CHECK: vldmia
define void @test(i64* %src) #0 {
entry:
  %arrayidx39 = getelementptr inbounds i64, i64* %src, i32 13
  %vecinit285 = shufflevector <16 x i64> undef, <16 x i64> <i64 15, i64 16, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef>, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 16, i32 17>
  store <16 x i64> %vecinit285, <16 x i64>* undef, align 128
  %0 = load i64, i64* undef, align 8
  %vecinit379 = insertelement <16 x i64> undef, i64 %0, i32 9
  %1 = load i64, i64* undef, align 8
  %vecinit419 = insertelement <16 x i64> undef, i64 %1, i32 15
  store <16 x i64> %vecinit419, <16 x i64>* undef, align 128
  %vecinit579 = insertelement <16 x i64> undef, i64 0, i32 4
  %vecinit582 = shufflevector <16 x i64> %vecinit579, <16 x i64> <i64 6, i64 7, i64 8, i64 9, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef>, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 16, i32 17, i32 18, i32 19, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %vecinit584 = insertelement <16 x i64> %vecinit582, i64 undef, i32 9
  %vecinit586 = insertelement <16 x i64> %vecinit584, i64 0, i32 10
  %vecinit589 = shufflevector <16 x i64> %vecinit586, <16 x i64> <i64 12, i64 13, i64 14, i64 15, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef>, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 16, i32 17, i32 18, i32 19, i32 undef>
  %2 = load i64, i64* undef, align 8
  %vecinit591 = insertelement <16 x i64> %vecinit589, i64 %2, i32 15
  store <16 x i64> %vecinit591, <16 x i64>* undef, align 128
  %vecinit694 = shufflevector <16 x i64> undef, <16 x i64> <i64 13, i64 14, i64 15, i64 16, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef>, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 16, i32 17, i32 18, i32 19>
  store <16 x i64> %vecinit694, <16 x i64>* undef, align 128
  %3 = load i64, i64* undef, align 8
  %vecinit1331 = insertelement <16 x i64> undef, i64 %3, i32 14
  %4 = load i64, i64* undef, align 8
  %vecinit1468 = insertelement <16 x i64> undef, i64 %4, i32 11
  %vecinit1471 = shufflevector <16 x i64> %vecinit1468, <16 x i64> <i64 13, i64 14, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef>, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 16, i32 17, i32 undef, i32 undef>
  %vecinit1474 = shufflevector <16 x i64> %vecinit1471, <16 x i64> <i64 15, i64 16, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef>, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 16, i32 17>
  store <16 x i64> %vecinit1474, <16 x i64>* undef, align 128
  %vecinit1552 = shufflevector <16 x i64> undef, <16 x i64> <i64 10, i64 11, i64 12, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef>, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 16, i32 17, i32 18, i32 undef, i32 undef, i32 undef, i32 undef>
  %vecinit1555 = shufflevector <16 x i64> %vecinit1552, <16 x i64> <i64 13, i64 14, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef>, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 16, i32 17, i32 undef, i32 undef>
  %vecinit1558 = shufflevector <16 x i64> %vecinit1555, <16 x i64> <i64 15, i64 16, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef>, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 16, i32 17>
  store <16 x i64> %vecinit1558, <16 x i64>* undef, align 128
  %vecinit1591 = shufflevector <16 x i64> undef, <16 x i64> <i64 3, i64 4, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef>, <16 x i32> <i32 0, i32 1, i32 16, i32 17, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %vecinit1594 = shufflevector <16 x i64> %vecinit1591, <16 x i64> <i64 5, i64 6, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef>, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 16, i32 17, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %vecinit1597 = shufflevector <16 x i64> %vecinit1594, <16 x i64> <i64 7, i64 8, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef>, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 16, i32 17, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %vecinit1599 = insertelement <16 x i64> %vecinit1597, i64 undef, i32 8
  %vecinit1601 = insertelement <16 x i64> %vecinit1599, i64 undef, i32 9
  %vecinit1603 = insertelement <16 x i64> %vecinit1601, i64 undef, i32 10
  %5 = load i64, i64* undef, align 8
  %vecinit1605 = insertelement <16 x i64> %vecinit1603, i64 %5, i32 11
  %vecinit1608 = shufflevector <16 x i64> %vecinit1605, <16 x i64> <i64 13, i64 14, i64 15, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef>, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 16, i32 17, i32 18, i32 undef>
  %6 = load i64, i64* undef, align 8
  %vecinit1610 = insertelement <16 x i64> %vecinit1608, i64 %6, i32 15
  store <16 x i64> %vecinit1610, <16 x i64>* undef, align 128
  %vecinit2226 = shufflevector <16 x i64> undef, <16 x i64> <i64 6, i64 7, i64 8, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef>, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 16, i32 17, i32 18, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %7 = load i64, i64* undef, align 8
  %vecinit2228 = insertelement <16 x i64> %vecinit2226, i64 %7, i32 8
  %vecinit2230 = insertelement <16 x i64> %vecinit2228, i64 undef, i32 9
  %vecinit2233 = shufflevector <16 x i64> %vecinit2230, <16 x i64> <i64 11, i64 12, i64 13, i64 14, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef>, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 16, i32 17, i32 18, i32 19, i32 undef, i32 undef>
  %vecinit2236 = shufflevector <16 x i64> %vecinit2233, <16 x i64> <i64 15, i64 16, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef>, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 16, i32 17>
  store <16 x i64> %vecinit2236, <16 x i64>* undef, align 128
  %vecinit2246 = shufflevector <16 x i64> undef, <16 x i64> <i64 4, i64 5, i64 6, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef>, <16 x i32> <i32 0, i32 1, i32 2, i32 16, i32 17, i32 18, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %vecinit2249 = shufflevector <16 x i64> %vecinit2246, <16 x i64> <i64 7, i64 8, i64 9, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef>, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 16, i32 17, i32 18, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %vecinit2252 = shufflevector <16 x i64> %vecinit2249, <16 x i64> <i64 10, i64 11, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef>, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 16, i32 17, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %vecinit2255 = shufflevector <16 x i64> %vecinit2252, <16 x i64> <i64 12, i64 13, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef>, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 16, i32 17, i32 undef, i32 undef, i32 undef>
  %8 = load i64, i64* %arrayidx39, align 8
  %vecinit2257 = insertelement <16 x i64> %vecinit2255, i64 %8, i32 13
  %vecinit2260 = shufflevector <16 x i64> %vecinit2257, <16 x i64> <i64 15, i64 16, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef>, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 16, i32 17>
  store <16 x i64> %vecinit2260, <16 x i64>* null, align 128
  ret void
}
attributes #0 = { noredzone "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-frame-pointer-elim-non-leaf"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
