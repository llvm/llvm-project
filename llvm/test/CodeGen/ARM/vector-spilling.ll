; RUN: llc < %s -mtriple=armv7-linux-gnueabihf -arm-atomic-cfg-tidy=0 -float-abi=hard -mcpu=cortex-a9 -O3 | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n32-S64"

; This test will generate spills/fills using vldmia instructions that access 24 bytes of memory.
; Check that we don't crash when we generate these instructions on Cortex-A9.

; CHECK: test:
; CHECK: vstmia
; CHECK: vldmia
define void @test(<8 x i64>* %src) #0 {
entry:
  %0 = getelementptr inbounds <8 x i64>, <8 x i64>* %src, i32 0
  %1 = load <8 x i64>, <8 x i64>* %0, align 8

  %2 = getelementptr inbounds <8 x i64>, <8 x i64>* %src, i32 1
  %3 = load <8 x i64>, <8 x i64>* %2, align 8

  %4 = getelementptr inbounds <8 x i64>, <8 x i64>* %src, i32 2
  %5 = load <8 x i64>, <8 x i64>* %4, align 8

  %6 = getelementptr inbounds <8 x i64>, <8 x i64>* %src, i32 3
  %7 = load <8 x i64>, <8 x i64>* %6, align 8

  %8 = shufflevector <8 x i64> %1, <8 x i64> %3, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
  %9 = shufflevector <8 x i64> %1, <8 x i64> %3, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>

  tail call void(<8 x i64>, <8 x i64>, <8 x i64>, <8 x i64>, <8 x i64>, <8 x i64>) @foo(<8 x i64> %1, <8 x i64> %3, <8 x i64> %5, <8 x i64> %7, <8 x i64> %8, <8 x i64> %9)
  ret void
}

declare void @foo(<8 x i64>, <8 x i64>, <8 x i64>, <8 x i64>, <8 x i64>, <8 x i64>)

attributes #0 = { noredzone "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-frame-pointer-elim-non-leaf"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
