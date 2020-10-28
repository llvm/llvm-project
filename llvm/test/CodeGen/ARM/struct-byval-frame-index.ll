; RUN: llc < %s -mcpu=cortex-a15 -verify-machineinstrs -arm-atomic-cfg-tidy=0 | FileCheck %s

; Check a spill right after a function call with large struct byval is correctly
; generated.
; PR16393

; We expect 4-byte spill and reload to be generated.

; CHECK-LABEL: set_stored_macroblock_parameters:
; CHECK:         str r0, [sp, #{{[0-9]}}] @ 4-byte Spill
; CHECK:         @APP
; CHECK:         bl RestoreMVBlock8x8
; CHECK:         ldr r0, [sp, #{{[0-9]}}] @ 4-byte Reload

target triple = "armv7l-unknown-linux-gnueabihf"

%structN = type { i32, [16 x [16 x i32]], [16 x [16 x i32]], [16 x [16 x i32]], [3 x [16 x [16 x i32]]], [4 x i16], [4 x i8], [4 x i8], [4 x i8], [16 x [16 x i16]], [16 x [16 x i16]], [16 x [16 x i32]] }

@tr8x8 = external global %structN, align 4
@luma_transform_size_8x8_flag = external global i32, align 4

; Function Attrs: nounwind
define void @set_stored_macroblock_parameters(i16 %a0, i32 %a1) #1 {
entry:
  %0 = load i32, i32* @luma_transform_size_8x8_flag, align 4
  tail call void asm sideeffect "", "~{r1},~{r2},~{r3},~{r4},~{r5},~{r6},~{r7},~{r8},~{r9},~{r10},~{r11}"()
  tail call void @RestoreMVBlock8x8(i32 1, i32 2, %structN* byval(%structN) @tr8x8, i32 0)
  %arrayidx313 = getelementptr inbounds i8*, i8** null, i32 %0
  %1 = load i8*, i8** %arrayidx313, align 4
  %arrayidx314 = getelementptr inbounds i8, i8* %1, i32 0
  store i8 -1, i8* %arrayidx314, align 1
  ret void
}

; Function Attrs: nounwind
declare void @RestoreMVBlock8x8(i32, i32, %structN* byval nocapture, i32) #1

attributes #1 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
