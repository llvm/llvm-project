; Check EVEX is not compressed into VEX when egpr is used.
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx512vl,+egpr -show-mc-encoding | FileCheck %s

define void @test_x86_vcvtps2ph_256_m(ptr nocapture %d, <8 x float> %a) nounwind {
; CHECK:    vcvtps2ph $3, %ymm0, (%r16) # encoding: [0x62,0xfb,0x7d,0x28,0x1d,0x00,0x03]
entry:
  %0 = load i32, ptr %d, align 4
  call void asm sideeffect "", "~{eax},~{ebx},~{ecx},~{edx},~{edi},~{esi},~{ebp},~{esp},~{r8d},~{r9d},~{r10d},~{r11d},~{r12d},~{r13d},~{r14d},~{r15d}"()
  %1 = tail call <8 x i16> @llvm.x86.vcvtps2ph.256(<8 x float> %a, i32 3)
  store <8 x i16> %1, ptr %d, align 16
  ret void
}

declare <8 x i16> @llvm.x86.vcvtps2ph.256(<8 x float>, i32) nounwind readonly
