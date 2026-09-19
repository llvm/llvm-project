; RUN: llc -mtriple=thumbv8m.main-none-eabi -mcpu=cortex-m33 -mattr=-dsp,-mve,-fpregs < %s | FileCheck %s --check-prefix=FAST --implicit-check-not=ldrb --implicit-check-not=strb
; RUN: llc -mtriple=thumbv8m.main-none-eabi -mcpu=cortex-m33 -mattr=-dsp,-mve,-fpregs,+strict-align < %s | FileCheck %s --check-prefix=STRICT --implicit-check-not=bic

define void @add4(ptr noalias %dst, ptr noalias readonly %src) {
; FAST-LABEL: add4:
; FAST: ldr{{(\.w)?}} {{.*}}, [r1]
; FAST: ldr{{(\.w)?}} {{.*}}, [r0]
; FAST: bic{{(\.w)?}} {{.*}}, {{.*}}, #-2139062144
; FAST: bic{{(\.w)?}} {{.*}}, {{.*}}, #-2139062144
; FAST: eor
; FAST: add
; FAST: bic{{(\.w)?}} {{.*}}, {{.*}}, #2139062143
; FAST: eor
; FAST: str{{(\.w)?}} {{.*}}, [r0]
; FAST: bx lr
; STRICT-LABEL: add4:
; STRICT: ldrb
; STRICT: strb
entry:
  %d0 = load i8, ptr %dst, align 1
  %s0 = load i8, ptr %src, align 1
  %a0 = add i8 %s0, %d0
  store i8 %a0, ptr %dst, align 1

  %dst1 = getelementptr inbounds i8, ptr %dst, i32 1
  %src1 = getelementptr inbounds i8, ptr %src, i32 1
  %d1 = load i8, ptr %dst1, align 1
  %s1 = load i8, ptr %src1, align 1
  %a1 = add i8 %s1, %d1
  store i8 %a1, ptr %dst1, align 1

  %dst2 = getelementptr inbounds i8, ptr %dst, i32 2
  %src2 = getelementptr inbounds i8, ptr %src, i32 2
  %d2 = load i8, ptr %dst2, align 1
  %s2 = load i8, ptr %src2, align 1
  %a2 = add i8 %s2, %d2
  store i8 %a2, ptr %dst2, align 1

  %dst3 = getelementptr inbounds i8, ptr %dst, i32 3
  %src3 = getelementptr inbounds i8, ptr %src, i32 3
  %d3 = load i8, ptr %dst3, align 1
  %s3 = load i8, ptr %src3, align 1
  %a3 = add i8 %s3, %d3
  store i8 %a3, ptr %dst3, align 1
  ret void
}
