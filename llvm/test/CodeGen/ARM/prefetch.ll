; RUN: llc < %s -march=thumb -mattr=-thumb2 | not grep pld
; RUN: llc < %s -march=thumb -mattr=+v7         | FileCheck %s -check-prefix=THUMB2
; RUN: llc < %s -march=arm   -mattr=+v7         | FileCheck %s -check-prefix=ARM
; RUN: llc < %s -march=arm   -mcpu=cortex-a9-mp | FileCheck %s -check-prefix=ARM-MP
; rdar://8601536

define void @t1(i8* %ptr) nounwind  {
entry:
; ARM: t1:
; ARM-NOT: pldw [r0]
; ARM: pld [r0]

; ARM-MP: t1:
; ARM-MP: pldw [r0]
; ARM-MP: pld [r0]

; THUMB2: t1:
; THUMB2-NOT: pldw [r0]
; THUMB2: pld [r0]
  tail call void @llvm.prefetch( i8* %ptr, i32 1, i32 3, i32 1 )
  tail call void @llvm.prefetch( i8* %ptr, i32 0, i32 3, i32 1 )
  ret void
}

define void @t2(i8* %ptr) nounwind  {
entry:
; ARM: t2:
; ARM: pld [r0, #1023]

; THUMB2: t2:
; THUMB2: pld [r0, #1023]
  %tmp = getelementptr i8* %ptr, i32 1023
  tail call void @llvm.prefetch( i8* %tmp, i32 0, i32 3, i32 1 )
  ret void
}

define void @t3(i32 %base, i32 %offset) nounwind  {
entry:
; ARM: t3:
; ARM: pld [r0, r1, lsr #2]

; THUMB2: t3:
; THUMB2: lsrs r1, r1, #2
; THUMB2: pld [r0, r1]
  %tmp1 = lshr i32 %offset, 2
  %tmp2 = add i32 %base, %tmp1
  %tmp3 = inttoptr i32 %tmp2 to i8*
  tail call void @llvm.prefetch( i8* %tmp3, i32 0, i32 3, i32 1 )
  ret void
}

define void @t4(i32 %base, i32 %offset) nounwind  {
entry:
; ARM: t4:
; ARM: pld [r0, r1, lsl #2]

; THUMB2: t4:
; THUMB2: pld [r0, r1, lsl #2]
  %tmp1 = shl i32 %offset, 2
  %tmp2 = add i32 %base, %tmp1
  %tmp3 = inttoptr i32 %tmp2 to i8*
  tail call void @llvm.prefetch( i8* %tmp3, i32 0, i32 3, i32 1 )
  ret void
}

declare void @llvm.prefetch(i8*, i32, i32, i32) nounwind

define void @t5(i8* %ptr) nounwind  {
entry:
; ARM: t5:
; ARM: pli [r0]

; THUMB2: t5:
; THUMB2: pli [r0]
  tail call void @llvm.prefetch( i8* %ptr, i32 0, i32 3, i32 0 )
  ret void
}
