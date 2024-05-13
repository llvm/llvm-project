; RUN: llc -mtriple=thumb-eabi -mattr=-thumb2 %s -o - | FileCheck %s -check-prefix CHECK-T1
; RUN: llc -mtriple=thumb-eabi -mattr=+v7 %s -o - | FileCheck %s -check-prefix=THUMB2
; RUN: llc -mtriple=thumb-eabi -mattr=+v7 -mattr=+mp %s -o - | FileCheck %s -check-prefix=THUMB2-MP
; RUN: llc -mtriple=arm-eabi -mattr=+v7 %s -o - | FileCheck %s -check-prefix=ARM
; RUN: llc -mtriple=arm-eabi -mcpu=cortex-a9 %s -o - | FileCheck %s -check-prefix=ARM-MP
; rdar://8601536

; CHECK-T1-NOT: pld

define void @t1(ptr %ptr) nounwind  {
entry:
; ARM-LABEL: t1:
; ARM-NOT: pldw [r0]
; ARM: pld [r0]

; ARM-MP-LABEL: t1:
; ARM-MP: pldw [r0]
; ARM-MP: pld [r0]

; THUMB2-LABEL: t1:
; THUMB2-NOT: pldw [r0]
; THUMB2: pld [r0]
  tail call void @llvm.prefetch( ptr %ptr, i32 1, i32 3, i32 1 )
  tail call void @llvm.prefetch( ptr %ptr, i32 0, i32 3, i32 1 )
  ret void
}

define void @t2(ptr %ptr) nounwind  {
entry:
; ARM-LABEL: t2:
; ARM: pld [r0, #1023]

; THUMB2-LABEL: t2:
; THUMB2: pld [r0, #1023]
  %tmp = getelementptr i8, ptr %ptr, i32 1023
  tail call void @llvm.prefetch( ptr %tmp, i32 0, i32 3, i32 1 )
  ret void
}

define void @t3(i32 %base, i32 %offset) nounwind  {
entry:
; ARM-LABEL: t3:
; ARM: pld [r0, r1, lsr #2]

; THUMB2-LABEL: t3:
; THUMB2: lsrs r1, r1, #2
; THUMB2: pld [r0, r1]
  %tmp1 = lshr i32 %offset, 2
  %tmp2 = add i32 %base, %tmp1
  %tmp3 = inttoptr i32 %tmp2 to ptr
  tail call void @llvm.prefetch( ptr %tmp3, i32 0, i32 3, i32 1 )
  ret void
}

define void @t4(i32 %base, i32 %offset) nounwind  {
entry:
; ARM-LABEL: t4:
; ARM: pld [r0, r1, lsl #2]

; THUMB2-LABEL: t4:
; THUMB2: pld [r0, r1, lsl #2]
  %tmp1 = shl i32 %offset, 2
  %tmp2 = add i32 %base, %tmp1
  %tmp3 = inttoptr i32 %tmp2 to ptr
  tail call void @llvm.prefetch( ptr %tmp3, i32 0, i32 3, i32 1 )
  ret void
}

declare void @llvm.prefetch(ptr, i32, i32, i32) nounwind

define void @t5(ptr %ptr) nounwind  {
entry:
; ARM-LABEL: t5:
; ARM: pli [r0]

; THUMB2-LABEL: t5:
; THUMB2: pli [r0]
  tail call void @llvm.prefetch( ptr %ptr, i32 0, i32 3, i32 0 )
  ret void
}

define void @t6() {
entry:
;ARM-LABEL: t6:
;ARM: pld [sp]
;ARM: pld [sp, #50]
;ARM: pld [sp, #-50]

;THUMB2-LABEL: t6:
;THUMB2: pld [sp]
;THUMB2: pld [sp, #50]
;THUMB2: pld [sp, #-50]

%red = alloca [100 x i8], align 1
%0 = getelementptr inbounds [100 x i8], ptr %red, i32 0, i32 50
%1 = getelementptr inbounds [100 x i8], ptr %red, i32 0, i32 -50
call void @llvm.prefetch(ptr %red, i32 0, i32 3, i32 1)
call void @llvm.prefetch(ptr %0, i32 0, i32 3, i32 1)
call void @llvm.prefetch(ptr %1, i32 0, i32 3, i32 1)
ret void
}

define void @t7() {
entry:
;ARM-LABEL: t7:
;ARM-MP: pldw [sp]
;ARM-MP: pldw [sp, #50]
;ARM-MP: pldw [sp, #-50]

;THUMB2-MP-LABEL: t7:
;THUMB2-MP: pldw [sp]
;THUMB2-MP: pldw [sp, #50]
;THUMB2-MP: pldw [sp, #-50]

%red = alloca [100 x i8], align 1
%0 = getelementptr inbounds [100 x i8], ptr %red, i32 0, i32 50
%1 = getelementptr inbounds [100 x i8], ptr %red, i32 0, i32 -50
call void @llvm.prefetch(ptr %red, i32 1, i32 3, i32 1)
call void @llvm.prefetch(ptr %0, i32 1, i32 3, i32 1)
call void @llvm.prefetch(ptr %1, i32 1, i32 3, i32 1)
ret void
}

define void @t8() {
entry:
;ARM-LABEL: t8:
;ARM: pli [sp]
;ARM: pli [sp, #50]
;ARM: pli [sp, #-50]

;THUMB2-LABEL: t8:
;THUMB2: pli [sp]
;THUMB2: pli [sp, #50]
;THUMB2: pli [sp, #-50]

%red = alloca [100 x i8], align 1
%0 = getelementptr inbounds [100 x i8], ptr %red, i32 0, i32 50
%1 = getelementptr inbounds [100 x i8], ptr %red, i32 0, i32 -50
call void @llvm.prefetch(ptr %red, i32 0, i32 3, i32 0)
call void @llvm.prefetch(ptr %0, i32 0, i32 3, i32 0)
call void @llvm.prefetch(ptr %1, i32 0, i32 3, i32 0)
ret void
}
