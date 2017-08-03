; RUN: llc < %s -mtriple=thumb-linux-androideabi -march=thumb -mcpu=arm1156t2-s -mattr=+thumb2 -verify-machineinstrs | FileCheck %s -check-prefix=Thumb-android
; RUN: llc < %s -mtriple=thumb-linux-androideabi -march=thumb -mcpu=arm1156t2-s -mattr=+thumb2 -filetype=obj


; Just to prevent the alloca from being optimized away
declare void @dummy_use(i32*, i32)

define void @test_basic() #0 {
        %mem = alloca i32, i32 10
        call void @dummy_use (i32* %mem, i32 10)
	ret void

; Thumb-android:      test_basic:

; Thumb-android:      push    {r4, r5}
; Thumb-android-NEXT: mrc     p15, #0, r4, c13, c0, #3
; Thumb-android-NEXT: mov     r5, sp
; Thumb-android-NEXT: ldr     r4, [r4, #252]
; Thumb-android-NEXT: cmp     r4, r5
; Thumb-android-NEXT: blo     .LBB0_2

; Thumb-android:      mov     r4, #48
; Thumb-android-NEXT: mov     r5, #0
; Thumb-android-NEXT: push    {lr}
; Thumb-android-NEXT: bl      __morestack
; Thumb-android-NEXT: ldr     lr, [sp], #4
; Thumb-android-NEXT: pop     {r4, r5}
; Thumb-android-NEXT: bx      lr

; Thumb-android:      pop     {r4, r5}

}

attributes #0 = { "split-stack" }
