; RUN: llc < %s -mtriple=thumb-linux-unknown-gnueabi -verify-machineinstrs | FileCheck %s -check-prefix=Thumb-linux
; RUN: llc < %s -mtriple=thumb-linux-androideabi -verify-machineinstrs | FileCheck %s -check-prefix=Thumb-android
; RUN: llc < %s -mtriple=thumb-linux-unknown-gnueabi -filetype=obj
; RUN: llc < %s -mtriple=thumb-linux-androideabi -filetype=obj

; Just to prevent the alloca from being optimized away
declare void @dummy_use(ptr, i32)

define i32 @test_basic(i32 %l) #0 {
        %mem = alloca i32, i32 %l
        call void @dummy_use (ptr %mem, i32 %l)
        %terminate = icmp eq i32 %l, 0
        br i1 %terminate, label %true, label %false

true:
        ret i32 0

false:
        %newlen = sub i32 %l, 1
        %retvalue = call i32 @test_basic(i32 %newlen)
        ret i32 %retvalue

; Thumb-linux:      test_basic:

; Thumb-linux:      push {r4, r5}
; Thumb-linux:      mov	r5, sp
; Thumb-linux-NEXT: ldr r4, .LCPI0_0
; Thumb-linux-NEXT: ldr r4, [r4]
; Thumb-linux-NEXT: cmp	r4, r5
; Thumb-linux-NEXT: bls	.LBB0_2

; Thumb-linux:      mov r4, #16
; Thumb-linux-NEXT: mov r5, #0
; Thumb-linux-NEXT: push {lr}
; Thumb-linux-NEXT: bl	__morestack
; Thumb-linux-NEXT: pop {r4}
; Thumb-linux-NEXT: mov lr, r4
; Thumb-linux-NEXT: pop {r4, r5}
; Thumb-linux-NEXT: bx lr

; Thumb-linux:      pop {r4, r5}

; Thumb-android:      test_basic:

; Thumb-android:      push {r4, r5}
; Thumb-android:      mov	r5, sp
; Thumb-android-NEXT: ldr r4, .LCPI0_0
; Thumb-android-NEXT: ldr r4, [r4]
; Thumb-android-NEXT: cmp	r4, r5
; Thumb-android-NEXT: bls	.LBB0_2

; Thumb-android:      mov r4, #16
; Thumb-android-NEXT: mov r5, #0
; Thumb-android-NEXT: push {lr}
; Thumb-android-NEXT: bl __morestack
; Thumb-android-NEXT: pop {r4}
; Thumb-android-NEXT: mov lr, r4
; Thumb-android-NEXT: pop {r4, r5}
; Thumb-android-NEXT: bx lr

; Thumb-android:      pop {r4, r5}

}

attributes #0 = { "split-stack" }
