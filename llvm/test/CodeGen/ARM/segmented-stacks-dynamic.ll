; RUN: llc < %s -mtriple=arm-linux-androideabi -verify-machineinstrs | FileCheck %s -check-prefix=ARM-android
; RUN: llc < %s -mtriple=arm-linux-unknown-gnueabi -verify-machineinstrs | FileCheck %s -check-prefix=ARM-linux
; RUN: llc < %s -mtriple=arm-linux-androideabi -filetype=obj
; RUN: llc < %s -mtriple=arm-linux-unknown-gnueabi -filetype=obj

; Just to prevent the alloca from being optimized away
declare void @dummy_use(i32*, i32)

define i32 @test_basic(i32 %l) #0 {
        %mem = alloca i32, i32 %l
        call void @dummy_use (i32* %mem, i32 %l)
        %terminate = icmp eq i32 %l, 0
        br i1 %terminate, label %true, label %false

true:
        ret i32 0

false:
        %newlen = sub i32 %l, 1
        %retvalue = call i32 @test_basic(i32 %newlen)
        ret i32 %retvalue

; ARM-linux:      test_basic:

; ARM-linux:      push    {r4, r5}
; ARM-linux-NEXT: mrc     p15, #0, r4, c13, c0, #3
; ARM-linux-NEXT: mov     r5, sp
; ARM-linux-NEXT: ldr     r4, [r4, #4]
; ARM-linux-NEXT: cmp     r4, r5
; ARM-linux-NEXT: blo     .LBB0_2

; ARM-linux:      mov     r4, #16
; ARM-linux-NEXT: mov     r5, #0
; ARM-linux-NEXT: stmdb   sp!, {lr}
; ARM-linux-NEXT: bl      __morestack
; ARM-linux-NEXT: ldm     sp!, {lr}
; ARM-linux-NEXT: pop     {r4, r5}
; ARM-linux-NEXT: bx      lr

; ARM-linux:      pop     {r4, r5}


; ARM-android:      test_basic:

; ARM-android:      push    {r4, r5}
; ARM-android-NEXT: mrc     p15, #0, r4, c13, c0, #3
; ARM-android-NEXT: mov     r5, sp
; ARM-android-NEXT: ldr     r4, [r4, #252]
; ARM-android-NEXT: cmp     r4, r5
; ARM-android-NEXT: blo     .LBB0_2

; ARM-android:      mov     r4, #16
; ARM-android-NEXT: mov     r5, #0
; ARM-android-NEXT: stmdb   sp!, {lr}
; ARM-android-NEXT: bl      __morestack
; ARM-android-NEXT: ldm     sp!, {lr}
; ARM-android-NEXT: pop     {r4, r5}
; ARM-android-NEXT: bx      lr

; ARM-android:      pop     {r4, r5}

}

attributes #0 = { "split-stack" }
