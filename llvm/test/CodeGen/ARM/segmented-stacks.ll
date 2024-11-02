; RUN: llc < %s -mtriple=arm-linux-androideabi -mattr=+v4t -verify-machineinstrs | FileCheck %s -check-prefix=ARM-android
; RUN: llc < %s -mtriple=arm-linux-unknown-gnueabi -mattr=+v4t  -verify-machineinstrs | FileCheck %s -check-prefix=ARM-linux

; We used to crash with filetype=obj
; RUN: llc < %s -mtriple=arm-linux-androideabi -filetype=obj
; RUN: llc < %s -mtriple=arm-linux-unknown-gnueabi -filetype=obj


; Just to prevent the alloca from being optimized away
declare void @dummy_use(i32*, i32)

define void @test_basic() #0 {
        %mem = alloca i32, i32 10
        call void @dummy_use (i32* %mem, i32 10)
	ret void

; ARM-linux-LABEL: test_basic:

; ARM-linux:      push    {r4, r5}
; ARM-linux-NEXT: mrc     p15, #0, r4, c13, c0, #3
; ARM-linux-NEXT: mov     r5, sp
; ARM-linux-NEXT: ldr     r4, [r4, #4]
; ARM-linux-NEXT: cmp     r4, r5
; ARM-linux-NEXT: blo     .LBB0_2

; ARM-linux:      mov     r4, #48
; ARM-linux-NEXT: mov     r5, #0
; ARM-linux-NEXT: stmdb   sp!, {lr}
; ARM-linux-NEXT: bl      __morestack
; ARM-linux-NEXT: ldm     sp!, {lr}
; ARM-linux-NEXT: pop     {r4, r5}
; ARM-linux-NEXT: bx      lr

; ARM-linux:      pop     {r4, r5}

; ARM-android-LABEL: test_basic:

; ARM-android:      push    {r4, r5}
; ARM-android-NEXT: mrc     p15, #0, r4, c13, c0, #3
; ARM-android-NEXT: mov     r5, sp
; ARM-android-NEXT: ldr     r4, [r4, #252]
; ARM-android-NEXT: cmp     r4, r5
; ARM-android-NEXT: blo     .LBB0_2

; ARM-android:      mov     r4, #48
; ARM-android-NEXT: mov     r5, #0
; ARM-android-NEXT: stmdb   sp!, {lr}
; ARM-android-NEXT: bl      __morestack
; ARM-android-NEXT: ldm     sp!, {lr}
; ARM-android-NEXT: pop     {r4, r5}
; ARM-android-NEXT: bx      lr

; ARM-android:      pop     {r4, r5}

}

define i32 @test_nested(i32 * nest %closure, i32 %other) #0 {
       %addend = load i32 , i32 * %closure
       %result = add i32 %other, %addend
       %mem = alloca i32, i32 10
       call void @dummy_use (i32* %mem, i32 10)
       ret i32 %result

; ARM-linux-LABEL: test_nested:

; ARM-linux:      push    {r4, r5}
; ARM-linux-NEXT: mrc     p15, #0, r4, c13, c0, #3
; ARM-linux-NEXT: mov     r5, sp
; ARM-linux-NEXT: ldr     r4, [r4, #4]
; ARM-linux-NEXT: cmp     r4, r5
; ARM-linux-NEXT: blo     .LBB1_2

; ARM-linux:      mov     r4, #56
; ARM-linux-NEXT: mov     r5, #0
; ARM-linux-NEXT: stmdb   sp!, {lr}
; ARM-linux-NEXT: bl      __morestack
; ARM-linux-NEXT: ldm     sp!, {lr}
; ARM-linux-NEXT: pop     {r4, r5}
; ARM-linux-NEXT: bx      lr

; ARM-linux:      pop     {r4, r5}

; ARM-android-LABEL: test_nested:

; ARM-android:      push    {r4, r5}
; ARM-android-NEXT: mrc     p15, #0, r4, c13, c0, #3
; ARM-android-NEXT: mov     r5, sp
; ARM-android-NEXT: ldr     r4, [r4, #252]
; ARM-android-NEXT: cmp     r4, r5
; ARM-android-NEXT: blo     .LBB1_2

; ARM-android:      mov     r4, #56
; ARM-android-NEXT: mov     r5, #0
; ARM-android-NEXT: stmdb   sp!, {lr}
; ARM-android-NEXT: bl      __morestack
; ARM-android-NEXT: ldm     sp!, {lr}
; ARM-android-NEXT: pop     {r4, r5}
; ARM-android-NEXT: bx      lr

; ARM-android:      pop     {r4, r5}

}

define void @test_large() #0 {
        %mem = alloca i32, i32 10000
        call void @dummy_use (i32* %mem, i32 0)
        ret void

; ARM-linux-LABEL: test_large:

; ARM-linux:      push    {r4, r5}
; ARM-linux-NEXT: ldr     r4, .LCPI2_0
; ARM-linux-NEXT: sub     r5, sp, r4
; ARM-linux-NEXT: mrc     p15, #0, r4, c13, c0, #3
; ARM-linux-NEXT: ldr     r4, [r4, #4]
; ARM-linux-NEXT: cmp     r4, r5
; ARM-linux-NEXT: blo     .LBB2_2

; ARM-linux:      ldr     r4, .LCPI2_0
; ARM-linux-NEXT: mov     r5, #0
; ARM-linux-NEXT: stmdb   sp!, {lr}
; ARM-linux-NEXT: bl      __morestack
; ARM-linux-NEXT: ldm     sp!, {lr}
; ARM-linux-NEXT: pop     {r4, r5}
; ARM-linux-NEXT: bx      lr

; ARM-linux:      pop     {r4, r5}

; ARM-linux:      .LCPI2_0:
; ARM-linux-NEXT: .long   40192

; ARM-android-LABEL: test_large:

; ARM-android:      push    {r4, r5}
; ARM-android-NEXT: ldr     r4, .LCPI2_0
; ARM-android-NEXT: sub     r5, sp, r4
; ARM-android-NEXT: mrc     p15, #0, r4, c13, c0, #3
; ARM-android-NEXT: ldr     r4, [r4, #252]
; ARM-android-NEXT: cmp     r4, r5
; ARM-android-NEXT: blo     .LBB2_2

; ARM-android:      ldr     r4, .LCPI2_0
; ARM-android-NEXT: mov     r5, #0
; ARM-android-NEXT: stmdb   sp!, {lr}
; ARM-android-NEXT: bl      __morestack
; ARM-android-NEXT: ldm     sp!, {lr}
; ARM-android-NEXT: pop     {r4, r5}
; ARM-android-NEXT: bx      lr

; ARM-android:      pop     {r4, r5}

; ARM-android:      .LCPI2_0:
; ARM-android-NEXT: .long   40192

}

define fastcc void @test_fastcc() #0 {
        %mem = alloca i32, i32 10
        call void @dummy_use (i32* %mem, i32 10)
        ret void

; ARM-linux-LABEL: test_fastcc:

; ARM-linux:      push    {r4, r5}
; ARM-linux-NEXT: mrc     p15, #0, r4, c13, c0, #3
; ARM-linux-NEXT: mov     r5, sp
; ARM-linux-NEXT: ldr     r4, [r4, #4]
; ARM-linux-NEXT: cmp     r4, r5
; ARM-linux-NEXT: blo     .LBB3_2

; ARM-linux:      mov     r4, #48
; ARM-linux-NEXT: mov     r5, #0
; ARM-linux-NEXT: stmdb   sp!, {lr}
; ARM-linux-NEXT: bl      __morestack
; ARM-linux-NEXT: ldm     sp!, {lr}
; ARM-linux-NEXT: pop     {r4, r5}
; ARM-linux-NEXT: bx      lr

; ARM-linux:      pop     {r4, r5}

; ARM-android-LABEL: test_fastcc:

; ARM-android:      push    {r4, r5}
; ARM-android-NEXT: mrc     p15, #0, r4, c13, c0, #3
; ARM-android-NEXT: mov     r5, sp
; ARM-android-NEXT: ldr     r4, [r4, #252]
; ARM-android-NEXT: cmp     r4, r5
; ARM-android-NEXT: blo     .LBB3_2

; ARM-android:      mov     r4, #48
; ARM-android-NEXT: mov     r5, #0
; ARM-android-NEXT: stmdb   sp!, {lr}
; ARM-android-NEXT: bl      __morestack
; ARM-android-NEXT: ldm     sp!, {lr}
; ARM-android-NEXT: pop     {r4, r5}
; ARM-android-NEXT: bx      lr

; ARM-android:      pop     {r4, r5}

}

define fastcc void @test_fastcc_large() #0 {
        %mem = alloca i32, i32 10000
        call void @dummy_use (i32* %mem, i32 0)
        ret void

; ARM-linux-LABEL: test_fastcc_large:

; ARM-linux:      push    {r4, r5}
; ARM-linux-NEXT: ldr     r4, .LCPI4_0
; ARM-linux-NEXT: sub     r5, sp, r4
; ARM-linux-NEXT: mrc     p15, #0, r4, c13, c0, #3
; ARM-linux-NEXT: ldr     r4, [r4, #4]
; ARM-linux-NEXT: cmp     r4, r5
; ARM-linux-NEXT: blo     .LBB4_2

; ARM-linux:      ldr     r4, .LCPI4_0
; ARM-linux-NEXT: mov     r5, #0
; ARM-linux-NEXT: stmdb   sp!, {lr}
; ARM-linux-NEXT: bl      __morestack
; ARM-linux-NEXT: ldm     sp!, {lr}
; ARM-linux-NEXT: pop     {r4, r5}
; ARM-linux-NEXT: bx      lr

; ARM-linux:      pop     {r4, r5}

; ARM-linux:      .LCPI4_0:
; ARM-linux-NEXT: .long   40192

; ARM-android-LABEL: test_fastcc_large:

; ARM-android:      push    {r4, r5}
; ARM-android-NEXT: ldr     r4, .LCPI4_0
; ARM-android-NEXT: sub     r5, sp, r4
; ARM-android-NEXT: mrc     p15, #0, r4, c13, c0, #3
; ARM-android-NEXT: ldr     r4, [r4, #252]
; ARM-android-NEXT: cmp     r4, r5
; ARM-android-NEXT: blo     .LBB4_2

; ARM-android:      ldr     r4, .LCPI4_0
; ARM-android-NEXT: mov     r5, #0
; ARM-android-NEXT: stmdb   sp!, {lr}
; ARM-android-NEXT: bl      __morestack
; ARM-android-NEXT: ldm     sp!, {lr}
; ARM-android-NEXT: pop     {r4, r5}
; ARM-android-NEXT: bx      lr

; ARM-android:      pop     {r4, r5}

; ARM-android:      .LCPI4_0:
; ARM-android-NEXT: .long   40192

}

define void @test_nostack() #0 {
	ret void

; ARM-linux-LABEL: test_nostack:
; ARM-linux-NOT:   bl __morestack

; ARM-android-LABEL: test_nostack:
; ARM-android-NOT:   bl __morestack
}

; Test to make sure that a morestack call is generated if there is a
; sibling call, even if the function in question has no stack frame
; (PR37807).

declare i32 @callee(i32)

define i32 @test_sibling_call_empty_frame(i32 %x) #0 {
  %call = tail call i32 @callee(i32 %x) #0
  ret i32 %call

; ARM-linux-LABEL: test_sibling_call_empty_frame:
; ARM-linux:      bl      __morestack

; ARM-android-LABEL: test_sibling_call_empty_frame:
; ARM-android:      bl      __morestack

}


declare void @panic() unnamed_addr

; We used to crash while compiling the following function.
; ARM-linux-LABEL: build_should_not_segfault:
; ARM-android-LABEL: build_should_not_segfault:
define void @build_should_not_segfault(i8 %x) unnamed_addr #0 {
start:
  %_0 = icmp ult i8 %x, 16
  %or.cond = select i1 undef, i1 true, i1 %_0
  br i1 %or.cond, label %bb1, label %bb2

bb1:
  ret void

bb2:
  call void @panic()
  unreachable
}

attributes #0 = { "split-stack" }
