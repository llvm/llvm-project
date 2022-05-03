; RUN: llc < %s -mtriple=thumb-linux-androideabi -verify-machineinstrs | FileCheck %s -check-prefix=Thumb-android
; RUN: llc < %s -mtriple=thumb-linux-unknown-gnueabi -verify-machineinstrs | FileCheck %s -check-prefix=Thumb-linux
; RUN: llc < %s -mtriple=thumb-linux-androideabi -filetype=obj
; RUN: llc < %s -mtriple=thumb-linux-unknown-gnueabi -filetype=obj


; Just to prevent the alloca from being optimized away
declare void @dummy_use(i32*, i32)

define void @test_basic() #0 {
        %mem = alloca i32, i32 10
        call void @dummy_use (i32* %mem, i32 10)
	ret void

; Thumb-android-LABEL:      test_basic:

; Thumb-android:      push    {r4, r5}
; Thumb-android-NEXT: mov     r5, sp
; Thumb-android-NEXT: ldr     r4, .LCPI0_0
; Thumb-android-NEXT: ldr     r4, [r4]
; Thumb-android-NEXT: cmp     r4, r5
; Thumb-android-NEXT: blo     .LBB0_2

; Thumb-android:      mov     r4, #48
; Thumb-android-NEXT: mov     r5, #0
; Thumb-android-NEXT: push    {lr}
; Thumb-android-NEXT: bl      __morestack
; Thumb-android-NEXT: pop     {r4}
; Thumb-android-NEXT: mov     lr, r4
; Thumb-android-NEXT: pop     {r4, r5}
; Thumb-android-NEXT: bx      lr

; Thumb-android:      pop     {r4, r5}

; Thumb-android: .p2align 2
; Thumb-android: .LCPI0_0:
; Thumb-android-NEXT: .long __STACK_LIMIT

; Thumb-linux-LABEL:      test_basic:

; Thumb-linux:      push    {r4, r5}
; Thumb-linux-NEXT: mov     r5, sp
; Thumb-linux-NEXT: ldr     r4, .LCPI0_0
; Thumb-linux-NEXT: ldr     r4, [r4]
; Thumb-linux-NEXT: cmp     r4, r5
; Thumb-linux-NEXT: blo     .LBB0_2

; Thumb-linux:      mov     r4, #48
; Thumb-linux-NEXT: mov     r5, #0
; Thumb-linux-NEXT: push    {lr}
; Thumb-linux-NEXT: bl      __morestack
; Thumb-linux-NEXT: pop     {r4}
; Thumb-linux-NEXT: mov     lr, r4
; Thumb-linux-NEXT: pop     {r4, r5}
; Thumb-linux-NEXT: bx      lr

; Thumb-linux:      pop     {r4, r5}

}

define i32 @test_nested(i32 * nest %closure, i32 %other) #0 {
       %addend = load i32 , i32 * %closure
       %result = add i32 %other, %addend
       %mem = alloca i32, i32 10
       call void @dummy_use (i32* %mem, i32 10)
       ret i32 %result

; Thumb-android-LABEL:      test_nested:

; Thumb-android:      push  {r4, r5}
; Thumb-android-NEXT: mov     r5, sp
; Thumb-android-NEXT: ldr     r4, .LCPI1_0
; Thumb-android-NEXT: ldr     r4, [r4]
; Thumb-android-NEXT: cmp     r4, r5
; Thumb-android-NEXT: blo     .LBB1_2

; Thumb-android:      mov     r4, #56
; Thumb-android-NEXT: mov     r5, #0
; Thumb-android-NEXT: push    {lr}
; Thumb-android-NEXT: bl      __morestack
; Thumb-android-NEXT: pop     {r4}
; Thumb-android-NEXT: mov     lr, r4
; Thumb-android-NEXT: pop     {r4, r5}
; Thumb-android-NEXT: bx      lr

; Thumb-android:      pop     {r4, r5}

; Thumb-linux-LABEL:      test_nested:

; Thumb-linux:      push    {r4, r5}
; Thumb-linux-NEXT: mov     r5, sp
; Thumb-linux-NEXT: ldr     r4, .LCPI1_0
; Thumb-linux-NEXT: ldr     r4, [r4]
; Thumb-linux-NEXT: cmp     r4, r5
; Thumb-linux-NEXT: blo     .LBB1_2

; Thumb-linux:      mov     r4, #56
; Thumb-linux-NEXT: mov     r5, #0
; Thumb-linux-NEXT: push    {lr}
; Thumb-linux-NEXT: bl      __morestack
; Thumb-linux-NEXT: pop     {r4}
; Thumb-linux-NEXT: mov     lr, r4
; Thumb-linux-NEXT: pop     {r4, r5}
; Thumb-linux-NEXT: bx      lr

; Thumb-linux:      pop     {r4, r5}

}

define void @test_large() #0 {
        %mem = alloca i32, i32 10000
        call void @dummy_use (i32* %mem, i32 0)
        ret void

; Thumb-android-LABEL:      test_large:

; Thumb-android:      push    {r4, r5}
; Thumb-android-NEXT: mov     r5, sp
; Thumb-android-NEXT: ldr     r4, .LCPI2_2
; Thumb-android-NEXT: sub     r5, r5, r4
; Thumb-android-NEXT: ldr     r4, .LCPI2_3
; Thumb-android-NEXT: ldr     r4, [r4]
; Thumb-android-NEXT: cmp     r4, r5
; Thumb-android-NEXT: blo     .LBB2_2

; Thumb-android:      ldr     r4, .LCPI2_2
; Thumb-android-NEXT: mov     r5, #0
; Thumb-android-NEXT: push    {lr}
; Thumb-android-NEXT: bl      __morestack
; Thumb-android-NEXT: pop     {r4}
; Thumb-android-NEXT: mov     lr, r4
; Thumb-android-NEXT: pop     {r4, r5}
; Thumb-android-NEXT: bx      lr

; Thumb-android:      pop     {r4, r5}

; Thumb-android:      .LCPI2_2:
; Thumb-android-NEXT: .long   40192

; Thumb-linux-LABEL:      test_large:

; Thumb-linux:      push    {r4, r5}
; Thumb-linux-NEXT: mov     r5, sp
; Thumb-linux-NEXT: ldr     r4, .LCPI2_2
; Thumb-linux-NEXT: sub     r5, r5, r4
; Thumb-linux-NEXT: ldr     r4, .LCPI2_3
; Thumb-linux-NEXT: ldr     r4, [r4]
; Thumb-linux-NEXT: cmp     r4, r5
; Thumb-linux-NEXT: blo     .LBB2_2

; Thumb-linux:      ldr     r4, .LCPI2_2
; Thumb-linux-NEXT: mov     r5, #0
; Thumb-linux-NEXT: push    {lr}
; Thumb-linux-NEXT: bl      __morestack
; Thumb-linux-NEXT: pop     {r4}
; Thumb-linux-NEXT: mov     lr, r4
; Thumb-linux-NEXT: pop     {r4, r5}
; Thumb-linux-NEXT: bx      lr

; Thumb-linux:      pop     {r4, r5}

}

define fastcc void @test_fastcc() #0 {
        %mem = alloca i32, i32 10
        call void @dummy_use (i32* %mem, i32 10)
        ret void

; Thumb-android-LABEL:      test_fastcc:

; Thumb-android:      push    {r4, r5}
; Thumb-android-NEXT: mov     r5, sp
; Thumb-android-NEXT: ldr     r4, .LCPI3_0
; Thumb-android-NEXT: ldr     r4, [r4]
; Thumb-android-NEXT: cmp     r4, r5
; Thumb-android-NEXT: blo     .LBB3_2

; Thumb-android:      mov     r4, #48
; Thumb-android-NEXT: mov     r5, #0
; Thumb-android-NEXT: push    {lr}
; Thumb-android-NEXT: bl      __morestack
; Thumb-android-NEXT: pop     {r4}
; Thumb-android-NEXT: mov     lr, r4
; Thumb-android-NEXT: pop     {r4, r5}
; Thumb-android-NEXT: bx      lr

; Thumb-android:      pop     {r4, r5}

; Thumb-linux-LABEL:      test_fastcc:

; Thumb-linux:      push    {r4, r5}
; Thumb-linux-NEXT: mov     r5, sp
; Thumb-linux-NEXT: ldr     r4, .LCPI3_0
; Thumb-linux-NEXT: ldr     r4, [r4]
; Thumb-linux-NEXT: cmp     r4, r5
; Thumb-linux-NEXT: blo     .LBB3_2

; Thumb-linux:      mov     r4, #48
; Thumb-linux-NEXT: mov     r5, #0
; Thumb-linux-NEXT: push    {lr}
; Thumb-linux-NEXT: bl      __morestack
; Thumb-linux-NEXT: pop     {r4}
; Thumb-linux-NEXT: mov     lr, r4
; Thumb-linux-NEXT: pop     {r4, r5}
; Thumb-linux-NEXT: bx      lr

; Thumb-linux:      pop     {r4, r5}

}

define fastcc void @test_fastcc_large() #0 {
        %mem = alloca i32, i32 10000
        call void @dummy_use (i32* %mem, i32 0)
        ret void

; Thumb-android-LABEL:      test_fastcc_large:

; Thumb-android:      push    {r4, r5}
; Thumb-android-NEXT: mov     r5, sp
; Thumb-android-NEXT: ldr     r4, .LCPI4_2
; Thumb-android-NEXT: sub     r5, r5, r4
; Thumb-android-NEXT: ldr     r4, .LCPI4_3
; Thumb-android-NEXT: ldr     r4, [r4]
; Thumb-android-NEXT: cmp     r4, r5
; Thumb-android-NEXT: blo     .LBB4_2

; Thumb-android:      ldr     r4, .LCPI4_2
; Thumb-android-NEXT: mov     r5, #0
; Thumb-android-NEXT: push    {lr}
; Thumb-android-NEXT: bl      __morestack
; Thumb-android-NEXT: pop     {r4}
; Thumb-android-NEXT: mov     lr, r4
; Thumb-android-NEXT: pop     {r4, r5}
; Thumb-android-NEXT: bx      lr

; Thumb-android:      pop     {r4, r5}

; Thumb-android:      .LCPI4_2:
; Thumb-android-NEXT: .long   40192

; Thumb-linux-LABEL:      test_fastcc_large:

; Thumb-linux:      push    {r4, r5}
; Thumb-linux-NEXT: mov     r5, sp
; Thumb-linux-NEXT: ldr     r4, .LCPI4_2
; Thumb-linux-NEXT: sub     r5, r5, r4
; Thumb-linux-NEXT: ldr     r4, .LCPI4_3
; Thumb-linux-NEXT: ldr     r4, [r4]
; Thumb-linux-NEXT: cmp     r4, r5
; Thumb-linux-NEXT: blo     .LBB4_2

; Thumb-linux:      ldr     r4, .LCPI4_2
; Thumb-linux-NEXT: mov     r5, #0
; Thumb-linux-NEXT: push    {lr}
; Thumb-linux-NEXT: bl      __morestack
; Thumb-linux-NEXT: pop     {r4}
; Thumb-linux-NEXT: mov     lr, r4
; Thumb-linux-NEXT: pop     {r4, r5}
; Thumb-linux-NEXT: bx      lr

; Thumb-linux:      pop     {r4, r5}

; Thumb-linux:      .LCPI4_2:
; Thumb-linux-NEXT: .long   40192

}

define void @test_nostack() #0 {
	ret void

; Thumb-android-LABEL: test_nostack:
; Thumb-android-NOT:   bl __morestack

; Thumb-linux-LABEL: test_nostack:
; Thumb-linux-NOT:   bl __morestack
}


declare void @panic() unnamed_addr

; We used to crash while compiling the following function.
; Thumb-linux-LABEL: build_should_not_segfault:
; Thumb-android-LABEL: build_should_not_segfault:
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
