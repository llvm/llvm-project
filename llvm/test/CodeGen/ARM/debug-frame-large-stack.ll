; RUN: llc -filetype=asm -o - < %s -mtriple arm-arm-netbsd-eabi -frame-pointer=all| FileCheck %s --check-prefix=CHECK-ARM
; RUN: llc -filetype=asm -o - < %s -mtriple arm-arm-netbsd-eabi | FileCheck %s --check-prefix=CHECK-ARM-FP-ELIM

define void @test1() {
    %tmp = alloca [ 64 x i32 ] , align 4
    ret void
}

; CHECK-ARM-LABEL: test1:
; CHECK-ARM: .cfi_startproc
; CHECK-ARM: sub    sp, sp, #256
; CHECK-ARM: .cfi_endproc

; CHECK-ARM-FP-ELIM-LABEL: test1:
; CHECK-ARM-FP-ELIM: .cfi_startproc
; CHECK-ARM-FP-ELIM: sub    sp, sp, #256
; CHECK-ARM-FP-ELIM: .cfi_endproc

define void @test2() {
    %tmp = alloca [ 4168 x i8 ] , align 4
    ret void
}

; CHECK-ARM-LABEL: test2:
; CHECK-ARM: .cfi_startproc
; CHECK-ARM: push    {r4, r5, r11, lr}
; CHECK-ARM: .cfi_def_cfa_offset 16
; CHECK-ARM: .cfi_offset lr, -4
; CHECK-ARM: .cfi_offset r11, -8
; CHECK-ARM: .cfi_offset r5, -12
; CHECK-ARM: .cfi_offset r4, -16
; CHECK-ARM: add    r11, sp, #8
; CHECK-ARM: .cfi_def_cfa r11, 8
; CHECK-ARM: sub    sp, sp, #72
; CHECK-ARM: sub    sp, sp, #4096
; CHECK-ARM: .cfi_endproc

; FIXME: Misspelled CHECK-ARM-FP-ELIM
; CHECK-ARM-FP_ELIM-LABEL: test2:
; CHECK-ARM-FP_ELIM: .cfi_startproc
; CHECK-ARM-FP_ELIM: push    {r4, r5}
; CHECK-ARM-FP_ELIM: .cfi_def_cfa_offset 8
; CHECK-ARM-FP_ELIM: .cfi_offset 54, -4
; CHECK-ARM-FP_ELIM: .cfi_offset r4, -8
; CHECK-ARM-FP_ELIM: sub    sp, sp, #72
; CHECK-ARM-FP_ELIM: sub    sp, sp, #4096
; CHECK-ARM-FP_ELIM: .cfi_def_cfa_offset 4176
; CHECK-ARM-FP_ELIM: .cfi_endproc

define i32 @test3() {
	%retval = alloca i32, align 4
	%tmp = alloca i32, align 4
	%a = alloca [805306369 x i8], align 16
	store i32 0, i32* %tmp
	%tmp1 = load i32, i32* %tmp
        ret i32 %tmp1
}

; CHECK-ARM-LABEL: test3:
; CHECK-ARM: .cfi_startproc
; CHECK-ARM: push    {r4, r5, r11, lr}
; CHECK-ARM: .cfi_def_cfa_offset 16
; CHECK-ARM: .cfi_offset lr, -4
; CHECK-ARM: .cfi_offset r11, -8
; CHECK-ARM: .cfi_offset r5, -12
; CHECK-ARM: .cfi_offset r4, -16
; CHECK-ARM: add    r11, sp, #8
; CHECK-ARM: .cfi_def_cfa r11, 8
; CHECK-ARM: sub    sp, sp, #16
; CHECK-ARM: sub    sp, sp, #805306368
; CHECK-ARM: bic    sp, sp, #15
; CHECK-ARM: .cfi_endproc

; CHECK-ARM-FP-ELIM-LABEL: test3:
; CHECK-ARM-FP-ELIM: .cfi_startproc
; CHECK-ARM-FP-ELIM: push    {r4, r5, r11}
; CHECK-ARM-FP-ELIM: .cfi_def_cfa_offset 12
; CHECK-ARM-FP-ELIM: .cfi_offset r11, -4
; CHECK-ARM-FP-ELIM: .cfi_offset r5, -8
; CHECK-ARM-FP-ELIM: .cfi_offset r4, -12
; CHECK-ARM-FP-ELIM: add    r11, sp, #8
; CHECK-ARM-FP-ELIM: .cfi_def_cfa r11, 4
; CHECK-ARM-FP-ELIM: sub    sp, sp, #20
; CHECK-ARM-FP-ELIM: sub    sp, sp, #805306368
; CHECK-ARM-FP-ELIM: bic    sp, sp, #15
; CHECK-ARM-FP-ELIM: .cfi_endproc

