// RUN: llvm-mc -triple=arm64-apple-ios -filetype=obj < %s | \
// RUN: llvm-objdump --macho --unwind-info - | \
// RUN: FileCheck %s
//
// rdar://13070556

// CHECK:      Contents of __compact_unwind section:
// CHECK-NEXT:   Entry at offset 0x0:
// CHECK-NEXT:     start:                0x0 ltmp0
// CHECK-NEXT:     length:               0x8
// CHECK-NEXT:     compact encoding:     0x02000000
// CHECK-NEXT:   Entry at offset 0x20:
// CHECK-NEXT:     start:                0x8 _foo2
// CHECK-NEXT:     length:               0x40
// CHECK-NEXT:     compact encoding:     0x02009000
// CHECK-NEXT:   Entry at offset 0x40:
// CHECK-NEXT:     start:                0x48 _foo3
// CHECK-NEXT:     length:               0xd4
// CHECK-NEXT:     compact encoding:     0x0200400f
// CHECK-NEXT:   Entry at offset 0x60:
// CHECK-NEXT:     start:                0x11c _foo4
// CHECK-NEXT:     length:               0x54
// CHECK-NEXT:     compact encoding:     0x02021010

	.section	__TEXT,__text,regular,pure_instructions
	.globl	_foo1
	.align	2
_foo1:                                  ; @foo1
	.cfi_startproc
; %bb.0:                                ; %entry
	add	w0, w0, #42             ; =#42
	ret
	.cfi_endproc

	.globl	_foo2
	.align	2
_foo2:                                  ; @foo2
	.cfi_startproc
; %bb.0:                                ; %entry
	sub	sp, sp, #144            ; =#144
Ltmp2:
	.cfi_def_cfa_offset 144
	mov	x9, xzr
	mov	x8, sp
LBB1_1:                                 ; %for.body
                                        ; =>This Inner Loop Header: Depth=1
	str	w9, [x8, x9, lsl #2]
	add	x9, x9, #1              ; =#1
	cmp	w9, #36                 ; =#36
	b.ne	LBB1_1
; %bb.2:
	mov	x9, xzr
	mov	w0, wzr
LBB1_3:                                 ; %for.body4
                                        ; =>This Inner Loop Header: Depth=1
	ldr	w10, [x8, x9]
	add	x9, x9, #4              ; =#4
	cmp	w9, #144                ; =#144
	add	w0, w10, w0
	b.ne	LBB1_3
; %bb.4:                                ; %for.end9
	add	sp, sp, #144            ; =#144
	ret
	.cfi_endproc

	.globl	_foo3
	.align	2
_foo3:                                  ; @foo3
	.cfi_startproc
; %bb.0:                                ; %entry
	stp	x26, x25, [sp, #-64]!
	stp	x24, x23, [sp, #16]
	stp	x22, x21, [sp, #32]
	stp	x20, x19, [sp, #48]
Ltmp3:
	.cfi_def_cfa_offset 64
Ltmp4:
	.cfi_offset w19, -16
Ltmp5:
	.cfi_offset w20, -24
Ltmp6:
	.cfi_offset w21, -32
Ltmp7:
	.cfi_offset w22, -40
Ltmp8:
	.cfi_offset w23, -48
Ltmp9:
	.cfi_offset w24, -56
Ltmp10:
	.cfi_offset w25, -64
Ltmp11:
	.cfi_offset w26, -72
Lloh0:
	adrp	x8, _bar@GOTPAGE
Lloh1:
	ldr	x8, [x8, _bar@GOTPAGEOFF]
	ldr	w9, [x8]
	ldr	w10, [x8]
	ldr	w11, [x8]
	ldr	w12, [x8]
	ldr	w13, [x8]
	ldr	w14, [x8]
	ldr	w15, [x8]
	ldr	w16, [x8]
	ldr	w17, [x8]
	ldr	w0, [x8]
	ldr	w19, [x8]
	ldr	w20, [x8]
	ldr	w21, [x8]
	ldr	w22, [x8]
	ldr	w23, [x8]
	ldr	w24, [x8]
	ldr	w25, [x8]
	ldr	w8, [x8]
	add	w9, w10, w9
	add	w9, w9, w11
	add	w9, w9, w12
	add	w9, w9, w13
	add	w9, w9, w14
	add	w9, w9, w15
	add	w9, w9, w16
	add	w9, w9, w17
	add	w9, w9, w0
	add	w9, w9, w19
	add	w9, w9, w20
	add	w9, w9, w21
	add	w9, w9, w22
	add	w9, w9, w23
	add	w9, w9, w24
	add	w9, w9, w25
	sub	w8, w8, w9
	sub	w8, w8, w7, lsl #1
	sub	w8, w8, w6, lsl #1
	sub	w8, w8, w5, lsl #1
	sub	w8, w8, w4, lsl #1
	sub	w8, w8, w3, lsl #1
	sub	w8, w8, w2, lsl #1
	sub	w0, w8, w1, lsl #1
	ldp	x20, x19, [sp, #48]
	ldp	x22, x21, [sp, #32]
	ldp	x24, x23, [sp, #16]
	ldp	x26, x25, [sp], #64
	ret
	.loh AdrpLdrGot	Lloh0, Lloh1
	.cfi_endproc

	.globl	_foo4
	.align	2
_foo4:                                  ; @foo4
	.cfi_startproc
; %bb.0:                                ; %entry
	stp	x28, x27, [sp, #-16]!
	sub	sp, sp, #512            ; =#512
Ltmp12:
	.cfi_def_cfa_offset 528
Ltmp13:
	.cfi_offset w27, -16
Ltmp14:
	.cfi_offset w28, -24
                                        ; kill: def W0 killed W0 def X0
	mov	x9, xzr
	ubfx	x10, x0, #0, #32
	mov	x8, sp
LBB3_1:                                 ; %for.body
                                        ; =>This Inner Loop Header: Depth=1
	add	w11, w10, w9
	str	w11, [x8, x9, lsl #2]
	add	x9, x9, #1              ; =#1
	cmp	w9, #128                ; =#128
	b.ne	LBB3_1
; %bb.2:                                ; %for.cond2.preheader
	mov	x9, xzr
	mov	w0, wzr
	add	x8, x8, w5, sxtw #2
LBB3_3:                                 ; %for.body4
                                        ; =>This Inner Loop Header: Depth=1
	ldr	w10, [x8, x9]
	add	x9, x9, #4              ; =#4
	cmp	w9, #512                ; =#512
	add	w0, w10, w0
	b.ne	LBB3_3
; %bb.4:                                ; %for.end11
	add	sp, sp, #512            ; =#512
	ldp	x28, x27, [sp], #16
	ret
	.cfi_endproc

	.comm	_bar,4,2                ; @bar

.subsections_via_symbols
