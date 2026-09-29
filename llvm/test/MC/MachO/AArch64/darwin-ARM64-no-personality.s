# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos11.0 %s | llvm-objdump --unwind-info - | FileCheck %s

# Check that we emit the right encoding for the no-personality case.

# CHECK: Contents of __compact_unwind section:
# CHECK:  Entry at offset {{.+}}
# CHECK:    start:                {{.+}} ltmp0
# CHECK:    length:               {{.+}}
# CHECK:    compact encoding:     0x02001000
# CHECK:  Entry at offset {{.+}}
# CHECK:    start:                {{.+}} __Z3foov
# CHECK:    length:               {{.+}}
# CHECK:    compact encoding:     0x04000000
# CHECK:  Entry at offset {{.+}}
# CHECK:    start:                {{.+}} _main
# CHECK:    length:               {{.+}}
# CHECK:    compact encoding:     0x04000000


       	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 11, 0
	.globl	__Z3barPi                     
	.p2align	2
__Z3barPi:         
	.cfi_startproc
	sub	sp, sp, #16
	.cfi_def_cfa_offset 16
	str	x0, [sp, #8]
	add	sp, sp, #16
	ret
	.cfi_endproc

	.globl	__Z3foov           
	.p2align	2
__Z3foov:                             
	.cfi_startproc

	sub	sp, sp, #32
	.cfi_def_cfa_offset 32
	stp	x29, x30, [sp, #16]   
	add	x29, sp, #16
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	sub	x0, x29, #4
	bl	__Z3barPi
	ldp	x29, x30, [sp, #16]         
	add	sp, sp, #32
	ret
	.cfi_endproc
                                       
	.globl	_main                    
	.p2align	2
_main:                               
	.cfi_startproc
	sub	sp, sp, #32
	.cfi_def_cfa_offset 32
	stp	x29, x30, [sp, #16]          
	add	x29, sp, #16
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	mov	w8, #0                         
	str	w8, [sp, #8]                  
	stur	wzr, [x29, #-4]
	bl	__Z3foov
	ldr	w0, [sp, #8]                  
	ldp	x29, x30, [sp, #16]           
	add	sp, sp, #32
	ret
	.cfi_endproc

.subsections_via_symbols
