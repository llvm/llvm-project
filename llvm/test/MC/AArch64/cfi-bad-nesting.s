; RUN: not llvm-mc -triple arm64-apple-darwin %s -filetype=obj -o /dev/null 2>&1 | FileCheck %s

; REQUIRES: aarch64-registered-target

	.section	__TEXT,locomotive,regular,pure_instructions

	.globl	_locomotive
	.p2align	2
_locomotive:
	.cfi_startproc
	ret

	; It is invalid to have a non-private label between .cfi_startproc / .cfi_endproc
	.section	__TEXT,__text,regular,pure_instructions
	.globl	_caboose
	.p2align	2
_caboose:
; CHECK: [[#@LINE-1]]:1: error: non-private labels cannot appear between .cfi_startproc / .cfi_endproc pairs
; CHECK: [[#@LINE-9]]:2: error: previous .cfi_startproc was here
	ret
	.cfi_endproc

.subsections_via_symbols