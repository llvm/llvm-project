; RUN: not llvm-mc -triple arm64-apple-darwin %s -filetype=obj -o /dev/null 2>&1 | FileCheck %s --check-prefix=DARWIN

; REQUIRES: aarch64-registered-target

	.section	__TEXT,locomotive,regular,pure_instructions

	.globl	_locomotive
	.p2align	2
_locomotive:
	.cfi_startproc
	; An N_ALT_ENTRY symbol can be defined in the middle of a subsection, so
	; these are opted out of the .cfi_{start,end}proc nesting check.
	.alt_entry _engineer
_engineer:
	ret

	; It is invalid to have a non-private label between .cfi_startproc and
	; .cfi_endproc on MachO platforms.
	.section	__TEXT,__text,regular,pure_instructions
	.globl	_caboose
	.p2align	2
_caboose:
; DARWIN: [[#@LINE-1]]:1: error: non-private labels cannot appear between .cfi_startproc / .cfi_endproc pairs
; DARWIN: [[#@LINE-14]]:2: error: previous .cfi_startproc was here
	ret
	.cfi_endproc

.subsections_via_symbols