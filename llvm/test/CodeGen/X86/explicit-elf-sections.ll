; RUN: llc < %s -mtriple=x86_64-pc-linux -function-sections=1 -data-sections=1 | FileCheck %s -check-prefix=SECTIONS
; RUN: llc < %s -mtriple=x86_64-pc-linux -function-sections=0 -data-sections=0 | FileCheck %s -check-prefix=NSECTIONS

define void @f() section "aaa" { ret void }
define void @g() section "aaa" { ret void }

define void @h() { ret void }

@x = global i32 1, section "aaa"
@y = global i32 1, section "aaa"
@z = global i32 1

define void @i() section "aaa" { ret void }

@w = global i32 1, section "aaa"

; NSECTIONS:		.section	aaa,"ax",@progbits
; NSECTIONS-NOT:	{{\.section|\.text|\.data}}
; NSECTIONS:		f:
; NSECTIONS-NOT:	{{\.section|\.text|\.data}}
; NSECTIONS:		g:
; NSECTIONS:		.text
; NSECTIONS-NOT:	{{\.section|\.text|\.data}}
; NSECTIONS:		h:
; NSECTIONS:		.section	aaa,"ax",@progbits
; NSECTIONS-NOT:	{{\.section|\.text|\.data}}
; NSECTIONS:		i:
; NSECTIONS-NOT:	{{\.section|\.text|\.data}}
; NSECTIONS:		x:
; NSECTIONS-NOT:	{{\.section|\.text|\.data}}
; NSECTIONS:		y:
; NSECTIONS:		.data
; NSECTIONS-NOT:	{{\.section|\.text|\.data}}
; NSECTIONS:		z:
; NSECTIONS:		.section	aaa,"ax",@progbits
; NSECTIONS-NOT:	{{\.section|\.text|\.data}}
; NSECTIONS:		w:


; SECTIONS:	.section	aaa,"ax",@progbits,unique,1
; SECTIONS-NOT:	{{\.section|\.text|\.data}}
; SECTIONS:	f:
; SECTIONS:	.section	aaa,"ax",@progbits,unique,2
; SECTIONS-NOT:	{{\.section|\.text|\.data}}
; SECTIONS:	g:
; SECTIONS:	.section        .text.h,"ax",@progbits
; SECTIONS-NOT:	{{\.section|\.text|\.data}}
; SECTIONS:	h:
; SECTIONS:	.section	aaa,"ax",@progbits,unique,3
; SECTIONS-NOT:	{{\.section|\.text|\.data}}
; SECTIONS:	i:
; SECTIONS:	.section	aaa,"aw",@progbits,unique,4
; SECTIONS-NOT:	{{\.section|\.text|\.data}}
; SECTIONS:	x:
; SECTIONS:	.section	aaa,"aw",@progbits,unique,5
; SECTIONS-NOT:	{{\.section|\.text|\.data}}
; SECTIONS:	y:
; SECTIONS:	.section        .data.z,"aw",@progbits
; SECTIONS-NOT:	{{\.section|\.text|\.data}}
; SECTIONS:	z:
; SECTIONS:	.section	aaa,"aw",@progbits,unique,6
; SECTIONS-NOT:	{{\.section|\.text|\.data}}
; SECTIONS:	w:


