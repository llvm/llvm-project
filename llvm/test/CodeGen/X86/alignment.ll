; RUN: llc %s -o - -mtriple=x86_64-linux-gnu | FileCheck %s

; This cannot get rounded up to the preferred alignment (16) if they have an
; explicit alignment specified.
@GlobalA = global { [384 x i8] } zeroinitializer, align 8 

; CHECK:	.bss
; CHECK:	.globl	GlobalA
; CHECK:	.align	8
; CHECK: GlobalA:
; CHECK:	.zero	384

; Common variables should not get rounded up to the preferred alignment (16) if
; they have an explicit alignment specified.
; PR6921
@GlobalB = common global { [384 x i8] } zeroinitializer, align 8

; CHECK: 	.comm	GlobalB,384,8


@GlobalC = common global { [384 x i8] } zeroinitializer, align 2

; CHECK: 	.comm	GlobalC,384,2



; This cannot get rounded up to the preferred alignment (16) if they have an
; explicit alignment specified *and* a section specified.
@GlobalAS = global { [384 x i8] } zeroinitializer, align 8, section "foo"

; CHECK:	.globl	GlobalAS
; CHECK:	.align	8
; CHECK: GlobalAS:
; CHECK:	.zero	384

; Common variables should not get rounded up to the preferred alignment (16) if
; they have an explicit alignment specified and a section specified.
; PR6921
@GlobalBS = common global { [384 x i8] } zeroinitializer, align 8, section "foo"
; CHECK: 	.comm	GlobalBS,384,8

@GlobalCS = common global { [384 x i8] } zeroinitializer, align 2, section "foo"
; CHECK: 	.comm	GlobalCS,384,2
