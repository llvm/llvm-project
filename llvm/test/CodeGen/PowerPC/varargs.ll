; RUN: llc < %s -mtriple=powerpc-apple-darwin | FileCheck -check-prefix=P32 %s
; RUN: llc < %s -mtriple=powerpc64-apple-darwin | FileCheck -check-prefix=P64 %s

; PR8327
define i8* @test1(i8** %foo) nounwind {
  %A = va_arg i8** %foo, i8*
  ret i8* %A
}

; P32: test1:
; P32: 	lwz r4, 0(r3)
; P32:	addi r5, r4, 4
; P32:	stw r5, 0(r3)
; P32:	lwz r3, 0(r4)
; P32:	blr 

; P64: test1:
; P64: ld r4, 0(r3)
; P64: addi r5, r4, 8
; P64: std r5, 0(r3)
; P64: ld r3, 0(r4)
; P64: blr
