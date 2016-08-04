; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-apple-darwin | FileCheck -check-prefix=P32 %s
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-apple-darwin | FileCheck -check-prefix=P64 %s

; PR8327
define i8* @test1(i8** %foo) nounwind {
  %A = va_arg i8** %foo, i8*
  ret i8* %A
}

; P32-LABEL: test1:
; P32: lwz r2, 0(r3)
; P32: addi r4, r2, 4
; P32: stw r4, 0(r3)
; P32: lwz r3, 0(r2)
; P32: blr 

; P64-LABEL: test1:
; P64: ld r2, 0(r3)
; P64: addi r4, r2, 8
; P64: std r4, 0(r3)
; P64: ld r3, 0(r2)
; P64: blr 

