; RUN: llc -verify-machineinstrs -ppc-asm-full-reg-names < %s -mtriple=powerpc-unknown-linux-gnu | FileCheck -check-prefix=P32 %s
; RUN: llc -verify-machineinstrs -ppc-asm-full-reg-names < %s -mtriple=powerpc64-unknown-linux-gnu | FileCheck -check-prefix=P64 %s
; RUN: llc -verify-machineinstrs -ppc-asm-full-reg-names < %s -mtriple=powerpc64le-unknown-linux-gnu | FileCheck -check-prefix=P64 %s

; PR8327
define i8* @test1(i8** %foo) nounwind {
  %A = va_arg i8** %foo, i8*
  ret i8* %A
}

; P32-LABEL: test1:
; P32:  lbz [[REG1:r[0-9]+]], 0(r3)
; P32:  addi [[REG2:r[0-9]+]], [[REG1]], 1
;	P32:  stb [[REG2]], 0(r3)
;	P32:  cmpwi	[[REG1]], 8
;	P32:  lwz [[REG3:r[0-9]+]], 4(r3)
;	P32:  slwi [[REG4:r[0-9]+]], [[REG1]], 2
;	P32:  addi [[REG5:r[0-9]+]], [[REG3]], 4
;	P32:  bc 12, lt, .LBB0_1
;	P32:  b .LBB0_2
; P32:  .LBB0_1:
; P32:  addi [[REG5]], [[REG3]], 0
;	P32: .LBB0_2:
;	P32:  stw [[REG5]], 4(r3)
;	P32:  lwz [[REG6:r[0-9]+]], 8(r3)
;	P32:  add [[REG6]], [[REG6]], [[REG4]]
; P32:  bc 12, lt, .LBB0_4
;	P32: # %bb.3:
;	P32:  ori [[REG6]], [[REG2]], 0
;	P32:  b .LBB0_4
;	P32: .LBB0_4:
;	P32:  lwz r3, 0([[REG6]])
;	P32:  blr

; P64-LABEL: test1:
; P64: ld [[REG1:r[0-9]+]], 0(r3)
; P64: addi [[REG2:r[0-9]+]], [[REG1]], 8
; P64: std [[REG2]], 0(r3)
; P64: ld r3, 0([[REG1]])
; P64: blr 

