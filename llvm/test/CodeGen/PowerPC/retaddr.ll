; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-linux-gnu | FileCheck %s -check-prefix=PPC32
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu | FileCheck %s -check-prefix=PPC64

; PPC32: foo
; PPC32: mflr 0
; PPC32: stw 0, 4(1)
; PPC32: stwu 1, -[[STACK:[0-9]+]](1)
; PPC32: lwz [[REG:[0-9]+]], [[RETADDR:[0-9]+]](1)
; PPC32: stw [[REG]], 0(3)
; PPC32: lwz 0, [[RETADDR]](1)
; PPC32: addi 1, 1, [[STACK]]
; PPC32: mtlr 0
; PPC32: blr

; PPC64: foo
; PPC64: mflr 0
; PPC64: std 0, [[RETADDR:[0-9]+]]
; PPC64: stdu 1, -[[STACK:[0-9]+]]
; PPC64: ld [[REG:[0-9]+]]
; PPC64: std [[REG]], 0(3)
; PPC64: addi 1, 1, [[STACK]]
; PPC64: ld 0, [[RETADDR]]
; PPC64: mtlr 0
; PPC64: blr

define void @foo(i8** %X) nounwind {
entry:
	%tmp = tail call i8* @llvm.returnaddress( i32 0 )		; <i8*> [#uses=1]
	store i8* %tmp, i8** %X, align 4
	ret void
}

declare i8* @llvm.returnaddress(i32)

