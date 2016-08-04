; RUN: llc -verify-machineinstrs < %s -march=ppc32 | grep mflr
; RUN: llc -verify-machineinstrs < %s -march=ppc32 | grep lwz
; RUN: llc -verify-machineinstrs < %s -march=ppc64 | grep "ld r., 16(r1)"

target triple = "powerpc-apple-darwin8"

define void @foo(i8** %X) nounwind {
entry:
	%tmp = tail call i8* @llvm.returnaddress( i32 0 )		; <i8*> [#uses=1]
	store i8* %tmp, i8** %X, align 4
	ret void
}

declare i8* @llvm.returnaddress(i32)

