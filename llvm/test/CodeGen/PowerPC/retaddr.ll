; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-apple-darwin8 | grep mflr
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-apple-darwin8 | grep lwz
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-apple-darwin8 | grep "ld r., 16(r1)"

define void @foo(i8** %X) nounwind {
entry:
	%tmp = tail call i8* @llvm.returnaddress( i32 0 )		; <i8*> [#uses=1]
	store i8* %tmp, i8** %X, align 4
	ret void
}

declare i8* @llvm.returnaddress(i32)

