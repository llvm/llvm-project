; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-linux-gnu -no-integrated-as | FileCheck %s

; PR1351

define i32 @test1(i32 %Y, i32 %X) nounwind {
; CHECK: foo 3, 4
	%tmp1 = tail call i32 asm "foo${1:I} $0, $1", "=r,rI"( i32 %X )
	ret i32 %tmp1
}

define i32 @test2(i32 %Y, i32 %X) nounwind {
; CHECK: bari 3, 47
	%tmp1 = tail call i32 asm "bar${1:I} $0, $1", "=r,rI"( i32 47 )
	ret i32 %tmp1
}
