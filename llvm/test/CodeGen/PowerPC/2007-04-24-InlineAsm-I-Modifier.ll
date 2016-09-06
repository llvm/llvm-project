; RUN: llc -verify-machineinstrs < %s -march=ppc32 -mtriple=powerpc-apple-darwin8.8.0 -no-integrated-as | grep "foo r3, r4"
; RUN: llc -verify-machineinstrs < %s -march=ppc32 -mtriple=powerpc-apple-darwin8.8.0 -no-integrated-as | grep "bari r3, 47"

; PR1351

define i32 @test1(i32 %Y, i32 %X) nounwind {
	%tmp1 = tail call i32 asm "foo${1:I} $0, $1", "=r,rI"( i32 %X )
	ret i32 %tmp1
}

define i32 @test2(i32 %Y, i32 %X) nounwind {
	%tmp1 = tail call i32 asm "bar${1:I} $0, $1", "=r,rI"( i32 47 )
	ret i32 %tmp1
}
