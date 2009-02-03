; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep {GLOBAL.*align 16}
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep {tmp = load}

@GLOBAL = internal global [4 x i32] zeroinitializer

declare <16 x i8> @llvm.x86.sse2.loadu.dq(i8*)


define <16 x i8> @foo(<2 x i64> %x) {
entry:
	%tmp = tail call <16 x i8> @llvm.x86.sse2.loadu.dq( i8* bitcast ([4 x i32]* @GLOBAL to i8*) )
	ret <16 x i8> %tmp
}

