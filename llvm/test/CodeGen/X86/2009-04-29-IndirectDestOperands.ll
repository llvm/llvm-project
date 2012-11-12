; RUN: llc < %s | grep "movl.*%ebx, 8(%esi)"
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9.0"

define void @cpuid(i32* %data) nounwind {
entry:
	%arrayidx = getelementptr i32* %data, i32 1		; <i32*> [#uses=1]
	%arrayidx2 = getelementptr i32* %data, i32 2		; <i32*> [#uses=1]
	%arrayidx4 = getelementptr i32* %data, i32 3		; <i32*> [#uses=1]
	%arrayidx6 = getelementptr i32* %data, i32 4		; <i32*> [#uses=1]
	%arrayidx8 = getelementptr i32* %data, i32 5		; <i32*> [#uses=1]
	%tmp9 = load i32* %arrayidx8		; <i32> [#uses=1]
	%arrayidx11 = getelementptr i32* %data, i32 6		; <i32*> [#uses=1]
	%tmp12 = load i32* %arrayidx11		; <i32> [#uses=1]
	%arrayidx14 = getelementptr i32* %data, i32 7		; <i32*> [#uses=1]
	%tmp15 = load i32* %arrayidx14		; <i32> [#uses=1]
	%arrayidx17 = getelementptr i32* %data, i32 8		; <i32*> [#uses=1]
	%tmp18 = load i32* %arrayidx17		; <i32> [#uses=1]
	%0 = call i32 asm "cpuid", "={ax},=*{bx},=*{cx},=*{dx},{ax},{bx},{cx},{dx},~{dirflag},~{fpsr},~{flags}"(i32* %arrayidx2, i32* %arrayidx4, i32* %arrayidx6, i32 %tmp9, i32 %tmp12, i32 %tmp15, i32 %tmp18) nounwind		; <i32> [#uses=1]
	store i32 %0, i32* %arrayidx
	ret void
}
