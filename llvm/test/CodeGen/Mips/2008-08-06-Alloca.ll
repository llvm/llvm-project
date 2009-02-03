; RUN: llvm-as < %s | llc -march=mips | grep {subu.*sp} | count 2

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "mipsallegrexel-psp-elf"

define i32 @twoalloca(i32 %size) nounwind {
entry:
	alloca i8, i32 %size		; <i8*>:0 [#uses=1]
	alloca i8, i32 %size		; <i8*>:1 [#uses=1]
	call i32 @foo( i8* %0 ) nounwind		; <i32>:2 [#uses=1]
	call i32 @foo( i8* %1 ) nounwind		; <i32>:3 [#uses=1]
	add i32 %3, %2		; <i32>:4 [#uses=1]
	ret i32 %4
}

declare i32 @foo(i8*)
