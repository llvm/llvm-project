; RUN: llvm-as < %s | opt -simplify-libcalls | llvm-dis | grep {llvm.memset}
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i686-pc-linux-gnu"

define i8* @test(i8* %a, i32 %b, i32 %x) {
entry:
	%call = call i8* @memset(i8* %a, i32 %b, i32 %x )
	ret i8* %call
}

declare i8* @memset(i8*,i32,i32)

