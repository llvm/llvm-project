; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep true

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"
	%packed = type <{ x86_fp80, i8 }>
	%unpacked = type { x86_fp80, i8 }

define i1 @q() nounwind  {
entry:
	%char_p = getelementptr %packed* null, i32 0, i32 1		; <i8*> [#uses=1]
	%char_u = getelementptr %unpacked* null, i32 0, i32 1		; <i8*> [#uses=1]
	%res = icmp eq i8* %char_p, %char_u		; <i1> [#uses=1]
	ret i1 %res
}
