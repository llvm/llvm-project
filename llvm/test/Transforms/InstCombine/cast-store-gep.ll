; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep inttoptr
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep alloca

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-apple-darwin10.0"
	%T = type { i8*, i8 }

define i8* @test(i8* %Val, i64 %V) nounwind {
entry:
	%A = alloca %T, align 8	
	%mrv_gep = bitcast %T* %A to i64*		; <i64*> [#uses=1]
	%B = getelementptr %T* %A, i64 0, i32 0		; <i8**> [#uses=1]
        
      	store i64 %V, i64* %mrv_gep
	%C = load i8** %B, align 8		; <i8*> [#uses=1]
	ret i8* %C
}
