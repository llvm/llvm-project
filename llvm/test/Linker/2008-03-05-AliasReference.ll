; PR2054
; RUN: llvm-as %s -o %t1.bc -f
; RUN: llvm-as %p/2008-03-05-AliasReference2.ll -o %t2.bc -f
; RUN: llvm-link %t2.bc %t1.bc -f -o %t3.bc

; ModuleID = 'bug.o'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"
@foo = weak global i32 0		; <i32*> [#uses=1]

@bar = alias weak i32* @foo		; <i32*> [#uses=1]

define i32 @baz() nounwind  {
entry:
	%tmp1 = load i32* @bar, align 4		; <i32> [#uses=1]
	ret i32 %tmp1
}
