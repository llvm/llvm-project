; RUN: opt < %s -aa-pipeline=basic-aa -passes=gvn,instcombine -S | FileCheck %s
; PR2436
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin8"

define i1 @foo(i32 %i) nounwind  {
; CHECK: ret i1 true
entry:
	%arr = alloca [10 x ptr]		; <ptr> [#uses=1]
	%tmp2 = call ptr @getPtr( ) nounwind 		; <ptr> [#uses=2]
	%tmp4 = getelementptr [10 x ptr], ptr %arr, i32 0, i32 %i		; <ptr> [#uses=2]
	store ptr %tmp2, ptr %tmp4, align 4
	%tmp10 = getelementptr i8, ptr %tmp2, i32 10		; <ptr> [#uses=1]
	store i8 42, ptr %tmp10, align 1
	%tmp14 = load ptr, ptr %tmp4, align 4		; <ptr> [#uses=1]
	%tmp16 = getelementptr i8, ptr %tmp14, i32 10		; <ptr> [#uses=1]
	%tmp17 = load i8, ptr %tmp16, align 1		; <i8> [#uses=1]
	%tmp19 = icmp eq i8 %tmp17, 42		; <i1> [#uses=1]
	ret i1 %tmp19
}

declare ptr @getPtr()

declare void @abort() noreturn nounwind 
