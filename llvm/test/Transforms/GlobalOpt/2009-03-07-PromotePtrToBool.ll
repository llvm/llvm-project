; RUN: opt < %s -passes=globalopt -S | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin7"

; CHECK: @X = internal unnamed_addr global ptr null
@X = internal global ptr null		; <ptr> [#uses=2]
@Y = internal global i32 0		; <ptr> [#uses=1]

define void @foo() nounwind {
entry:
	store ptr @Y, ptr @X, align 4
	ret void
}

define ptr @get() nounwind {
entry:
	%0 = load ptr, ptr @X, align 4		; <ptr> [#uses=1]
	ret ptr %0
}
