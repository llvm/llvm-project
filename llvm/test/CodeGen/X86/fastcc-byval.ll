; RUN: llc < %s -tailcallopt=false | FileCheck %s
; CHECK: movl 8(%esp), %eax
; CHECK: movl 8(%esp), %eax
; CHECK-NOT: movl 8(%esp), %eax

; PR3122
; rdar://6400815

; byval requires a copy, even with fastcc.

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9.5"
	%struct.MVT = type { i32 }

define fastcc i32 @bar() nounwind {
	%V = alloca %struct.MVT
	store i32 1, ptr %V
	call fastcc void @foo(ptr byval(%struct.MVT) %V) nounwind
	%t = load i32, ptr %V
	ret i32 %t
}

declare fastcc void @foo(ptr byval(%struct.MVT))
