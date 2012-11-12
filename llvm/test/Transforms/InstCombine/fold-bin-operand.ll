; RUN: opt < %s -instcombine -S | FileCheck %s
target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128"

define i1 @f(i1 %x) {
; CHECK: @f
; CHECK: ret i1 false
	%b = and i1 %x, icmp eq (i8* inttoptr (i32 1 to i8*), i8* inttoptr (i32 2 to i8*))
	ret i1 %b
}

define i32 @g(i32 %x) {
; CHECK: @g
; CHECK: ret i32 %x
	%b = add i32 %x, zext (i1 icmp eq (i8* inttoptr (i32 1000000 to i8*), i8* inttoptr (i32 2000000 to i8*)) to i32)
	ret i32 %b
}

