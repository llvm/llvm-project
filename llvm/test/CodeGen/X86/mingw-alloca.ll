; RUN: llc < %s | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "i386-pc-mingw32"

define void @foo1(i32 %N) nounwind {
entry:
; CHECK: _foo1:
; CHECK: calll __alloca
	%tmp14 = alloca i32, i32 %N		; <i32*> [#uses=1]
	call void @bar1( i32* %tmp14 )
	ret void
}

declare void @bar1(i32*)

define void @foo2(i32 inreg  %N) nounwind {
entry:
; CHECK: _foo2:
; CHECK: andl $-16, %esp
; CHECK: pushl %eax
; CHECK: calll __alloca
; CHECK: movl	8028(%esp), %eax
	%A2 = alloca [2000 x i32], align 16		; <[2000 x i32]*> [#uses=1]
	%A2.sub = getelementptr [2000 x i32]* %A2, i32 0, i32 0		; <i32*> [#uses=1]
	call void @bar2( i32* %A2.sub, i32 %N )
	ret void
}

declare void @bar2(i32*, i32)
