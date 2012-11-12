; RUN: llc < %s -march=x86 -x86-asm-syntax=intel
; PR1061
target datalayout = "e-p:32:32"
target triple = "i686-pc-linux-gnu"

define void @bar(i32 %n) {
entry:
	switch i32 %n, label %bb12 [
		 i32 1, label %bb
		 i32 2, label %bb6
		 i32 4, label %bb7
		 i32 5, label %bb8
		 i32 6, label %bb10
		 i32 7, label %bb1
		 i32 8, label %bb3
		 i32 9, label %bb4
		 i32 10, label %bb9
		 i32 11, label %bb2
		 i32 12, label %bb5
		 i32 13, label %bb11
	]

bb:		; preds = %entry
	call void (...)* @foo1( )
	ret void

bb1:		; preds = %entry
	call void (...)* @foo2( )
	ret void

bb2:		; preds = %entry
	call void (...)* @foo6( )
	ret void

bb3:		; preds = %entry
	call void (...)* @foo3( )
	ret void

bb4:		; preds = %entry
	call void (...)* @foo4( )
	ret void

bb5:		; preds = %entry
	call void (...)* @foo5( )
	ret void

bb6:		; preds = %entry
	call void (...)* @foo1( )
	ret void

bb7:		; preds = %entry
	call void (...)* @foo2( )
	ret void

bb8:		; preds = %entry
	call void (...)* @foo6( )
	ret void

bb9:		; preds = %entry
	call void (...)* @foo3( )
	ret void

bb10:		; preds = %entry
	call void (...)* @foo4( )
	ret void

bb11:		; preds = %entry
	call void (...)* @foo5( )
	ret void

bb12:		; preds = %entry
	call void (...)* @foo6( )
	ret void
}

declare void @foo1(...)

declare void @foo2(...)

declare void @foo6(...)

declare void @foo3(...)

declare void @foo4(...)

declare void @foo5(...)
