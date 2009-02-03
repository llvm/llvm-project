; RUN: llvm-as < %s | llc | grep unnamed_1_0.eh
; ModuleID = '<stdin>'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin8"

define void @_Z3bazv() {
	call i32 @0( )		; <i32>:1 [#uses=0]
	br label %2
; <label>:2		; preds = %0
	ret void
}

define internal i32 @""() {
	alloca i32		; <i32*>:1 [#uses=2]
	alloca i32		; <i32*>:2 [#uses=2]
	bitcast i32 0 to i32		; <i32>:3 [#uses=0]
	call i32 @_Z3barv( )		; <i32>:4 [#uses=1]
	store i32 %4, i32* %2, align 4
	load i32* %2, align 4		; <i32>:5 [#uses=1]
	store i32 %5, i32* %1, align 4
	br label %6
; <label>:6		; preds = %0
	load i32* %1		; <i32>:7 [#uses=1]
	ret i32 %7
}

declare i32 @_Z3barv()
