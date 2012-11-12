; RUN: llc < %s -disable-cfi | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin8"

define void @_Z3bazv() {
	call void @0( )		; <i32>:1 [#uses=0]
	ret void
}

define internal void @""() {
	call i32 @_Z3barv( )		; <i32>:4 [#uses=1]
	ret void
}
; CHECK: unnamed_1.eh

declare i32 @_Z3barv()
