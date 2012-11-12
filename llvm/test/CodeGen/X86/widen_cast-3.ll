; RUN: llc < %s -march=x86 -mattr=+sse42 | FileCheck %s
; CHECK: paddd
; CHECK: pextrd
; CHECK: pextrd

; bitcast v12i8 to v3i32

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i686-apple-darwin10.0.0d2"

define void @convert(<12 x i8>* %dst.addr, <3 x i32> %src) nounwind {
entry:
	%add = add <3 x i32> %src, < i32 1, i32 1, i32 1 >		; <<3 x i32>> [#uses=1]
	%conv = bitcast <3 x i32> %add to <12 x i8>		; <<12 x i8>> [#uses=1]
	store <12 x i8> %conv, <12 x i8>* %dst.addr
	ret void
}
