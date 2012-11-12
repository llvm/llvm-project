; RUN: opt < %s -instcombine -S | FileCheck %s

define i32 @testAdd(i32 %X, i32 %Y) {
	%tmp = add i32 %X, %Y
; CHECK: %tmp = add i32 %X, %Y
	%tmp.l = bitcast i32 %tmp to i32
	ret i32 %tmp.l
; CHECK: ret i32 %tmp
}
