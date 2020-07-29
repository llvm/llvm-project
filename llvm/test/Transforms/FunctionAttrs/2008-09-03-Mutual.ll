; RUN: opt < %s -function-attrs -S | FileCheck %s
; RUN: opt < %s -passes=function-attrs -S | FileCheck %s

; CHECK: Function Attrs
; CHECK-SAME: readnone
; CHECK-NEXT: define i32 @a
define i32 @a() {
	%tmp = call i32 @b( )		; <i32> [#uses=1]
	ret i32 %tmp
}

; CHECK: Function Attrs
; CHECK-SAME: readnone
; CHECK-NEXT: define i32 @b
define i32 @b() {
	%tmp = call i32 @a( )		; <i32> [#uses=1]
	ret i32 %tmp
}
