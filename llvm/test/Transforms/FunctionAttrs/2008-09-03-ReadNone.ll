; RUN: opt < %s -basic-aa -functionattrs -S | FileCheck %s
; RUN: opt < %s -aa-pipeline=basic-aa -passes=function-attrs -S | FileCheck %s

@x = global i32 0

; CHECK: Function Attrs
; CHECK-SAME: readnone
; CHECK-NEXT: declare i32 @e
declare i32 @e() readnone

; CHECK: Function Attrs
; CHECK-SAME: readnone
; CHECK-NEXT: define i32 @f
define i32 @f() {
	%tmp = call i32 @e( )		; <i32> [#uses=1]
	ret i32 %tmp
}

; CHECK: Function Attrs
; CHECK-SAME: readnone
; CHECK-NEXT: define i32 @g
define i32 @g() readonly {
	ret i32 0
}

; CHECK: Function Attrs
; CHECK-SAME: readnone
; CHECK-NEXT: define i32 @h
define i32 @h() readnone {
	%tmp = load i32, i32* @x		; <i32> [#uses=1]
	ret i32 %tmp
}
