; RUN: opt < %s -basicaa -functionattrs -S | FileCheck %s
@x = global i32 0

; CHECK: declare i32 @e() #0
declare i32 @e() readnone

; CHECK: define i32 @f() #0
define i32 @f() {
	%tmp = call i32 @e( )		; <i32> [#uses=1]
	ret i32 %tmp
}

; CHECK: define i32 @g() #0
define i32 @g() readonly {
	ret i32 0
}

; CHECK: define i32 @h() #0
define i32 @h() readnone {
	%tmp = load i32, i32* @x		; <i32> [#uses=1]
	ret i32 %tmp
}

; CHECK: attributes #0 = { readnone }
