; RUN: opt < %s -passes=function-attrs -S | FileCheck %s
; PR2792

@g = global i32 0		; <ptr> [#uses=1]

define i32 @f() {
	%t = load volatile i32, ptr @g		; <i32> [#uses=1]
	ret i32 %t
}

; CHECK-NOT: attributes #{{.*}} read
