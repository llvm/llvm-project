; RUN: opt < %s -passes=strip -S | FileCheck %s

; CHECK: foo
; CHECK: bar
; CHECK: foo
; CHECK: bar

@llvm.used = appending global [2 x ptr] [ ptr @foo, ptr @bar ], section "llvm.metadata"		; <ptr> [#uses=0]
@foo = internal constant i32 41		; <ptr> [#uses=1]

define internal i32 @bar() nounwind  {
entry:
	ret i32 42
}

define i32 @main() nounwind  {
entry:
	ret i32 0
}

