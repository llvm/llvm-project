; RUN: llc -mtriple=arm-eabi %s -o - | FileCheck %s

define i32 @foo() {
entry:
	%A = alloca [1123 x i32], align 16		; <ptr> [#uses=1]
	%B = alloca [3123 x i32], align 16		; <ptr> [#uses=1]
	%C = alloca [12312 x i32], align 16		; <ptr> [#uses=1]
	%tmp = call i32 (...) @bar( ptr %B, ptr %A, ptr %C )		; <i32> [#uses=0]
	ret i32 undef
}

declare i32 @bar(...)

; CHECK-NOT: add{{.*}}#0

