; RUN: llc -mtriple=arm-eabi -mattr=+v4t %s -o - | FileCheck %s

; CHECK-LABEL: bar
; CHECK-NOT: orr
; CHECK-NOT: mov
define void @bar(ptr %P, ptr %Q) {
entry:
	%tmp = load i16, ptr %Q, align 1		; <i16> [#uses=1]
	store i16 %tmp, ptr %P, align 1
	ret void
}

; CHECK-LABEL: foo
; CHECK-NOT: orr
; CHECK-NOT: mov
define void @foo(ptr %P, ptr %Q) {
entry:
	%tmp = load i32, ptr %Q, align 1		; <i32> [#uses=1]
	store i32 %tmp, ptr %P, align 1
	ret void
}
