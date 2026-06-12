; RUN: opt -passes=inline -S < %s | FileCheck %s

; Check that we don't blow up when the callee is in a RefSCC with only one
; function (@g), but references itself.

; CHECK: define void @f
; CHECK-NEXT: call void @g(), !inline_history [[HISTORY:![0-9]+]]
; CHECK-NEXT: call void @g(), !inline_history [[HISTORY]]
; CHECK-NEXT: call void @g(), !inline_history [[HISTORY]]
; CHECK-NEXT: call void @g(), !inline_history [[HISTORY]]
; CHECK-NEXT: ret void
define void @f() {
	call void @g()
	ret void
}

; minsize/function-inline-cost to prevent the inliner from inlining calls in the context of @g
define void @g() minsize {
	call void @h(ptr @g)
	call void @h(ptr @g)
	ret void
}

define void @h(ptr %p) "function-inline-cost"="20" {
	call void %p()
	call void %p()
	ret void
}

; CHECK: [[HISTORY]] = !{ptr @h, ptr @g}
