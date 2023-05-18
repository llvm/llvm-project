; Test to make sure that the 'private' is used correctly.
;
; RUN: llc < %s -march=xcore | FileCheck %s

define private void @foo() {
; CHECK: .Lfoo:
        ret void
}

@baz = private global i32 4

define i32 @bar() {
; CHECK-LABEL: bar:
; CHECK: bl .Lfoo
; CHECK: ldw r0, dp[.Lbaz]
        call void @foo()
	%1 = load i32, ptr @baz, align 4
        ret i32 %1
}

; CHECK: .Lbaz:
