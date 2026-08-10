; Test to make sure that the 'private' is used correctly.
;
; RUN: llc < %s -mtriple=x86_64-pc-linux | FileCheck %s

define private void @foo() {
        ret void

; CHECK: .Lfoo:
}

define i32 @bar() {
        call void @foo()
	%1 = load i32, ptr @baz, align 4
        ret i32 %1

; CHECK-LABEL: bar:
; CHECK: callq .Lfoo
; CHECK: movl	.Lbaz(%rip)
}

@baz = private global i32 4
; CHECK: .Lbaz:
