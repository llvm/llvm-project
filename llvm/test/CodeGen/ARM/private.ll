; Test to make sure that the 'private' is used correctly.
;
; RUN: llc < %s -mtriple=arm-linux-gnueabi | FileCheck %s
; CHECK: .Lfoo:
; CHECK-LABEL: bar:
; CHECK: bl .Lfoo
; CHECK: .long .Lbaz
; CHECK: .Lbaz:

define private void @foo() {
        ret void
}

@baz = private global i32 4

define i32 @bar() {
        call void @foo()
	%1 = load i32, ptr @baz, align 4
        ret i32 %1
}

