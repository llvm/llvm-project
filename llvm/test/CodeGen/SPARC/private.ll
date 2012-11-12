; Test to make sure that the 'private' is used correctly.
;
; RUN: llc < %s  -march=sparc | FileCheck %s

define private void @foo() {
        ret void
}
; CHECK: [[FOO:\..*foo]]:

@baz = private global i32 4

define i32 @bar() {
        call void @foo()
	%1 = load i32* @baz, align 4
        ret i32 %1
}

; CHECK: call [[FOO]]
; CHECK: ld {{.+}}[[BAZ:\..*baz]]

; CHECK: [[BAZ]]
