; Test to make sure that the 'private' is used correctly.
;
; RUN: llc < %s -mtriple=x86_64-pc-linux | grep .Lfoo:
; RUN: llc < %s -mtriple=x86_64-pc-linux | grep call.*\.Lfoo
; RUN: llc < %s -mtriple=x86_64-pc-linux | grep .Lbaz:
; RUN: llc < %s -mtriple=x86_64-pc-linux | grep movl.*\.Lbaz

define private void @foo() {
        ret void
}

@baz = private global i32 4

define i32 @bar() {
        call void @foo()
	%1 = load i32* @baz, align 4
        ret i32 %1
}
