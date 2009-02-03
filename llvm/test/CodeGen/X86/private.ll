; Test to make sure that the 'private' is used correctly.
;
; RUN: llvm-as < %s | llc -mtriple=x86_64-pc-linux | grep .Lfoo:
; RUN: llvm-as < %s | llc -mtriple=x86_64-pc-linux | grep call.*\.Lfoo
; RUN: llvm-as < %s | llc -mtriple=x86_64-pc-linux | grep .Lbaz:
; RUN: llvm-as < %s | llc -mtriple=x86_64-pc-linux | grep movl.*\.Lbaz

declare void @foo()

define private void @foo() {
        ret void
}

@baz = private global i32 4;

define i32 @bar() {
        call void @foo()
	%1 = load i32* @baz, align 4
        ret i32 %1
}
