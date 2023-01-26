; Test to make sure that the 'private' is used correctly.
;
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-linux-gnu | FileCheck %s

; CHECK: .Lfoo:
define private void @foo() nounwind {
        ret void
}

define i32 @bar() nounwind {
; CHECK: bl{{.*}}.Lfoo
        call void @foo()

; CHECK: lis{{.*}}.Lbaz
	%1 = load i32, ptr @baz, align 4
        ret i32 %1
}

; CHECK: .Lbaz:
@baz = private global i32 4
