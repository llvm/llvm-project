; Test to make sure that the 'private' is used correctly.
;
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-linux-gnu | \
; RUN: FileCheck --check-prefix=LINUX %s
;
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-apple-darwin | \
; RUN: FileCheck --check-prefix=OSX %s

; LINUX: .Lfoo:
; OSX: l_foo:
define private void @foo() nounwind {
        ret void
}

define i32 @bar() nounwind {
; LINUX: bl{{.*}}.Lfoo
; OSX: bl{{.*}}l_foo
        call void @foo()

; LINUX: lis{{.*}}.Lbaz
; OSX:  lis{{.*}}l_baz
	%1 = load i32, i32* @baz, align 4
        ret i32 %1
}

; LINUX: .Lbaz:
; OSX: l_baz:
@baz = private global i32 4
