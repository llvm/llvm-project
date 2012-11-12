; This test case ensures that when the visibility of a global declaration is 
; emitted they are not treated as definitions.  Test case for r132825.
; Fixes <rdar://problem/9429892>.
;
; RUN: llc -mtriple=x86_64-apple-darwin %s -o - | FileCheck %s

@foo_private_extern_str = external hidden global i8*

define void @foo1() nounwind ssp {
entry:
  %tmp = load i8** @foo_private_extern_str, align 8
  call void @foo3(i8* %tmp)
  ret void
}

declare void @foo3(i8*)

; CHECK-NOT: .private_extern
