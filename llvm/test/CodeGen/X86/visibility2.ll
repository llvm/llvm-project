; This test case ensures that when the visibility of a global declaration is 
; emitted they are not treated as definitions.  Test case for r132825.
; Fixes <rdar://problem/9429892>.
;
; RUN: llc -mtriple=x86_64-apple-darwin %s -o - | FileCheck %s

@foo_private_extern_str = external hidden global ptr

define void @foo1() nounwind ssp {
entry:
  %tmp = load ptr, ptr @foo_private_extern_str, align 8
  call void @foo3(ptr %tmp)
  ret void
}

declare void @foo3(ptr)

; CHECK-NOT: .private_extern
