; RUN: not llc -mtriple=aarch64-darwin-- -filetype=obj %s -o /dev/null 2>&1 >/dev/null | FileCheck %s
; CHECK: error: unsupported relocation of local symbol 'L_foo_end'.
; CHECK-SAME: Must have non-local symbol earlier in section.

; Make sure that we emit an error when we try to reference something that
; doesn't belong to a section.
define void @foo() local_unnamed_addr {
  call void asm sideeffect "b L_foo_end\0A", ""()
  ret void
}
