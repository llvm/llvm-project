; RUN: not llc -mtriple=x86_64-unknown-linux-gnu -filetype=obj %s -o %t.o 2>&1 | FileCheck %s

; An inline assembly error must prevent deferred DWARF finalization.
define void @f() {
  call void asm sideeffect ".file 3 \22b\22", ""()
  ret void
}

; CHECK: error: unassigned file number: 1 for .file directives
; CHECK: error: unassigned file number: 2 for .file directives
; CHECK-NOT: Assertion
