; RUN: llc -O0 -fast-isel=false -march=x86 < %s | FileCheck %s

; No need for branching when the default and only destination follows
; immediately after the switch.
; CHECK-LABEL: no_branch:
; CHECK-NOT: jmp
; CHECK: ret

define void @no_branch(i32 %x) {
entry:
  switch i32 %x, label %exit [ ]
exit:
  ret void
}
