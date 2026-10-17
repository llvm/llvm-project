; RUN: not llc -mtriple=bpfel -mcpu=v4 -filetype=asm -o - %s 2>&1 | FileCheck %s

; The caller of a nounwind function does not look at r3, so an unwind that
; starts there would be lost silently.

declare i32 @_Unwind_RaiseException(ptr)

define void @cannot() nounwind uwtable {
entry:
  %err = call i32 @_Unwind_RaiseException(ptr null)
  unreachable
}

; CHECK: error: {{.*}} BPF cannot unwind out of a function that is nounwind or has no uwtable
