; RUN: not llc -mtriple=bpfel -mcpu=v4 -filetype=asm -o - %s 2>&1 | FileCheck %s

; There is no unwind_goto after a call of a nounwind function, so an unwind
; cannot start there.

declare i32 @_Unwind_RaiseException(ptr)

define void @cannot() nounwind uwtable {
entry:
  %err = call i32 @_Unwind_RaiseException(ptr null)
  unreachable
}

; CHECK: error: {{.*}} BPF cannot unwind out of a function that is nounwind or has no uwtable
