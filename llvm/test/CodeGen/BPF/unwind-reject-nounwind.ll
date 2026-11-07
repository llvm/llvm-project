; RUN: not llc -mtriple=bpfel -mcpu=v4 -filetype=asm -o - %s 2>&1 | FileCheck %s
; RUN: not llc -mtriple=bpfel -mcpu=v4 -filetype=obj -o /dev/null %s 2>&1 | FileCheck %s

; There is no unwind_goto after a call of a nounwind function, so an unwind
; cannot start there.

define void @cannot() nounwind uwtable {
entry:
  call void @llvm.bpf.unwind.exit()
  unreachable
}

; CHECK: error: {{.*}} BPF cannot unwind out of a function that is nounwind or has no uwtable
