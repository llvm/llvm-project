; RUN: llc -mtriple x86_64-unknown-linux-gnu -O0 -fast-isel=true -relocation-model=pic -filetype asm -o - %s | FileCheck %s

declare void @f() local_unnamed_addr #0

define void @g() local_unnamed_addr {
entry:
  call void @f()
  ret void
}

attributes #0 = { nonlazybind }

; CHECK-LABEL: g:
; CHECK-LABEL: callq *f@GOTPCREL(%rip)
; CHECK-LABEL: retq

