; Make sure that when the stack may be misaligned on function entry, fixed frame
; elements (here: XMM spills) are accessed using instructions that tolerate
; unaligned access.
;
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -mcpu=x86-64 -mattr=+sse,+sse-unaligned-mem -stack-alignment=8 --frame-pointer=all < %s | FileCheck %s

define dso_local preserve_allcc void @func() #0 {
; CHECK-LABEL: func:
; CHECK: movups  %xmm0, -{{[0-9]+}}(%rbp)
  call void asm sideeffect "", "~{xmm0},~{dirflag},~{fpsr},~{flags}"() #1
; CHECK: movups  -{{[0-9]+}}(%rbp), %xmm0
  ret void
}

attributes #0 = { nounwind }
