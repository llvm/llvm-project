; RUN: llc -o - %s -mtriple=x86_64-linux-gnu -trap-unreachable | FileCheck %s
; RUN: llc -o - %s -mtriple=x86_64-linux-gnu -trap-unreachable -fast-isel | FileCheck %s

define dso_local void @foo() #0 {
entry:
  tail call void asm sideeffect "movl 3,%eax", "~{dirflag},~{fpsr},~{flags}"()
  unreachable
}
; CHECK-NOT: ud2

attributes #0 = { naked }
