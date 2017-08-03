; RUN: llc < %s -march=x86 | FileCheck %s
target datalayout = "e-p:32:32"
target triple = "i686-apple-darwin9"

define void @test() {
; CHECK-LABEL: test:
; CHECK-NOT: ret
; CHECK: psrlw $8, %xmm0
; CHECK: ret

  tail call void asm sideeffect "psrlw $0, %xmm0", "X,~{dirflag},~{fpsr},~{flags}"( i32 8 )
  ret void
}

