; RUN: llc < %s -mtriple=x86_64-linux-gnu -filetype=obj -o - \
; RUN:  | llvm-objdump --triple=x86_64-linux-gnu -d - \
; RUN:  | FileCheck %s

; CHECK: 0000000000000000 <test1>:
; CHECK-NEXT:   0: 74 00 je 0x2 <test1+0x2>
; CHECK-NEXT:   2: c3    retq

define void @test1() {
entry:
  callbr void asm sideeffect "je ${0:l}", "X,~{dirflag},~{fpsr},~{flags}"(i8* blockaddress(@test1, %a.b.normal.jump))
          to label %asm.fallthrough [label %a.b.normal.jump]

asm.fallthrough:
  ret void

a.b.normal.jump:
  ret void
}
