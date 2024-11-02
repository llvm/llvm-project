; RUN: llc < %s -mtriple=x86_64-linux-gnu -filetype=obj -o - \
; RUN:  | llvm-objdump --no-print-imm-hex --triple=x86_64-linux-gnu -d - \
; RUN:  | FileCheck %s

; CHECK: 0000000000000000 <test1>:
; CHECK-NEXT:   0: 74 00 je 0x2 <test1+0x2>
; CHECK-NEXT:   2: c3    retq

define void @test1() {
entry:
  callbr void asm sideeffect "je ${0:l}", "!i,~{dirflag},~{fpsr},~{flags}"()
          to label %asm.fallthrough [label %a.b.normal.jump]

asm.fallthrough:
  ret void

a.b.normal.jump:
  ret void
}
