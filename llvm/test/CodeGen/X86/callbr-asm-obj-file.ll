; RUN: llc < %s -mtriple=x86_64-linux-gnu -filetype=obj -o - \
; RUN:  | llvm-objdump --no-print-imm-hex --triple=x86_64-linux-gnu -d - \
; RUN:  | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-linux-gnu -stop-after=finalize-isel -o %t.mir
; RUN: not llc %t.mir -start-after=finalize-isel -filetype=obj -o %t.o 2>&1 \
; RUN:  | FileCheck %s --check-prefix=ERROR

; FIXME: MIR does not preserve LabelMustBeEmitted on the inline asm target.
; ERROR: error: expected a symbol reference
; ERROR-NEXT: je ""

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
