; RUN: split-file %s %t
; RUN: not llc -mtriple=x86_64-linux-gnu < %t/error.ll 2>&1 | FileCheck %s --check-prefix=ERROR
; RUN: llc -mtriple=x86_64-linux-gnu < %t/ok.ll | FileCheck %s --check-prefix=OK

; An indirect output with a register constraint is stored after the asm, but a
; callbr's asm can branch to an indirect target before then.

;--- error.ll
; ERROR: error: callbr cannot store indirect output for constraint 'r' on its indirect paths
define void @indirect_reg_output(ptr %p) {
entry:
  callbr void asm "", "=*r,!i"(ptr elementtype(i32) %p)
          to label %a [label %b]
a:
  ret void
b:
  ret void
}

;--- ok.ll
; Without indirect targets, the store is on the only path.
; OK-LABEL: no_indirect_targets:
; OK:       #APP
; OK-NEXT:  #NO_APP
; OK-NEXT:  movl %eax, (%rdi)
define void @no_indirect_targets(ptr %p) {
entry:
  callbr void asm "", "=*r"(ptr elementtype(i32) %p)
          to label %a []
a:
  ret void
}

; The asm writes a memory output itself.
; OK-LABEL: indirect_mem_output:
; OK:       #APP
; OK-NEXT:  # (%rdi)
; OK-NEXT:  #NO_APP
define void @indirect_mem_output(ptr %p) {
entry:
  callbr void asm "# $0", "=*m,!i"(ptr elementtype(i32) %p)
          to label %a [label %b]
a:
  ret void
b:
  ret void
}
