; RUN: not llc -march x86 < %s 2>&1 | FileCheck %s

; CHECK: error: invalid operand in inline asm: 'movl $0, ${1:H}' 'H' modifier used on an operand that is a non-offsetable memory reference.
define void @H() {
entry:
  tail call void asm sideeffect "movl $0, ${1:H}", "i,i,~{dirflag},~{fpsr},~{flags}"(i32 1, i32 2)
  ret void
}