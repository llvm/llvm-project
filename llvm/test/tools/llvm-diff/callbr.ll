; RUN: llvm-diff %s %s

define void @foo() {
entry:
  callbr void asm sideeffect "", "!i,!i,~{dirflag},~{fpsr},~{flags}"()
          to label %asm.fallthrough [label %return, label %t_no]

asm.fallthrough:
  br label %return

t_no:
  br label %return

return:
  ret void
}

define void @bar() {
entry:
  callbr void asm sideeffect "", "!i,!i,~{dirflag},~{fpsr},~{flags}"()
          to label %asm.fallthrough [label %return, label %t_no]

asm.fallthrough:
  br label %return

t_no:
  br label %return

return:
  ret void
}
