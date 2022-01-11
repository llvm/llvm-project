; RUN: not llvm-diff %s %s 2>&1 | FileCheck %s

define void @foo() {
entry:
  callbr void asm sideeffect "", "i,i,~{dirflag},~{fpsr},~{flags}"(i8* blockaddress(@foo, %return), i8* blockaddress(@foo, %t_no))
          to label %asm.fallthrough [label %return, label %t_no]

asm.fallthrough:
  br label %return

t_no:
  br label %return

return:
  ret void
}

; CHECK:      in function bar:
; CHECK-NOT:  in function foo:
; CHECK-NEXT:  in block %entry:
; CHECK-NEXT:    >   callbr void asm sideeffect "", "i,i,~{dirflag},~{fpsr},~{flags}"(i8* blockaddress(@foo, %t_no), i8* blockaddress(@foo, %return))
; CHECK-NEXT:          to label %asm.fallthrough [label %return, label %t_no]
; CHECK-NEXT:    <   callbr void asm sideeffect "", "i,i,~{dirflag},~{fpsr},~{flags}"(i8* blockaddress(@foo, %t_no), i8* blockaddress(@foo, %return))
; CHECK-NEXT:          to label %asm.fallthrough [label %return, label %t_no]

define void @bar() {
entry:
  callbr void asm sideeffect "", "i,i,~{dirflag},~{fpsr},~{flags}"(i8* blockaddress(@foo, %t_no), i8* blockaddress(@foo, %return))
          to label %asm.fallthrough [label %return, label %t_no]

asm.fallthrough:
  br label %return

t_no:
  br label %return

return:
  ret void
}
