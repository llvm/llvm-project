; RUN: llc -mtriple=aarch64-unknown-linux-gnu < %s | FileCheck %s
; CHECK-LABEL: foo:
; CHECK: TEST .Ltmp0
define void @foo() {
entry:
  br label %bar
bar:
  call void asm sideeffect "#TEST $0", "i,~{dirflag},~{fpsr},~{flags}"(i8* blockaddress(@foo, %bar))
  ret void
indirectgoto:
  indirectbr i8* undef, [label %bar]
}
