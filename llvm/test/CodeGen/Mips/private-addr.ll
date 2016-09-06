; RUN: llc -mtriple=mips-pc-linux -relocation-model=pic < %s | FileCheck %s

define private void @bar() {
  ret void
}

define void()* @foo() {
; CHECK:      foo:
; CHECK:      lw     $[[REG:.*]], %got($bar)($1)
; CHECK-NEXT: jr     $ra
; CHECK-NEXT: addiu  $2, $[[REG]], %lo($bar)

  ret void()* @bar
}
