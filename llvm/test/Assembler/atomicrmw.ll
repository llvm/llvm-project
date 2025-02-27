; RUN: llvm-as %s -o - | llvm-dis | FileCheck %s

; Make sure that we can parse an atomicrmw with an operand defined later in the function.

; CHECK: @f
; CHECK: atomicrmw
define void @f(i1 %arg) {
  entry:
    br label %def

  use:
    %x = atomicrmw add ptr undef, i32 %y monotonic
    ret void

  def:
    %y = add i32 undef, undef
    br i1 %arg, label %use, label %use
}
