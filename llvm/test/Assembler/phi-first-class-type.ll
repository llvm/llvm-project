; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; CHECK: error: phi node must have first class type

define void @test() {
entry:
  ret void

bb:
  %phi = phi void ()
  ret void
}
