; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; CHECK: invalid type for function argument
define void @test() {
bb:
  call void asm "", ""(label %bb)
  ret void
}

