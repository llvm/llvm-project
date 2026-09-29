; RUN: llc < %s -relocation-model=pic -mtriple=sparc | FileCheck %s

@value = external global i32

define i32 @test() nounwind {
; CHECK:    ld [%i0+value], %i0
entry:
  %0 = load i32, ptr @value
  ret i32 %0
}

!llvm.module.flags = !{!0}

!0 = !{i32 7, !"PIC Level", i32 1}
