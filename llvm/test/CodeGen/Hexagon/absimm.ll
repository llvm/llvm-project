; RUN: llc -march=hexagon < %s | FileCheck %s
; Check that we generate absolute addressing mode instructions
; with immediate value.

define i32 @f1(i32 %i) nounwind {
; CHECK: memw(##786432) = r{{[0-9]+}}
entry:
  store volatile i32 %i, i32* inttoptr (i32 786432 to i32*), align 262144
  ret i32 %i
}

define i32* @f2(i32* nocapture %i) nounwind {
entry:
; CHECK: r{{[0-9]+}} = memw(##786432)
  %0 = load volatile i32, i32* inttoptr (i32 786432 to i32*), align 262144
  %1 = inttoptr i32 %0 to i32*
  ret i32* %1
  }
