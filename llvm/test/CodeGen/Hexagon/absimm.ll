; RUN: llc -march=hexagon < %s | FileCheck %s
; Check that we generate absolute addressing mode instructions
; with immediate value.

define i32 @f1(i32 %i) nounwind {
; CHECK: memw(##786432) = r{{[0-9]+}}
entry:
  store volatile i32 %i, ptr inttoptr (i32 786432 to ptr), align 262144
  ret i32 %i
}

define ptr @f2(ptr nocapture %i) nounwind {
entry:
; CHECK: r{{[0-9]+}} = memw(##786432)
  %0 = load volatile i32, ptr inttoptr (i32 786432 to ptr), align 262144
  %1 = inttoptr i32 %0 to ptr
  ret ptr %1
  }
