; RUN: llc -mtriple=hexagon -O2 < %s | FileCheck %s

target triple = "hexagon-unknown--elf"

; Test to see if we inline memsets when the array size is small.
; CHECK-LABEL: f0
; CHECK-DAG: memw
; CHECK-DAG: memb
; CHECK-DAG: memh
define i32 @f0() #0 {
b0:
  %v0 = alloca [10 x i32], align 8
  call void @llvm.memset.p0.i32(ptr align 8 %v0, i8 0, i32 7, i1 false)
  call void @f1(ptr %v0) #0
  ret i32 0
}

; Function Attrs: nounwind
declare void @f1(ptr) #0

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0.i32(ptr nocapture writeonly, i8, i32, i1) #1

attributes #0 = { nounwind }
attributes #1 = { argmemonly nounwind }
