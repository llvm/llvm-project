; RUN: llc -march=hexagon -O2 < %s | FileCheck %s

target triple = "hexagon-unknown--elf"

; Test to see if we inline memsets when the array size is small.
; CHECK-LABEL: f0
; CHECK-DAG: memw
; CHECK-DAG: memb
; CHECK-DAG: memh
define i32 @f0() #0 {
b0:
  %v0 = alloca [10 x i32], align 8
  %v1 = bitcast [10 x i32]* %v0 to i8*
  call void @llvm.memset.p0i8.i32(i8* align 8 %v1, i8 0, i32 7, i1 false)
  %v2 = getelementptr inbounds [10 x i32], [10 x i32]* %v0, i32 0, i32 0
  call void @f1(i32* %v2) #0
  ret i32 0
}

; Function Attrs: nounwind
declare void @f1(i32*) #0

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i32(i8* nocapture writeonly, i8, i32, i1) #1

attributes #0 = { nounwind }
attributes #1 = { argmemonly nounwind }
