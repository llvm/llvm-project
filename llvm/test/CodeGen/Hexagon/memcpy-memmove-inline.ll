; RUN: llc -mtriple=hexagon -O2 -mno-pairing -mno-compound < %s | FileCheck %s

; Test to see if we inline calls to memcpy/memmove when
; the array size is small.

target triple = "hexagon-unknown--elf"

; CHECK-LABEL: f0:
; CHECK-DAG: [[REG1:r[0-9]*]] = memw(r{{[0-9]*}}+#0)
; CHECK-DAG: [[REG2:r[0-9]*]] = memuh(r{{[0-9]*}}+#4)
; CHECK-DAG: [[REG3:r[0-9]*]] = memub(r{{[0-9]*}}+#6)
; CHECK-DAG: memw(r{{[0-9]*}}+#0) = [[REG1]]
; CHECK-DAG: memh(r{{[0-9]*}}+#4) = [[REG2]]
; CHECK-DAG: memb(r{{[0-9]*}}+#6) = [[REG3]]

define i32 @f0(ptr %a0) #0 {
b0:
  %v0 = alloca [10 x i32], align 8
  call void @llvm.memcpy.p0.p0.i32(ptr align 4 %v0, ptr align 4 %a0, i32 7, i1 false)
  call void @f1(ptr %v0, ptr %a0) #0
  ret i32 0
}

declare void @f1(ptr, ptr)

; CHECK-LABEL: f2:
; CHECK-DAG: [[REG4:r[0-9]*]] = memub(r{{[0-9]*}}+#6)
; CHECK-DAG: [[REG5:r[0-9]*]] = memuh(r{{[0-9]*}}+#4)
; CHECK-DAG: [[REG6:r[0-9]*]] = memw(r{{[0-9]*}}+#0)
; CHECK-DAG: memw(r{{[0-9]*}}+#0) = [[REG6]]
; CHECK-DAG: memh(r{{[0-9]*}}+#4) = [[REG5]]
; CHECK-DAG: memb(r{{[0-9]*}}+#6) = [[REG4]]

define i32 @f2(ptr %a0, ptr %a1) #0 {
b0:
  call void @llvm.memmove.p0.p0.i32(ptr align 4 %a1, ptr align 4 %a0, i32 7, i1 false)
  tail call void @f1(ptr %a1, ptr %a0) #0
  ret i32 0
}

declare void @llvm.memcpy.p0.p0.i32(ptr nocapture writeonly, ptr nocapture readonly, i32, i1) #1
declare void @llvm.memmove.p0.p0.i32(ptr nocapture, ptr nocapture readonly, i32, i1) #1

attributes #0 = { nounwind }
attributes #1 = { argmemonly nounwind }
