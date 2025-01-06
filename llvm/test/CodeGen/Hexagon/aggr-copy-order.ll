; RUN: llc -mtriple=hexagon -mattr=-packets -hexagon-check-bank-conflict=0 < %s | FileCheck %s
; Do not check stores. They undergo some optimizations in the DAG combiner
; resulting in getting out of order. There is likely little that can be
; done to keep the original order.

target triple = "hexagon"

%s.0 = type { i32, i32, i32 }

; Function Attrs: nounwind
define void @f0(ptr %a0, ptr %a1) #0 {
b0:
; CHECK: = memw({{.*}}+#0)
; CHECK: = memw({{.*}}+#4)
; CHECK: = memw({{.*}}+#8)
  %v0 = alloca ptr, align 4
  %v1 = alloca ptr, align 4
  store ptr %a0, ptr %v0, align 4
  store ptr %a1, ptr %v1, align 4
  %v2 = load ptr, ptr %v0, align 4
  %v3 = load ptr, ptr %v1, align 4
  call void @llvm.memcpy.p0.p0.i32(ptr align 4 %v2, ptr align 4 %v3, i32 12, i1 false)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0.p0.i32(ptr nocapture writeonly, ptr nocapture readonly, i32, i1) #1

attributes #0 = { nounwind }
attributes #1 = { argmemonly nounwind }
