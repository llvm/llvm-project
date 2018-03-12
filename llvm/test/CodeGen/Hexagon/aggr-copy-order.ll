; RUN: llc -march=hexagon -mattr=-packets -hexagon-check-bank-conflict=0 < %s | FileCheck %s
; Do not check stores. They undergo some optimizations in the DAG combiner
; resulting in getting out of order. There is likely little that can be
; done to keep the original order.

target triple = "hexagon"

%s.0 = type { i32, i32, i32 }

; Function Attrs: nounwind
define void @f0(%s.0* %a0, %s.0* %a1) #0 {
b0:
; CHECK: = memw({{.*}}+#0)
; CHECK: = memw({{.*}}+#4)
; CHECK: = memw({{.*}}+#8)
  %v0 = alloca %s.0*, align 4
  %v1 = alloca %s.0*, align 4
  store %s.0* %a0, %s.0** %v0, align 4
  store %s.0* %a1, %s.0** %v1, align 4
  %v2 = load %s.0*, %s.0** %v0, align 4
  %v3 = load %s.0*, %s.0** %v1, align 4
  %v4 = bitcast %s.0* %v2 to i8*
  %v5 = bitcast %s.0* %v3 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 4 %v4, i8* align 4 %v5, i32 12, i1 false)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture writeonly, i8* nocapture readonly, i32, i1) #1

attributes #0 = { nounwind }
attributes #1 = { argmemonly nounwind }
