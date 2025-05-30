; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

%struct.B = type { [2 x i32] }
%struct.A = type { i64, %struct.B }

@__const.foo.b = private unnamed_addr addrspace(2) constant %struct.B { [2 x i32] [i32 1, i32 2] }, align 4
@__const.bar.a = private unnamed_addr addrspace(2) constant %struct.A { i64 0, %struct.B { [2 x i32] [i32 1, i32 2] } }, align 8

; Function Attrs: convergent nounwind
define spir_func void @foo(%struct.A* noalias sret(%struct.A) %agg.result) #0 {
entry:
  %b = alloca %struct.B, align 4
  %0 = bitcast %struct.B* %b to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %0) #2
  %1 = bitcast %struct.B* %b to i8*
  call void @llvm.memcpy.p0i8.p2i8.i32(i8* align 4 %1, i8 addrspace(2)* align 4 bitcast (%struct.B addrspace(2)* @__const.foo.b to i8 addrspace(2)*), i32 8, i1 false)
; CHECK: OpCopyMemorySized %[[#]] %[[#]] %[[#]] Aligned 4
  %b1 = getelementptr inbounds %struct.A, %struct.A* %agg.result, i32 0, i32 1
  %2 = bitcast %struct.B* %b1 to i8*
  %3 = bitcast %struct.B* %b to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 8 %2, i8* align 4 %3, i32 8, i1 false), !tbaa.struct !4
; CHECK: %[[#PTR1:]] = OpInBoundsPtrAccessChain %[[#]] %[[#]] %[[#]] %[[#]]
; CHECK: OpCopyMemorySized %[[#PTR1]] %[[#]] %[[#]] Aligned 8
  %4 = bitcast %struct.B* %b to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %4) #2
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* captures(none)) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p2i8.i32(i8* captures(none) writeonly, i8 addrspace(2)* captures(none) readonly, i32, i1) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i32(i8* captures(none) writeonly, i8* captures(none) readonly, i32, i1) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* captures(none)) #1

; Function Attrs: convergent nounwind
define spir_func void @bar(%struct.B* noalias sret(%struct.B) %agg.result) #0 {
entry:
  %a = alloca %struct.A, align 8
  %0 = bitcast %struct.A* %a to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* %0) #2
  %1 = bitcast %struct.A* %a to i8*
  call void @llvm.memcpy.p0i8.p2i8.i32(i8* align 8 %1, i8 addrspace(2)* align 8 bitcast (%struct.A addrspace(2)* @__const.bar.a to i8 addrspace(2)*), i32 16, i1 false)
; CHECK: OpCopyMemorySized %[[#]] %[[#]] %[[#]] Aligned 8
  %b = getelementptr inbounds %struct.A, %struct.A* %a, i32 0, i32 1
  %2 = bitcast %struct.B* %agg.result to i8*
  %3 = bitcast %struct.B* %b to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 4 %2, i8* align 8 %3, i32 8, i1 false), !tbaa.struct !4
; CHECK: %[[#PTR2:]] = OpInBoundsPtrAccessChain %[[#]] %[[#]] %[[#]] %[[#]]
; CHECK: OpCopyMemorySized %[[#]] %[[#PTR2]] %[[#]] Aligned 4
  %4 = bitcast %struct.A* %a to i8*
  call void @llvm.lifetime.end.p0i8(i64 16, i8* %4) #2
  ret void
}

attributes #0 = { convergent nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 0}
!2 = !{i32 1, i32 2}
!3 = !{!"clang version 9.0.0"}
!4 = !{i64 0, i64 8, !5}
!5 = !{!6, !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C/C++ TBAA"}
