; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

%struct.B = type { [2 x i32] }
%struct.A = type { i64, %struct.B }

@__const.foo.b = private unnamed_addr addrspace(2) constant %struct.B { [2 x i32] [i32 1, i32 2] }, align 4
@__const.bar.a = private unnamed_addr addrspace(2) constant %struct.A { i64 0, %struct.B { [2 x i32] [i32 1, i32 2] } }, align 8

define spir_func void @foo(ptr noalias sret(%struct.A) %agg.result) {
entry:
  %b = alloca %struct.B, align 4
  %0 = bitcast ptr %b to ptr
  call void @llvm.lifetime.start.p0(i64 8, ptr %0)
  %1 = bitcast ptr %b to ptr
  call void @llvm.memcpy.p0.p2.i32(ptr align 4 %1, ptr addrspace(2) align 4 @__const.foo.b, i32 8, i1 false)
; CHECK: OpCopyMemorySized %[[#]] %[[#]] %[[#]] Aligned 4
  %b1 = getelementptr inbounds %struct.A, ptr %agg.result, i32 0, i32 1
  %2 = bitcast ptr %b1 to ptr
  %3 = bitcast ptr %b to ptr
  call void @llvm.memcpy.p0.p0.i32(ptr align 8 %2, ptr align 4 %3, i32 8, i1 false)
; CHECK: %[[#PTR1:]] = OpInBoundsPtrAccessChain %[[#]] %[[#]] %[[#]] %[[#]]
; CHECK: OpCopyMemorySized %[[#PTR1]] %[[#]] %[[#]] Aligned 8
  %4 = bitcast ptr %b to ptr
  call void @llvm.lifetime.end.p0(i64 8, ptr %4)
  ret void
}

declare void @llvm.lifetime.start.p0(i64, ptr captures(none))

declare void @llvm.memcpy.p0.p2.i32(ptr captures(none) writeonly, ptr addrspace(2) captures(none) readonly, i32, i1)

declare void @llvm.memcpy.p0.p0.i32(ptr captures(none) writeonly, ptr captures(none) readonly, i32, i1)

declare void @llvm.lifetime.end.p0(i64, ptr captures(none))

define spir_func void @bar(ptr noalias sret(%struct.B) %agg.result) {
entry:
  %a = alloca %struct.A, align 8
  %0 = bitcast ptr %a to ptr
  call void @llvm.lifetime.start.p0(i64 16, ptr %0)
  %1 = bitcast ptr %a to ptr
  call void @llvm.memcpy.p0.p2.i32(ptr align 8 %1, ptr addrspace(2) align 8 @__const.bar.a, i32 16, i1 false)
; CHECK: OpCopyMemorySized %[[#]] %[[#]] %[[#]] Aligned 8
  %b = getelementptr inbounds %struct.A, ptr %a, i32 0, i32 1
  %2 = bitcast ptr %agg.result to ptr
  %3 = bitcast ptr %b to ptr
  call void @llvm.memcpy.p0.p0.i32(ptr align 4 %2, ptr align 8 %3, i32 8, i1 false)
; CHECK: %[[#PTR2:]] = OpInBoundsPtrAccessChain %[[#]] %[[#]] %[[#]] %[[#]]
; CHECK: OpCopyMemorySized %[[#]] %[[#PTR2]] %[[#]] Aligned 4
  %4 = bitcast ptr %a to ptr
  call void @llvm.lifetime.end.p0(i64 16, ptr %4)
  ret void
}
