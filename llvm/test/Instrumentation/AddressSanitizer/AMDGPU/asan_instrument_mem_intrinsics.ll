;RUN: opt < %s -passes=asan -S | FileCheck %s

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

@__const.__src__0 = private unnamed_addr constant [12 x i8] c"oldstring_0\00", align 16
@__const.__dest__0 = private unnamed_addr constant [12 x i8] c"newstring_0\00", align 16
@__const.__src__1 = private unnamed_addr addrspace(4) constant [12 x i8] c"oldstring_1\00", align 16
@__const.__dest__1 = private unnamed_addr addrspace(4) constant [12 x i8] c"newstring_1\00", align 16

declare void @llvm.memcpy.p0.p0.i64(ptr addrspace(4) noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)
declare void @llvm.memcpy.p0.p1.i64(ptr noalias nocapture writeonly, ptr addrspace(4) noalias nocapture readonly, i64, i1 immarg)
declare void @llvm.memcpy.p0.p2.i64(ptr addrspace(4) noalias nocapture writeonly, ptr addrspace(4) noalias nocapture readonly, i64, i1 immarg)

declare void @llvm.memcpy.p0.p0.i32(ptr addrspace(4) noalias nocapture writeonly, ptr noalias nocapture readonly, i32, i1 immarg)
declare void @llvm.memcpy.p0.p1.i32(ptr noalias nocapture writeonly, ptr addrspace(4) noalias nocapture readonly, i32, i1 immarg)
declare void @llvm.memcpy.p0.p2.i32(ptr addrspace(4) noalias nocapture writeonly, ptr addrspace(4) noalias nocapture readonly, i32, i1 immarg)

declare void @llvm.memmove.p0.p0.i64(ptr addrspace(4) nocapture writeonly, ptr nocapture readonly, i64, i1 immarg)
declare void @llvm.memmove.p0.p1.i64(ptr nocapture writeonly, ptr addrspace(4) nocapture readonly, i64, i1 immarg)
declare void @llvm.memmove.p0.p2.i64(ptr addrspace(4) nocapture writeonly, ptr addrspace(4) nocapture readonly, i64, i1 immarg)

declare void @llvm.memmove.p0.p0.i32(ptr addrspace(4) nocapture writeonly, ptr nocapture readonly, i32, i1 immarg)
declare void @llvm.memmove.p0.p1.i32(ptr nocapture writeonly, ptr addrspace(4) nocapture readonly, i32, i1 immarg)
declare void @llvm.memmove.p0.p2.i32(ptr addrspace(4) nocapture writeonly, ptr addrspace(4) nocapture readonly, i32, i1 immarg)

define weak hidden void @test_mem_intrinsic_memcpy() sanitize_address {
; CHECK: define weak hidden void @test_mem_intrinsic_memcpy() #0 {
; CHECK-NEXT:	entry:
; CHECK-NEXT: %0 = call ptr @__asan_memcpy(ptr addrspacecast (ptr addrspace(4) @__const.__dest__1 to ptr), ptr @__const.__src__0, i64 12)
; CHECK-NEXT: %1 = call ptr @__asan_memcpy(ptr @__const.__dest__0, ptr addrspacecast (ptr addrspace(4) @__const.__src__1 to ptr), i64 12)
; CHECK-NEXT: %2 = call ptr @__asan_memcpy(ptr addrspacecast (ptr addrspace(4) @__const.__dest__1 to ptr), ptr addrspacecast (ptr addrspace(4) @__const.__src__1 to ptr), i64 12)
; CHECK-NEXT: %3 = call ptr @__asan_memcpy(ptr addrspacecast (ptr addrspace(4) @__const.__dest__1 to ptr), ptr @__const.__src__0, i64 12)
; CHECK-NEXT: %4 = call ptr @__asan_memcpy(ptr @__const.__dest__0, ptr addrspacecast (ptr addrspace(4) @__const.__src__1 to ptr), i64 12)
; CHECK-NEXT: %5 = call ptr @__asan_memcpy(ptr addrspacecast (ptr addrspace(4) @__const.__dest__1 to ptr), ptr addrspacecast (ptr addrspace(4) @__const.__src__1 to ptr), i64 12)
; CHECK-NEXT: ret void
entry:
  call void @llvm.memcpy.p0.p0.i64(ptr addrspace(4) align 16 @__const.__dest__1, ptr align 16 @__const.__src__0, i64 12, i1 false)
  call void @llvm.memcpy.p0.p1.i64(ptr align 16 @__const.__dest__0, ptr addrspace(4) align 16 @__const.__src__1, i64 12, i1 false)
  call void @llvm.memcpy.p0.p2.i64(ptr addrspace(4) align 16 @__const.__dest__1, ptr addrspace(4) align 16 @__const.__src__1, i64 12, i1 false)
  call void @llvm.memcpy.p0.p0.i32(ptr addrspace(4) align 16 @__const.__dest__1, ptr align 16 @__const.__src__0, i32 12, i1 false)
  call void @llvm.memcpy.p0.p1.i32(ptr align 16 @__const.__dest__0, ptr addrspace(4) align 16 @__const.__src__1, i32 12, i1 false)
  call void @llvm.memcpy.p0.p2.i32(ptr addrspace(4) align 16 @__const.__dest__1, ptr addrspace(4) align 16 @__const.__src__1, i32 12, i1 false)
  ret void
}

define dso_local void @test_mem_intrinsic_memmove() sanitize_address {
; CHECK: define dso_local void @test_mem_intrinsic_memmove() #0 {
; CHECK-NEXT: entry:
; CHECK-NEXT: %0 = call ptr @__asan_memmove(ptr addrspacecast (ptr addrspace(4) @__const.__dest__1 to ptr), ptr @__const.__src__0, i64 12)
; CHECK-NEXT: %1 = call ptr @__asan_memmove(ptr @__const.__dest__0, ptr addrspacecast (ptr addrspace(4) @__const.__src__1 to ptr), i64 12)
; CHECK-NEXT: %2 = call ptr @__asan_memmove(ptr addrspacecast (ptr addrspace(4) @__const.__dest__1 to ptr), ptr addrspacecast (ptr addrspace(4) @__const.__src__1 to ptr), i64 12)
; CHECK-NEXT: %3 = call ptr @__asan_memmove(ptr addrspacecast (ptr addrspace(4) @__const.__dest__1 to ptr), ptr @__const.__src__0, i64 12)
; CHECK-NEXT: %4 = call ptr @__asan_memmove(ptr @__const.__dest__0, ptr addrspacecast (ptr addrspace(4) @__const.__src__1 to ptr), i64 12)
; CHECK-NEXT: %5 = call ptr @__asan_memmove(ptr addrspacecast (ptr addrspace(4) @__const.__dest__1 to ptr), ptr addrspacecast (ptr addrspace(4) @__const.__src__1 to ptr), i64 12)
; CHECK-NEXT: ret void
entry:
  call void @llvm.memmove.p0.p0.i64(ptr addrspace(4) align 16 @__const.__dest__1, ptr align 16 @__const.__src__0, i64 12, i1 false)
  call void @llvm.memmove.p0.p1.i64(ptr align 16 @__const.__dest__0, ptr addrspace(4) align 16 @__const.__src__1, i64 12, i1 false)
  call void @llvm.memmove.p0.p2.i64(ptr addrspace(4) align 16 @__const.__dest__1, ptr addrspace(4) align 16 @__const.__src__1, i64 12, i1 false)
  call void @llvm.memmove.p0.p0.i32(ptr addrspace(4) align 16 @__const.__dest__1, ptr align 16 @__const.__src__0, i32 12, i1 false)
  call void @llvm.memmove.p0.p1.i32(ptr align 16 @__const.__dest__0, ptr addrspace(4) align 16 @__const.__src__1, i32 12, i1 false)
  call void @llvm.memmove.p0.p2.i32(ptr addrspace(4) align 16 @__const.__dest__1, ptr addrspace(4) align 16 @__const.__src__1, i32 12, i1 false)
  ret void
}

attributes #0 = { sanitize_address }
