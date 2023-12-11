;RUN: opt < %s -passes=asan -S | FileCheck %s

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

@__const.__assert_fail.fmt = private unnamed_addr addrspace(4) constant [47 x i8] c"%s:%u: %s: Device-side assertion `%s' failed.\0A\00", align 16

declare void @llvm.memcpy.p0.p4.i64(ptr noalias nocapture writeonly, ptr addrspace(4) noalias nocapture readonly, i64, i1 immarg) #1

define weak hidden void @test_mem_intrinsic() sanitize_address #0 {
; CHECK: define weak hidden void @test_mem_intrinsic() #1 {
; CHECK-NEXT: entry:
; CHECK-NEXT: [[FMT:%.*]] = alloca [47 x i8], align 16, addrspace(5)
; CHECK-NEXT: [[FADDRC:%.*]] = addrspacecast ptr addrspace(5) [[FMT]] to ptr
; CHECK-NEXT: [[ITMP:%.*]] = call ptr @__asan_memcpy(ptr [[FADDRC]], ptr addrspacecast (ptr addrspace(4) @__const.__assert_fail.fmt to ptr), i64 47)
; CHECK-NEXT: ret
entry:
%fmt = alloca [47 x i8], align 16, addrspace(5)
%fmt.ascast = addrspacecast ptr addrspace(5) %fmt to ptr
call void @llvm.memcpy.p0.p4.i64(ptr align 16 %fmt.ascast, ptr addrspace(4) align 16 @__const.__assert_fail.fmt, i64 47, i1 false)
ret void
}

