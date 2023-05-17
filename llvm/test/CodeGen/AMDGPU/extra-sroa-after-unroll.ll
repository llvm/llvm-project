; RUN: opt -passes='default<O1>,instnamer' -mtriple=amdgcn-- -S -o - %s | FileCheck -check-prefixes=GCN,O1 %s
; RUN: opt -passes='default<O2>,instnamer' -mtriple=amdgcn-- -S -o - %s | FileCheck -check-prefixes=GCN,O2 %s
; RUN: opt -passes='default<O3>,instnamer' -mtriple=amdgcn-- -S -o - %s | FileCheck -check-prefixes=GCN,O3 %s
target datalayout = "A5"

; GCN-LABEL: t0
; O1-NOT: alloca
; O2-NOT: alloca
; O3-NOT: alloca
; GCN-COUNT-27: = load
; GCN-COUNT-26: = add
define protected amdgpu_kernel void @t0(ptr addrspace(1) %p.coerce) #0 {
entry:
  %p = alloca ptr, align 8, addrspace(5)
  %p.ascast = addrspacecast ptr addrspace(5) %p to ptr
  %p.addr = alloca ptr, align 8, addrspace(5)
  %p.addr.ascast = addrspacecast ptr addrspace(5) %p.addr to ptr
  %t = alloca [27 x i32], align 16, addrspace(5)
  %t.ascast = addrspacecast ptr addrspace(5) %t to ptr
  %sum = alloca i32, align 4, addrspace(5)
  %sum.ascast = addrspacecast ptr addrspace(5) %sum to ptr
  %i = alloca i32, align 4, addrspace(5)
  %i.ascast = addrspacecast ptr addrspace(5) %i to ptr
  %cleanup.dest.slot = alloca i32, align 4, addrspace(5)
  %0 = addrspacecast ptr addrspace(1) %p.coerce to ptr
  store ptr %0, ptr %p.ascast, align 8
  %p1 = load ptr, ptr %p.ascast, align 8
  store ptr %p1, ptr %p.addr.ascast, align 8
  call void @llvm.lifetime.start.p5(i64 48, ptr addrspace(5) %t)
  %1 = load ptr, ptr %p.addr.ascast, align 8
  call void @copy(ptr %t.ascast, ptr %1, i32 27)
  call void @llvm.lifetime.start.p5(i64 4, ptr addrspace(5) %sum)
  store i32 0, ptr %sum.ascast, align 4
  call void @llvm.lifetime.start.p5(i64 4, ptr addrspace(5) %i)
  store i32 0, ptr %i.ascast, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %2 = load i32, ptr %i.ascast, align 4
  %cmp = icmp slt i32 %2, 27
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  call void @llvm.lifetime.end.p5(i64 4, ptr addrspace(5) %i)
  br label %for.end

for.body:                                         ; preds = %for.cond
  %3 = load i32, ptr %i.ascast, align 4
  %idxprom = sext i32 %3 to i64
  %arrayidx = getelementptr inbounds [27 x i32], ptr %t.ascast, i64 0, i64 %idxprom
  %4 = load i32, ptr %arrayidx, align 4
  %5 = load i32, ptr %sum.ascast, align 4
  %add = add nsw i32 %5, %4
  store i32 %add, ptr %sum.ascast, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %6 = load i32, ptr %i.ascast, align 4
  %inc = add nsw i32 %6, 1
  store i32 %inc, ptr %i.ascast, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond.cleanup
  %7 = load i32, ptr %sum.ascast, align 4
  %8 = load ptr, ptr %p.addr.ascast, align 8
  store i32 %7, ptr %8, align 4
  call void @llvm.lifetime.end.p5(i64 4, ptr addrspace(5) %sum)
  call void @llvm.lifetime.end.p5(i64 48, ptr addrspace(5) %t)
  ret void
}

define internal void @copy(ptr %d, ptr %s, i32 %N) {
entry:
  %N8 = mul i32 %N, 4
  tail call void @llvm.memcpy.p0.p0.i32(ptr %d, ptr %s, i32 %N8, i1 false)
  ret void
}

declare void @llvm.lifetime.start.p5(i64 immarg, ptr addrspace(5) nocapture)
declare void @llvm.lifetime.end.p5(i64 immarg, ptr addrspace(5) nocapture)
declare void @llvm.memcpy.p0.p0.i32(ptr nocapture, ptr nocapture, i32, i1)
