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
define protected amdgpu_kernel void @t0(i32 addrspace(1)* %p.coerce) #0 {
entry:
  %p = alloca i32*, align 8, addrspace(5)
  %p.ascast = addrspacecast i32* addrspace(5)* %p to i32**
  %p.addr = alloca i32*, align 8, addrspace(5)
  %p.addr.ascast = addrspacecast i32* addrspace(5)* %p.addr to i32**
  %t = alloca [27 x i32], align 16, addrspace(5)
  %t.ascast = addrspacecast [27 x i32] addrspace(5)* %t to [27 x i32]*
  %sum = alloca i32, align 4, addrspace(5)
  %sum.ascast = addrspacecast i32 addrspace(5)* %sum to i32*
  %i = alloca i32, align 4, addrspace(5)
  %i.ascast = addrspacecast i32 addrspace(5)* %i to i32*
  %cleanup.dest.slot = alloca i32, align 4, addrspace(5)
  %0 = addrspacecast i32 addrspace(1)* %p.coerce to i32*
  store i32* %0, i32** %p.ascast, align 8
  %p1 = load i32*, i32** %p.ascast, align 8
  store i32* %p1, i32** %p.addr.ascast, align 8
  %1 = bitcast [27 x i32] addrspace(5)* %t to i8 addrspace(5)*
  call void @llvm.lifetime.start.p5i8(i64 48, i8 addrspace(5)* %1)
  %arraydecay = getelementptr inbounds [27 x i32], [27 x i32]* %t.ascast, i64 0, i64 0
  %2 = load i32*, i32** %p.addr.ascast, align 8
  call void @copy(i32* %arraydecay, i32* %2, i32 27)
  %3 = bitcast i32 addrspace(5)* %sum to i8 addrspace(5)*
  call void @llvm.lifetime.start.p5i8(i64 4, i8 addrspace(5)* %3)
  store i32 0, i32* %sum.ascast, align 4
  %4 = bitcast i32 addrspace(5)* %i to i8 addrspace(5)*
  call void @llvm.lifetime.start.p5i8(i64 4, i8 addrspace(5)* %4)
  store i32 0, i32* %i.ascast, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %5 = load i32, i32* %i.ascast, align 4
  %cmp = icmp slt i32 %5, 27
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  %6 = bitcast i32 addrspace(5)* %i to i8 addrspace(5)*
  call void @llvm.lifetime.end.p5i8(i64 4, i8 addrspace(5)* %6)
  br label %for.end

for.body:                                         ; preds = %for.cond
  %7 = load i32, i32* %i.ascast, align 4
  %idxprom = sext i32 %7 to i64
  %arrayidx = getelementptr inbounds [27 x i32], [27 x i32]* %t.ascast, i64 0, i64 %idxprom
  %8 = load i32, i32* %arrayidx, align 4
  %9 = load i32, i32* %sum.ascast, align 4
  %add = add nsw i32 %9, %8
  store i32 %add, i32* %sum.ascast, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %10 = load i32, i32* %i.ascast, align 4
  %inc = add nsw i32 %10, 1
  store i32 %inc, i32* %i.ascast, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond.cleanup
  %11 = load i32, i32* %sum.ascast, align 4
  %12 = load i32*, i32** %p.addr.ascast, align 8
  store i32 %11, i32* %12, align 4
  %13 = bitcast i32 addrspace(5)* %sum to i8 addrspace(5)*
  call void @llvm.lifetime.end.p5i8(i64 4, i8 addrspace(5)* %13)
  %14 = bitcast [27 x i32] addrspace(5)* %t to i8 addrspace(5)*
  call void @llvm.lifetime.end.p5i8(i64 48, i8 addrspace(5)* %14)
  ret void
}

define internal void @copy(i32* %d, i32* %s, i32 %N) {
entry:
  %d8 = bitcast i32* %d to i8*
  %s8 = bitcast i32* %s to i8*
  %N8 = mul i32 %N, 4
  tail call void @llvm.memcpy.p0i8.p0i8.i32(i8* %d8, i8* %s8, i32 %N8, i1 false)
  ret void
}

declare void @llvm.lifetime.start.p5i8(i64 immarg, i8 addrspace(5)* nocapture)
declare void @llvm.lifetime.end.p5i8(i64 immarg, i8 addrspace(5)* nocapture)
declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i1)
