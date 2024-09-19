;RUN: opt < %s -mtriple=amdgcn-amd-amdhsa -passes=asan -S | FileCheck %s

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"

declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)
declare void @llvm.memcpy.p0.p1.i64(ptr noalias nocapture writeonly, ptr addrspace(1) noalias nocapture readonly, i64, i1 immarg)
declare void @llvm.memcpy.p0.p2.i64(ptr noalias nocapture writeonly, ptr addrspace(2) noalias nocapture readonly, i64, i1 immarg)
declare void @llvm.memcpy.p0.p3.i64(ptr noalias nocapture writeonly, ptr addrspace(3) noalias nocapture readonly, i64, i1 immarg)
declare void @llvm.memcpy.p0.p4.i64(ptr noalias nocapture writeonly, ptr addrspace(4) noalias nocapture readonly, i64, i1 immarg)
declare void @llvm.memcpy.p0.p5.i64(ptr noalias nocapture writeonly, ptr addrspace(5) noalias nocapture readonly, i64, i1 immarg)

declare void @llvm.memcpy.p0.p0.i32(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i32, i1 immarg)
declare void @llvm.memcpy.p0.p1.i32(ptr noalias nocapture writeonly, ptr addrspace(1) noalias nocapture readonly, i32, i1 immarg)
declare void @llvm.memcpy.p0.p2.i32(ptr noalias nocapture writeonly, ptr addrspace(2) noalias nocapture readonly, i32, i1 immarg)
declare void @llvm.memcpy.p0.p3.i32(ptr noalias nocapture writeonly, ptr addrspace(3) noalias nocapture readonly, i32, i1 immarg)
declare void @llvm.memcpy.p0.p4.i32(ptr noalias nocapture writeonly, ptr addrspace(4) noalias nocapture readonly, i32, i1 immarg)
declare void @llvm.memcpy.p0.p5.i32(ptr noalias nocapture writeonly, ptr addrspace(5) noalias nocapture readonly, i32, i1 immarg)

declare void @llvm.memcpy.p1.p0.i64(ptr addrspace(1) noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)
declare void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) noalias nocapture writeonly, ptr addrspace(1) noalias nocapture readonly, i64, i1 immarg)
declare void @llvm.memcpy.p1.p2.i64(ptr addrspace(1) noalias nocapture writeonly, ptr addrspace(2) noalias nocapture readonly, i64, i1 immarg)
declare void @llvm.memcpy.p1.p3.i64(ptr addrspace(1) noalias nocapture writeonly, ptr addrspace(3) noalias nocapture readonly, i64, i1 immarg)
declare void @llvm.memcpy.p1.p4.i64(ptr addrspace(1) noalias nocapture writeonly, ptr addrspace(4) noalias nocapture readonly, i64, i1 immarg)
declare void @llvm.memcpy.p1.p5.i64(ptr addrspace(1) noalias nocapture writeonly, ptr addrspace(5) noalias nocapture readonly, i64, i1 immarg)

declare void @llvm.memcpy.p1.p0.i32(ptr addrspace(1) noalias nocapture writeonly, ptr noalias nocapture readonly, i32, i1 immarg)
declare void @llvm.memcpy.p1.p1.i32(ptr addrspace(1) noalias nocapture writeonly, ptr addrspace(1) noalias nocapture readonly, i32, i1 immarg)
declare void @llvm.memcpy.p1.p2.i32(ptr addrspace(1) noalias nocapture writeonly, ptr addrspace(2) noalias nocapture readonly, i32, i1 immarg)
declare void @llvm.memcpy.p1.p3.i32(ptr addrspace(1) noalias nocapture writeonly, ptr addrspace(3) noalias nocapture readonly, i32, i1 immarg)
declare void @llvm.memcpy.p1.p4.i32(ptr addrspace(1) noalias nocapture writeonly, ptr addrspace(4) noalias nocapture readonly, i32, i1 immarg)
declare void @llvm.memcpy.p1.p5.i32(ptr addrspace(1) noalias nocapture writeonly, ptr addrspace(5) noalias nocapture readonly, i32, i1 immarg)

declare void @llvm.memcpy.p2.p0.i64(ptr addrspace(2) noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)
declare void @llvm.memcpy.p2.p1.i64(ptr addrspace(2) noalias nocapture writeonly, ptr addrspace(1) noalias nocapture readonly, i64, i1 immarg)
declare void @llvm.memcpy.p2.p2.i64(ptr addrspace(2) noalias nocapture writeonly, ptr addrspace(2) noalias nocapture readonly, i64, i1 immarg)
declare void @llvm.memcpy.p2.p3.i64(ptr addrspace(2) noalias nocapture writeonly, ptr addrspace(3) noalias nocapture readonly, i64, i1 immarg)
declare void @llvm.memcpy.p2.p4.i64(ptr addrspace(2) noalias nocapture writeonly, ptr addrspace(4) noalias nocapture readonly, i64, i1 immarg)
declare void @llvm.memcpy.p2.p5.i64(ptr addrspace(2) noalias nocapture writeonly, ptr addrspace(5) noalias nocapture readonly, i64, i1 immarg)

declare void @llvm.memcpy.p2.p0.i32(ptr addrspace(2) noalias nocapture writeonly, ptr noalias nocapture readonly, i32, i1 immarg)
declare void @llvm.memcpy.p2.p1.i32(ptr addrspace(2) noalias nocapture writeonly, ptr addrspace(1) noalias nocapture readonly, i32, i1 immarg)
declare void @llvm.memcpy.p2.p2.i32(ptr addrspace(2) noalias nocapture writeonly, ptr addrspace(2) noalias nocapture readonly, i32, i1 immarg)
declare void @llvm.memcpy.p2.p3.i32(ptr addrspace(2) noalias nocapture writeonly, ptr addrspace(3) noalias nocapture readonly, i32, i1 immarg)
declare void @llvm.memcpy.p2.p4.i32(ptr addrspace(2) noalias nocapture writeonly, ptr addrspace(4) noalias nocapture readonly, i32, i1 immarg)
declare void @llvm.memcpy.p2.p5.i32(ptr addrspace(2) noalias nocapture writeonly, ptr addrspace(5) noalias nocapture readonly, i32, i1 immarg)

declare void @llvm.memcpy.p3.p0.i64(ptr addrspace(3) noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)
declare void @llvm.memcpy.p3.p1.i64(ptr addrspace(3) noalias nocapture writeonly, ptr addrspace(1) noalias nocapture readonly, i64, i1 immarg)
declare void @llvm.memcpy.p3.p2.i64(ptr addrspace(3) noalias nocapture writeonly, ptr addrspace(2) noalias nocapture readonly, i64, i1 immarg)
declare void @llvm.memcpy.p3.p3.i64(ptr addrspace(3) noalias nocapture writeonly, ptr addrspace(3) noalias nocapture readonly, i64, i1 immarg)
declare void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) noalias nocapture writeonly, ptr addrspace(4) noalias nocapture readonly, i64, i1 immarg)
declare void @llvm.memcpy.p3.p5.i64(ptr addrspace(3) noalias nocapture writeonly, ptr addrspace(5) noalias nocapture readonly, i64, i1 immarg)

declare void @llvm.memcpy.p3.p0.i32(ptr addrspace(3) noalias nocapture writeonly, ptr noalias nocapture readonly, i32, i1 immarg)
declare void @llvm.memcpy.p3.p1.i32(ptr addrspace(3) noalias nocapture writeonly, ptr addrspace(1) noalias nocapture readonly, i32, i1 immarg)
declare void @llvm.memcpy.p3.p2.i32(ptr addrspace(3) noalias nocapture writeonly, ptr addrspace(2) noalias nocapture readonly, i32, i1 immarg)
declare void @llvm.memcpy.p3.p3.i32(ptr addrspace(3) noalias nocapture writeonly, ptr addrspace(3) noalias nocapture readonly, i32, i1 immarg)
declare void @llvm.memcpy.p3.p4.i32(ptr addrspace(3) noalias nocapture writeonly, ptr addrspace(4) noalias nocapture readonly, i32, i1 immarg)
declare void @llvm.memcpy.p3.p5.i32(ptr addrspace(3) noalias nocapture writeonly, ptr addrspace(5) noalias nocapture readonly, i32, i1 immarg)

declare void @llvm.memcpy.p4.p0.i64(ptr addrspace(4) noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)
declare void @llvm.memcpy.p4.p1.i64(ptr addrspace(4) noalias nocapture writeonly, ptr addrspace(1) noalias nocapture readonly, i64, i1 immarg)
declare void @llvm.memcpy.p4.p2.i64(ptr addrspace(4) noalias nocapture writeonly, ptr addrspace(2) noalias nocapture readonly, i64, i1 immarg)
declare void @llvm.memcpy.p4.p3.i64(ptr addrspace(4) noalias nocapture writeonly, ptr addrspace(3) noalias nocapture readonly, i64, i1 immarg)
declare void @llvm.memcpy.p4.p4.i64(ptr addrspace(4) noalias nocapture writeonly, ptr addrspace(4) noalias nocapture readonly, i64, i1 immarg)
declare void @llvm.memcpy.p4.p5.i64(ptr addrspace(4) noalias nocapture writeonly, ptr addrspace(5) noalias nocapture readonly, i64, i1 immarg)

declare void @llvm.memcpy.p4.p0.i32(ptr addrspace(4) noalias nocapture writeonly, ptr noalias nocapture readonly, i32, i1 immarg)
declare void @llvm.memcpy.p4.p1.i32(ptr addrspace(4) noalias nocapture writeonly, ptr addrspace(1) noalias nocapture readonly, i32, i1 immarg)
declare void @llvm.memcpy.p4.p2.i32(ptr addrspace(4) noalias nocapture writeonly, ptr addrspace(2) noalias nocapture readonly, i32, i1 immarg)
declare void @llvm.memcpy.p4.p3.i32(ptr addrspace(4) noalias nocapture writeonly, ptr addrspace(3) noalias nocapture readonly, i32, i1 immarg)
declare void @llvm.memcpy.p4.p4.i32(ptr addrspace(4) noalias nocapture writeonly, ptr addrspace(4) noalias nocapture readonly, i32, i1 immarg)
declare void @llvm.memcpy.p4.p5.i32(ptr addrspace(4) noalias nocapture writeonly, ptr addrspace(5) noalias nocapture readonly, i32, i1 immarg)

declare void @llvm.memcpy.p5.p0.i64(ptr addrspace(5) noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)
declare void @llvm.memcpy.p5.p1.i64(ptr addrspace(5) noalias nocapture writeonly, ptr addrspace(1) noalias nocapture readonly, i64, i1 immarg)
declare void @llvm.memcpy.p5.p2.i64(ptr addrspace(5) noalias nocapture writeonly, ptr addrspace(2) noalias nocapture readonly, i64, i1 immarg)
declare void @llvm.memcpy.p5.p3.i64(ptr addrspace(5) noalias nocapture writeonly, ptr addrspace(3) noalias nocapture readonly, i64, i1 immarg)
declare void @llvm.memcpy.p5.p4.i64(ptr addrspace(5) noalias nocapture writeonly, ptr addrspace(4) noalias nocapture readonly, i64, i1 immarg)
declare void @llvm.memcpy.p5.p5.i64(ptr addrspace(5) noalias nocapture writeonly, ptr addrspace(5) noalias nocapture readonly, i64, i1 immarg)

declare void @llvm.memcpy.p5.p0.i32(ptr addrspace(5) noalias nocapture writeonly, ptr noalias nocapture readonly, i32, i1 immarg)
declare void @llvm.memcpy.p5.p1.i32(ptr addrspace(5) noalias nocapture writeonly, ptr addrspace(1) noalias nocapture readonly, i32, i1 immarg)
declare void @llvm.memcpy.p5.p2.i32(ptr addrspace(5) noalias nocapture writeonly, ptr addrspace(2) noalias nocapture readonly, i32, i1 immarg)
declare void @llvm.memcpy.p5.p3.i32(ptr addrspace(5) noalias nocapture writeonly, ptr addrspace(3) noalias nocapture readonly, i32, i1 immarg)
declare void @llvm.memcpy.p5.p4.i32(ptr addrspace(5) noalias nocapture writeonly, ptr addrspace(4) noalias nocapture readonly, i32, i1 immarg)
declare void @llvm.memcpy.p5.p5.i32(ptr addrspace(5) noalias nocapture writeonly, ptr addrspace(5) noalias nocapture readonly, i32, i1 immarg)

declare void @llvm.memmove.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1 immarg)
declare void @llvm.memmove.p0.p1.i64(ptr nocapture writeonly, ptr addrspace(1) nocapture readonly, i64, i1 immarg)
declare void @llvm.memmove.p0.p2.i64(ptr nocapture writeonly, ptr addrspace(2) nocapture readonly, i64, i1 immarg)
declare void @llvm.memmove.p0.p3.i64(ptr nocapture writeonly, ptr addrspace(3) nocapture readonly, i64, i1 immarg)
declare void @llvm.memmove.p0.p4.i64(ptr nocapture writeonly, ptr addrspace(4) nocapture readonly, i64, i1 immarg)
declare void @llvm.memmove.p0.p5.i64(ptr nocapture writeonly, ptr addrspace(5) nocapture readonly, i64, i1 immarg)

declare void @llvm.memmove.p0.p0.i32(ptr nocapture writeonly, ptr nocapture readonly, i32, i1 immarg)
declare void @llvm.memmove.p0.p1.i32(ptr nocapture writeonly, ptr addrspace(1) nocapture readonly, i32, i1 immarg)
declare void @llvm.memmove.p0.p2.i32(ptr nocapture writeonly, ptr addrspace(2) nocapture readonly, i32, i1 immarg)
declare void @llvm.memmove.p0.p3.i32(ptr nocapture writeonly, ptr addrspace(3) nocapture readonly, i32, i1 immarg)
declare void @llvm.memmove.p0.p4.i32(ptr nocapture writeonly, ptr addrspace(4) nocapture readonly, i32, i1 immarg)
declare void @llvm.memmove.p0.p5.i32(ptr nocapture writeonly, ptr addrspace(5) nocapture readonly, i32, i1 immarg)

declare void @llvm.memmove.p1.p0.i64(ptr addrspace(1) nocapture writeonly, ptr nocapture readonly, i64, i1 immarg)
declare void @llvm.memmove.p1.p1.i64(ptr addrspace(1) nocapture writeonly, ptr addrspace(1) nocapture readonly, i64, i1 immarg)
declare void @llvm.memmove.p1.p2.i64(ptr addrspace(1) nocapture writeonly, ptr addrspace(2) nocapture readonly, i64, i1 immarg)
declare void @llvm.memmove.p1.p3.i64(ptr addrspace(1) nocapture writeonly, ptr addrspace(3) nocapture readonly, i64, i1 immarg)
declare void @llvm.memmove.p1.p4.i64(ptr addrspace(1) nocapture writeonly, ptr addrspace(4) nocapture readonly, i64, i1 immarg)
declare void @llvm.memmove.p1.p5.i64(ptr addrspace(1) nocapture writeonly, ptr addrspace(5) nocapture readonly, i64, i1 immarg)

declare void @llvm.memmove.p1.p0.i32(ptr addrspace(1) nocapture writeonly, ptr nocapture readonly, i32, i1 immarg)
declare void @llvm.memmove.p1.p1.i32(ptr addrspace(1) nocapture writeonly, ptr addrspace(1) nocapture readonly, i32, i1 immarg)
declare void @llvm.memmove.p1.p2.i32(ptr addrspace(1) nocapture writeonly, ptr addrspace(2) nocapture readonly, i32, i1 immarg)
declare void @llvm.memmove.p1.p3.i32(ptr addrspace(1) nocapture writeonly, ptr addrspace(3) nocapture readonly, i32, i1 immarg)
declare void @llvm.memmove.p1.p4.i32(ptr addrspace(1) nocapture writeonly, ptr addrspace(4) nocapture readonly, i32, i1 immarg)
declare void @llvm.memmove.p1.p5.i32(ptr addrspace(1) nocapture writeonly, ptr addrspace(5) nocapture readonly, i32, i1 immarg)

declare void @llvm.memmove.p2.p0.i64(ptr addrspace(2) nocapture writeonly, ptr nocapture readonly, i64, i1 immarg)
declare void @llvm.memmove.p2.p1.i64(ptr addrspace(2) nocapture writeonly, ptr addrspace(1) nocapture readonly, i64, i1 immarg)
declare void @llvm.memmove.p2.p2.i64(ptr addrspace(2) nocapture writeonly, ptr addrspace(2) nocapture readonly, i64, i1 immarg)
declare void @llvm.memmove.p2.p3.i64(ptr addrspace(2) nocapture writeonly, ptr addrspace(3) nocapture readonly, i64, i1 immarg)
declare void @llvm.memmove.p2.p4.i64(ptr addrspace(2) nocapture writeonly, ptr addrspace(4) nocapture readonly, i64, i1 immarg)
declare void @llvm.memmove.p2.p5.i64(ptr addrspace(2) nocapture writeonly, ptr addrspace(5) nocapture readonly, i64, i1 immarg)

declare void @llvm.memmove.p2.p0.i32(ptr addrspace(2) nocapture writeonly, ptr nocapture readonly, i32, i1 immarg)
declare void @llvm.memmove.p2.p1.i32(ptr addrspace(2) nocapture writeonly, ptr addrspace(1) nocapture readonly, i32, i1 immarg)
declare void @llvm.memmove.p2.p2.i32(ptr addrspace(2) nocapture writeonly, ptr addrspace(2) nocapture readonly, i32, i1 immarg)
declare void @llvm.memmove.p2.p3.i32(ptr addrspace(2) nocapture writeonly, ptr addrspace(3) nocapture readonly, i32, i1 immarg)
declare void @llvm.memmove.p2.p4.i32(ptr addrspace(2) nocapture writeonly, ptr addrspace(4) nocapture readonly, i32, i1 immarg)
declare void @llvm.memmove.p2.p5.i32(ptr addrspace(2) nocapture writeonly, ptr addrspace(5) nocapture readonly, i32, i1 immarg)

declare void @llvm.memmove.p3.p0.i64(ptr addrspace(3) nocapture writeonly, ptr nocapture readonly, i64, i1 immarg)
declare void @llvm.memmove.p3.p1.i64(ptr addrspace(3) nocapture writeonly, ptr addrspace(1) nocapture readonly, i64, i1 immarg)
declare void @llvm.memmove.p3.p2.i64(ptr addrspace(3) nocapture writeonly, ptr addrspace(2) nocapture readonly, i64, i1 immarg)
declare void @llvm.memmove.p3.p3.i64(ptr addrspace(3) nocapture writeonly, ptr addrspace(3) nocapture readonly, i64, i1 immarg)
declare void @llvm.memmove.p3.p4.i64(ptr addrspace(3) nocapture writeonly, ptr addrspace(4) nocapture readonly, i64, i1 immarg)
declare void @llvm.memmove.p3.p5.i64(ptr addrspace(3) nocapture writeonly, ptr addrspace(5) nocapture readonly, i64, i1 immarg)

declare void @llvm.memmove.p3.p0.i32(ptr addrspace(3) nocapture writeonly, ptr nocapture readonly, i32, i1 immarg)
declare void @llvm.memmove.p3.p1.i32(ptr addrspace(3) nocapture writeonly, ptr addrspace(1) nocapture readonly, i32, i1 immarg)
declare void @llvm.memmove.p3.p2.i32(ptr addrspace(3) nocapture writeonly, ptr addrspace(2) nocapture readonly, i32, i1 immarg)
declare void @llvm.memmove.p3.p3.i32(ptr addrspace(3) nocapture writeonly, ptr addrspace(3) nocapture readonly, i32, i1 immarg)
declare void @llvm.memmove.p3.p4.i32(ptr addrspace(3) nocapture writeonly, ptr addrspace(4) nocapture readonly, i32, i1 immarg)
declare void @llvm.memmove.p3.p5.i32(ptr addrspace(3) nocapture writeonly, ptr addrspace(5) nocapture readonly, i32, i1 immarg)

declare void @llvm.memmove.p4.p0.i64(ptr addrspace(4) nocapture writeonly, ptr nocapture readonly, i64, i1 immarg)
declare void @llvm.memmove.p4.p1.i64(ptr addrspace(4) nocapture writeonly, ptr addrspace(1) nocapture readonly, i64, i1 immarg)
declare void @llvm.memmove.p4.p2.i64(ptr addrspace(4) nocapture writeonly, ptr addrspace(2) nocapture readonly, i64, i1 immarg)
declare void @llvm.memmove.p4.p3.i64(ptr addrspace(4) nocapture writeonly, ptr addrspace(3) nocapture readonly, i64, i1 immarg)
declare void @llvm.memmove.p4.p4.i64(ptr addrspace(4) nocapture writeonly, ptr addrspace(4) nocapture readonly, i64, i1 immarg)
declare void @llvm.memmove.p4.p5.i64(ptr addrspace(4) nocapture writeonly, ptr addrspace(5) nocapture readonly, i64, i1 immarg)

declare void @llvm.memmove.p4.p0.i32(ptr addrspace(4) nocapture writeonly, ptr nocapture readonly, i32, i1 immarg)
declare void @llvm.memmove.p4.p1.i32(ptr addrspace(4) nocapture writeonly, ptr addrspace(1) nocapture readonly, i32, i1 immarg)
declare void @llvm.memmove.p4.p2.i32(ptr addrspace(4) nocapture writeonly, ptr addrspace(2) nocapture readonly, i32, i1 immarg)
declare void @llvm.memmove.p4.p3.i32(ptr addrspace(4) nocapture writeonly, ptr addrspace(3) nocapture readonly, i32, i1 immarg)
declare void @llvm.memmove.p4.p4.i32(ptr addrspace(4) nocapture writeonly, ptr addrspace(4) nocapture readonly, i32, i1 immarg)
declare void @llvm.memmove.p4.p5.i32(ptr addrspace(4) nocapture writeonly, ptr addrspace(5) nocapture readonly, i32, i1 immarg)

declare void @llvm.memmove.p5.p0.i64(ptr addrspace(5) nocapture writeonly, ptr nocapture readonly, i64, i1 immarg)
declare void @llvm.memmove.p5.p1.i64(ptr addrspace(5) nocapture writeonly, ptr addrspace(1) nocapture readonly, i64, i1 immarg)
declare void @llvm.memmove.p5.p2.i64(ptr addrspace(5) nocapture writeonly, ptr addrspace(2) nocapture readonly, i64, i1 immarg)
declare void @llvm.memmove.p5.p3.i64(ptr addrspace(5) nocapture writeonly, ptr addrspace(3) nocapture readonly, i64, i1 immarg)
declare void @llvm.memmove.p5.p4.i64(ptr addrspace(5) nocapture writeonly, ptr addrspace(4) nocapture readonly, i64, i1 immarg)
declare void @llvm.memmove.p5.p5.i64(ptr addrspace(5) nocapture writeonly, ptr addrspace(5) nocapture readonly, i64, i1 immarg)

declare void @llvm.memmove.p5.p0.i32(ptr addrspace(5) nocapture writeonly, ptr nocapture readonly, i32, i1 immarg)
declare void @llvm.memmove.p5.p1.i32(ptr addrspace(5) nocapture writeonly, ptr addrspace(1) nocapture readonly, i32, i1 immarg)
declare void @llvm.memmove.p5.p2.i32(ptr addrspace(5) nocapture writeonly, ptr addrspace(2) nocapture readonly, i32, i1 immarg)
declare void @llvm.memmove.p5.p3.i32(ptr addrspace(5) nocapture writeonly, ptr addrspace(3) nocapture readonly, i32, i1 immarg)
declare void @llvm.memmove.p5.p4.i32(ptr addrspace(5) nocapture writeonly, ptr addrspace(4) nocapture readonly, i32, i1 immarg)
declare void @llvm.memmove.p5.p5.i32(ptr addrspace(5) nocapture writeonly, ptr addrspace(5) nocapture readonly, i32, i1 immarg)

declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg)
declare void @llvm.memset.p1.i64(ptr addrspace(1) nocapture writeonly, i8, i64, i1 immarg)
declare void @llvm.memset.p2.i64(ptr addrspace(2) nocapture writeonly, i8, i64, i1 immarg)
declare void @llvm.memset.p3.i64(ptr addrspace(3) nocapture writeonly, i8, i64, i1 immarg)
declare void @llvm.memset.p4.i64(ptr addrspace(4) nocapture writeonly, i8, i64, i1 immarg)
declare void @llvm.memset.p5.i64(ptr addrspace(5) nocapture writeonly, i8, i64, i1 immarg)

declare void @llvm.memset.p0.i32(ptr nocapture writeonly, i8, i32, i1 immarg)
declare void @llvm.memset.p1.i32(ptr addrspace(1) nocapture writeonly, i8, i32, i1 immarg)
declare void @llvm.memset.p2.i32(ptr addrspace(2) nocapture writeonly, i8, i32, i1 immarg)
declare void @llvm.memset.p3.i32(ptr addrspace(3) nocapture writeonly, i8, i32, i1 immarg)
declare void @llvm.memset.p4.i32(ptr addrspace(4) nocapture writeonly, i8, i32, i1 immarg)
declare void @llvm.memset.p5.i32(ptr addrspace(5) nocapture writeonly, i8, i32, i1 immarg)

define void @test_mem_intrinsic_memcpy(ptr %dest0,ptr %src0,ptr addrspace(1) %dest1,ptr addrspace(1) %src1,ptr addrspace(2) %dest2,ptr addrspace(2) %src2,ptr addrspace(3) %dest3,ptr addrspace(3) %src3,ptr addrspace(4) %dest4,ptr addrspace(4) %src4,ptr addrspace(5) %dest5,ptr addrspace(5) %src5) #0 {
entry:
  ;CHECK: define void @test_mem_intrinsic_memcpy(ptr [[DEST0:%.*]], ptr [[SRC0:%.*]], ptr addrspace(1) [[DEST1:%.*]], ptr addrspace(1) [[SRC1:%.*]], ptr addrspace(2) [[DEST2:%.*]], ptr addrspace(2) [[SRC2:%.*]], ptr addrspace(3) [[DEST3:%.*]], ptr addrspace(3) [[SRC3:%.*]], ptr addrspace(4) [[DEST4:%.*]], ptr addrspace(4) [[SRC4:%.*]], ptr addrspace(5) [[DEST5:%.*]], ptr addrspace(5) [[SRC5:%.*]]) #2 {
  ;CHECK-NEXT: entry:
  ;CHECK-NEXT: [[VR0:%.*]] = call ptr @__asan_memcpy(ptr [[DEST0]], ptr [[SRC0]], i64 64)
  ;CHECK-NEXT: [[VR1:%.*]] = addrspacecast ptr addrspace(1) [[SRC1]] to ptr
  ;CHECK-NEXT: [[VR2:%.*]] = call ptr @__asan_memcpy(ptr [[DEST0]], ptr [[VR1]], i64 64)
  ;CHECK-NEXT: [[VR3:%.*]] = addrspacecast ptr addrspace(2) [[SRC2]] to ptr
  ;CHECK-NEXT: [[VR4:%.*]] = call ptr @__asan_memcpy(ptr [[DEST0]], ptr [[VR3]], i64 64)
  ;CHECK-NEXT: [[VR5:%.*]] = addrspacecast ptr addrspace(3) [[SRC3]] to ptr
  ;CHECK-NEXT: [[VR6:%.*]] = call ptr @__asan_memcpy(ptr [[DEST0]], ptr [[VR5]], i64 64)
  ;CHECK-NEXT: [[VR7:%.*]] = addrspacecast ptr addrspace(4) [[SRC4]] to ptr
  ;CHECK-NEXT: [[VR8:%.*]] = call ptr @__asan_memcpy(ptr [[DEST0]], ptr [[VR7]], i64 64)
  ;CHECK-NEXT: [[VR9:%.*]] = addrspacecast ptr addrspace(5) [[SRC5]] to ptr
  ;CHECK-NEXT: [[VR10:%.*]] = call ptr @__asan_memcpy(ptr [[DEST0]], ptr [[VR9]], i64 64)
  ;CHECK-NEXT: [[VR11:%.*]] = call ptr @__asan_memcpy(ptr [[DEST0]], ptr [[SRC0]], i64 32)
  ;CHECK-NEXT: [[VR12:%.*]] = addrspacecast ptr addrspace(1) [[SRC1]] to ptr
  ;CHECK-NEXT: [[VR13:%.*]] = call ptr @__asan_memcpy(ptr [[DEST0]], ptr [[VR12]], i64 32)
  ;CHECK-NEXT: [[VR14:%.*]] = addrspacecast ptr addrspace(2) [[SRC2]] to ptr
  ;CHECK-NEXT: [[VR15:%.*]] = call ptr @__asan_memcpy(ptr [[DEST0]], ptr [[VR14]], i64 32)
  ;CHECK-NEXT: [[VR16:%.*]] = addrspacecast ptr addrspace(3) [[SRC3]] to ptr
  ;CHECK-NEXT: [[VR17:%.*]] = call ptr @__asan_memcpy(ptr [[DEST0]], ptr [[VR16]], i64 32)
  ;CHECK-NEXT: [[VR18:%.*]] = addrspacecast ptr addrspace(4) [[SRC4]] to ptr
  ;CHECK-NEXT: [[VR19:%.*]] = call ptr @__asan_memcpy(ptr [[DEST0]], ptr %18, i64 32)
  ;CHECK-NEXT: [[VR20:%.*]] = addrspacecast ptr addrspace(5) [[SRC5]] to ptr
  ;CHECK-NEXT: [[VR21:%.*]] = call ptr @__asan_memcpy(ptr [[DEST0]], ptr [[VR20]], i64 32)
  ;CHECK-NEXT: [[VR22:%.*]] = addrspacecast ptr addrspace(1) [[DEST1]] to ptr
  ;CHECK-NEXT: [[VR23:%.*]] = call ptr @__asan_memcpy(ptr [[VR22]], ptr [[SRC0]], i64 64)
  ;CHECK-NEXT: [[VR24:%.*]] = addrspacecast ptr addrspace(1) [[DEST1]] to ptr
  ;CHECK-NEXT: [[VR25:%.*]] = addrspacecast ptr addrspace(1) [[SRC1]] to ptr
  ;CHECK-NEXT: [[VR26:%.*]] = call ptr @__asan_memcpy(ptr [[VR24]], ptr [[VR25]], i64 64)
  ;CHECK-NEXT: [[VR27:%.*]] = addrspacecast ptr addrspace(1) [[DEST1]] to ptr
  ;CHECK-NEXT: [[VR28:%.*]] = addrspacecast ptr addrspace(2) [[SRC2]] to ptr
  ;CHECK-NEXT: [[VR29:%.*]] = call ptr @__asan_memcpy(ptr [[VR27]], ptr [[VR28]], i64 64)
  ;CHECK-NEXT: [[VR30:%.*]] = addrspacecast ptr addrspace(1) [[DEST1]] to ptr
  ;CHECK-NEXT: [[VR31:%.*]] = addrspacecast ptr addrspace(3) [[SRC3]] to ptr
  ;CHECK-NEXT: [[VR32:%.*]] = call ptr @__asan_memcpy(ptr [[VR30]], ptr [[VR31]], i64 64)
  ;CHECK-NEXT: [[VR33:%.*]] = addrspacecast ptr addrspace(1) [[DEST1]] to ptr
  ;CHECK-NEXT: [[VR34:%.*]] = addrspacecast ptr addrspace(4) [[SRC4]] to ptr
  ;CHECK-NEXT: [[VR35:%.*]] = call ptr @__asan_memcpy(ptr [[VR33]], ptr [[VR34]], i64 64)
  ;CHECK-NEXT: [[VR36:%.*]] = addrspacecast ptr addrspace(1) [[DEST1]] to ptr
  ;CHECK-NEXT: [[VR37:%.*]] = addrspacecast ptr addrspace(5) [[SRC5]] to ptr
  ;CHECK-NEXT: [[VR38:%.*]] = call ptr @__asan_memcpy(ptr [[VR36]], ptr [[VR37]], i64 64)
  ;CHECK-NEXT: [[VR39:%.*]] = addrspacecast ptr addrspace(1) [[DEST1]] to ptr
  ;CHECK-NEXT: [[VR40:%.*]] = call ptr @__asan_memcpy(ptr [[VR39]], ptr [[SRC0]], i64 32)
  ;CHECK-NEXT: [[VR41:%.*]] = addrspacecast ptr addrspace(1) [[DEST1]] to ptr
  ;CHECK-NEXT: [[VR42:%.*]] = addrspacecast ptr addrspace(1) [[SRC1]] to ptr
  ;CHECK-NEXT: [[VR43:%.*]] = call ptr @__asan_memcpy(ptr [[VR41]], ptr [[VR42]], i64 32)
  ;CHECK-NEXT: [[VR44:%.*]] = addrspacecast ptr addrspace(1) [[DEST1]] to ptr
  ;CHECK-NEXT: [[VR45:%.*]] = addrspacecast ptr addrspace(2) [[SRC2]] to ptr
  ;CHECK-NEXT: [[VR46:%.*]] = call ptr @__asan_memcpy(ptr [[VR44]], ptr [[VR45]], i64 32)
  ;CHECK-NEXT: [[VR47:%.*]] = addrspacecast ptr addrspace(1) [[DEST1]] to ptr
  ;CHECK-NEXT: [[VR48:%.*]] = addrspacecast ptr addrspace(3) [[SRC3]] to ptr
  ;CHECK-NEXT: [[VR49:%.*]] = call ptr @__asan_memcpy(ptr [[VR47]], ptr [[VR48]], i64 32)
  ;CHECK-NEXT: [[VR50:%.*]] = addrspacecast ptr addrspace(1) [[DEST1]] to ptr
  ;CHECK-NEXT: [[VR51:%.*]] = addrspacecast ptr addrspace(4) [[SRC4]] to ptr
  ;CHECK-NEXT: [[VR52:%.*]] = call ptr @__asan_memcpy(ptr [[VR50]], ptr [[VR51]], i64 32)
  ;CHECK-NEXT: [[VR53:%.*]] = addrspacecast ptr addrspace(1) [[DEST1]] to ptr
  ;CHECK-NEXT: [[VR54:%.*]] = addrspacecast ptr addrspace(5) [[SRC5]] to ptr
  ;CHECK-NEXT: [[VR55:%.*]] = call ptr @__asan_memcpy(ptr [[VR53]], ptr [[VR54]], i64 32)
  ;CHECK-NEXT: [[VR56:%.*]] = addrspacecast ptr addrspace(2) [[DEST2]] to ptr
  ;CHECK-NEXT: [[VR57:%.*]] = call ptr @__asan_memcpy(ptr [[VR56]], ptr [[SRC0]], i64 64)
  ;CHECK-NEXT: [[VR58:%.*]] = addrspacecast ptr addrspace(2) [[DEST2]] to ptr
  ;CHECK-NEXT: [[VR59:%.*]] = addrspacecast ptr addrspace(1) [[SRC1]] to ptr
  ;CHECK-NEXT: [[VR60:%.*]] = call ptr @__asan_memcpy(ptr [[VR58]], ptr [[VR59]], i64 64)
  ;CHECK-NEXT: [[VR61:%.*]] = addrspacecast ptr addrspace(2) [[DEST2]] to ptr
  ;CHECK-NEXT: [[VR62:%.*]] = addrspacecast ptr addrspace(2) [[SRC2]] to ptr
  ;CHECK-NEXT: [[VR63:%.*]] = call ptr @__asan_memcpy(ptr [[VR61]], ptr [[VR62]], i64 64)
  ;CHECK-NEXT: [[VR64:%.*]] = addrspacecast ptr addrspace(2) [[DEST2]] to ptr
  ;CHECK-NEXT: [[VR65:%.*]] = addrspacecast ptr addrspace(3) [[SRC3]] to ptr
  ;CHECK-NEXT: [[VR66:%.*]] = call ptr @__asan_memcpy(ptr [[VR64]], ptr [[VR65]], i64 64)
  ;CHECK-NEXT: [[VR67:%.*]] = addrspacecast ptr addrspace(2) [[DEST2]] to ptr
  ;CHECK-NEXT: [[VR68:%.*]] = addrspacecast ptr addrspace(4) [[SRC4]] to ptr
  ;CHECK-NEXT: [[VR69:%.*]] = call ptr @__asan_memcpy(ptr [[VR67]], ptr [[VR68]], i64 64)
  ;CHECK-NEXT: [[VR70:%.*]] = addrspacecast ptr addrspace(2) [[DEST2]] to ptr
  ;CHECK-NEXT: [[VR71:%.*]] = addrspacecast ptr addrspace(5) [[SRC5]] to ptr
  ;CHECK-NEXT: [[VR72:%.*]] = call ptr @__asan_memcpy(ptr [[VR70]], ptr [[VR71]], i64 64)
  ;CHECK-NEXT: [[VR73:%.*]] = addrspacecast ptr addrspace(2) [[DEST2]] to ptr
  ;CHECK-NEXT: [[VR74:%.*]] = call ptr @__asan_memcpy(ptr [[VR73]], ptr [[SRC0]], i64 32)
  ;CHECK-NEXT: [[VR75:%.*]] = addrspacecast ptr addrspace(2) [[DEST2]] to ptr
  ;CHECK-NEXT: [[VR76:%.*]] = addrspacecast ptr addrspace(1) [[SRC1]] to ptr
  ;CHECK-NEXT: [[VR77:%.*]] = call ptr @__asan_memcpy(ptr [[VR75]], ptr [[VR76]], i64 32)
  ;CHECK-NEXT: [[VR78:%.*]] = addrspacecast ptr addrspace(2) [[DEST2]] to ptr
  ;CHECK-NEXT: [[VR79:%.*]] = addrspacecast ptr addrspace(2) [[SRC2]] to ptr
  ;CHECK-NEXT: [[VR80:%.*]] = call ptr @__asan_memcpy(ptr [[VR78]], ptr [[VR79]], i64 32)
  ;CHECK-NEXT: [[VR81:%.*]] = addrspacecast ptr addrspace(2) [[DEST2]] to ptr
  ;CHECK-NEXT: [[VR82:%.*]] = addrspacecast ptr addrspace(3) [[SRC3]] to ptr
  ;CHECK-NEXT: [[VR83:%.*]] = call ptr @__asan_memcpy(ptr [[VR81]], ptr [[VR82]], i64 32)
  ;CHECK-NEXT: [[VR84:%.*]] = addrspacecast ptr addrspace(2) [[DEST2]] to ptr
  ;CHECK-NEXT: [[VR85:%.*]] = addrspacecast ptr addrspace(4) [[SRC4]] to ptr
  ;CHECK-NEXT: [[VR86:%.*]] = call ptr @__asan_memcpy(ptr [[VR84]], ptr [[VR85]], i64 32)
  ;CHECK-NEXT: [[VR87:%.*]] = addrspacecast ptr addrspace(2) [[DEST2]] to ptr
  ;CHECK-NEXT: [[VR88:%.*]] = addrspacecast ptr addrspace(5) [[SRC5]] to ptr
  ;CHECK-NEXT: [[VR89:%.*]] = call ptr @__asan_memcpy(ptr [[VR87]], ptr [[VR88]], i64 32)
  ;CHECK-NEXT: [[VR90:%.*]] = addrspacecast ptr addrspace(3) [[DEST3]] to ptr
  ;CHECK-NEXT: [[VR91:%.*]] = call ptr @__asan_memcpy(ptr [[VR90]], ptr [[SRC0]], i64 64)
  ;CHECK-NEXT: [[VR92:%.*]] = addrspacecast ptr addrspace(3) [[DEST3]] to ptr
  ;CHECK-NEXT: [[VR93:%.*]] = addrspacecast ptr addrspace(1) [[SRC1]] to ptr
  ;CHECK-NEXT: [[VR94:%.*]] = call ptr @__asan_memcpy(ptr [[VR92]], ptr [[VR93]], i64 64)
  ;CHECK-NEXT: [[VR95:%.*]] = addrspacecast ptr addrspace(3) [[DEST3]] to ptr
  ;CHECK-NEXT: [[VR96:%.*]] = addrspacecast ptr addrspace(2) [[SRC2]] to ptr
  ;CHECK-NEXT: [[VR97:%.*]] = call ptr @__asan_memcpy(ptr [[VR95]], ptr [[VR96]], i64 64)
  ;CHECK-NEXT: [[VR98:%.*]] = addrspacecast ptr addrspace(3) [[DEST3]] to ptr
  ;CHECK-NEXT: [[VR99:%.*]] = addrspacecast ptr addrspace(3) [[SRC3]] to ptr
  ;CHECK-NEXT: [[VR100:%.*]] = call ptr @__asan_memcpy(ptr [[VR98]], ptr [[VR99]], i64 64)
  ;CHECK-NEXT: [[VR101:%.*]] = addrspacecast ptr addrspace(3) [[DEST3]] to ptr
  ;CHECK-NEXT: [[VR102:%.*]] = addrspacecast ptr addrspace(4) [[SRC4]] to ptr
  ;CHECK-NEXT: [[VR103:%.*]] = call ptr @__asan_memcpy(ptr [[VR101]], ptr [[VR102]], i64 64)
  ;CHECK-NEXT: [[VR104:%.*]] = addrspacecast ptr addrspace(3) [[DEST3]] to ptr
  ;CHECK-NEXT: [[VR105:%.*]] = addrspacecast ptr addrspace(5) [[SRC5]] to ptr
  ;CHECK-NEXT: [[VR106:%.*]] = call ptr @__asan_memcpy(ptr [[VR104]], ptr [[VR105]], i64 64)
  ;CHECK-NEXT: [[VR107:%.*]] = addrspacecast ptr addrspace(3) [[DEST3]] to ptr
  ;CHECK-NEXT: [[VR108:%.*]] = call ptr @__asan_memcpy(ptr [[VR107]], ptr [[SRC0]], i64 32)
  ;CHECK-NEXT: [[VR109:%.*]] = addrspacecast ptr addrspace(3) [[DEST3]] to ptr
  ;CHECK-NEXT: [[VR110:%.*]] = addrspacecast ptr addrspace(1) [[SRC1]] to ptr
  ;CHECK-NEXT: [[VR111:%.*]] = call ptr @__asan_memcpy(ptr [[VR109]], ptr [[VR110]], i64 32)
  ;CHECK-NEXT: [[VR112:%.*]] = addrspacecast ptr addrspace(3) [[DEST3]] to ptr
  ;CHECK-NEXT: [[VR113:%.*]] = addrspacecast ptr addrspace(2) [[SRC2]] to ptr
  ;CHECK-NEXT: [[VR114:%.*]] = call ptr @__asan_memcpy(ptr [[VR112]], ptr [[VR113]], i64 32)
  ;CHECK-NEXT: [[VR115:%.*]] = addrspacecast ptr addrspace(3) [[DEST3]] to ptr
  ;CHECK-NEXT: [[VR116:%.*]] = addrspacecast ptr addrspace(3) [[SRC3]] to ptr
  ;CHECK-NEXT: [[VR117:%.*]] = call ptr @__asan_memcpy(ptr [[VR115]], ptr [[VR116]], i64 32)
  ;CHECK-NEXT: [[VR118:%.*]] = addrspacecast ptr addrspace(3) [[DEST3]] to ptr
  ;CHECK-NEXT: [[VR119:%.*]] = addrspacecast ptr addrspace(4) [[SRC4]] to ptr
  ;CHECK-NEXT: [[VR120:%.*]] = call ptr @__asan_memcpy(ptr [[VR118]], ptr [[VR119]], i64 32)
  ;CHECK-NEXT: [[VR121:%.*]] = addrspacecast ptr addrspace(3) [[DEST3]] to ptr
  ;CHECK-NEXT: [[VR122:%.*]] = addrspacecast ptr addrspace(5) [[SRC5]] to ptr
  ;CHECK-NEXT: [[VR123:%.*]] = call ptr @__asan_memcpy(ptr [[VR121]], ptr [[VR122]], i64 32)
  ;CHECK-NEXT: [[VR124:%.*]] = addrspacecast ptr addrspace(4) [[DEST4]] to ptr
  ;CHECK-NEXT: [[VR125:%.*]] = call ptr @__asan_memcpy(ptr [[VR124]], ptr [[SRC0]], i64 64)
  ;CHECK-NEXT: [[VR126:%.*]] = addrspacecast ptr addrspace(4) [[DEST4]] to ptr
  ;CHECK-NEXT: [[VR127:%.*]] = addrspacecast ptr addrspace(1) [[SRC1]] to ptr
  ;CHECK-NEXT: [[VR128:%.*]] = call ptr @__asan_memcpy(ptr [[VR126]], ptr [[VR127]], i64 64)
  ;CHECK-NEXT: [[VR129:%.*]] = addrspacecast ptr addrspace(4) [[DEST4]] to ptr
  ;CHECK-NEXT: [[VR130:%.*]] = addrspacecast ptr addrspace(2) [[SRC2]] to ptr
  ;CHECK-NEXT: [[VR131:%.*]] = call ptr @__asan_memcpy(ptr [[VR129]], ptr [[VR130]], i64 64)
  ;CHECK-NEXT: [[VR132:%.*]] = addrspacecast ptr addrspace(4) [[DEST4]] to ptr
  ;CHECK-NEXT: [[VR133:%.*]] = addrspacecast ptr addrspace(3) [[SRC3]] to ptr
  ;CHECK-NEXT: [[VR134:%.*]] = call ptr @__asan_memcpy(ptr [[VR132]], ptr [[VR133]], i64 64)
  ;CHECK-NEXT: [[VR135:%.*]] = addrspacecast ptr addrspace(4) [[DEST4]] to ptr
  ;CHECK-NEXT: [[VR136:%.*]] = addrspacecast ptr addrspace(4) [[SRC4]] to ptr
  ;CHECK-NEXT: [[VR137:%.*]] = call ptr @__asan_memcpy(ptr [[VR135]], ptr [[VR136]], i64 64)
  ;CHECK-NEXT: [[VR138:%.*]] = addrspacecast ptr addrspace(4) [[DEST4]] to ptr
  ;CHECK-NEXT: [[VR139:%.*]] = addrspacecast ptr addrspace(5) [[SRC5]] to ptr
  ;CHECK-NEXT: [[VR140:%.*]] = call ptr @__asan_memcpy(ptr [[VR138]], ptr [[VR139]], i64 64)
  ;CHECK-NEXT: [[VR141:%.*]] = addrspacecast ptr addrspace(4) [[DEST4]] to ptr
  ;CHECK-NEXT: [[VR142:%.*]] = call ptr @__asan_memcpy(ptr [[VR141]], ptr [[SRC0]], i64 32)
  ;CHECK-NEXT: [[VR143:%.*]] = addrspacecast ptr addrspace(4) [[DEST4]] to ptr
  ;CHECK-NEXT: [[VR144:%.*]] = addrspacecast ptr addrspace(1) [[SRC1]] to ptr
  ;CHECK-NEXT: [[VR145:%.*]] = call ptr @__asan_memcpy(ptr [[VR143]], ptr [[VR144]], i64 32)
  ;CHECK-NEXT: [[VR146:%.*]] = addrspacecast ptr addrspace(4) [[DEST4]] to ptr
  ;CHECK-NEXT: [[VR147:%.*]] = addrspacecast ptr addrspace(2) [[SRC2]] to ptr
  ;CHECK-NEXT: [[VR148:%.*]] = call ptr @__asan_memcpy(ptr [[VR146]], ptr [[VR147]], i64 32)
  ;CHECK-NEXT: [[VR149:%.*]] = addrspacecast ptr addrspace(4) [[DEST4]] to ptr
  ;CHECK-NEXT: [[VR150:%.*]] = addrspacecast ptr addrspace(3) [[SRC3]] to ptr
  ;CHECK-NEXT: [[VR151:%.*]] = call ptr @__asan_memcpy(ptr [[VR149]], ptr [[VR150]], i64 32)
  ;CHECK-NEXT: [[VR152:%.*]] = addrspacecast ptr addrspace(4) [[DEST4]] to ptr
  ;CHECK-NEXT: [[VR153:%.*]] = addrspacecast ptr addrspace(4) [[SRC4]] to ptr
  ;CHECK-NEXT: [[VR154:%.*]] = call ptr @__asan_memcpy(ptr [[VR152]], ptr [[VR153]], i64 32)
  ;CHECK-NEXT: [[VR155:%.*]] = addrspacecast ptr addrspace(4) [[DEST4]] to ptr
  ;CHECK-NEXT: [[VR156:%.*]] = addrspacecast ptr addrspace(5) [[SRC5]] to ptr
  ;CHECK-NEXT: [[VR157:%.*]] = call ptr @__asan_memcpy(ptr [[VR155]], ptr [[VR156]], i64 32)
  ;CHECK-NEXT: [[VR158:%.*]] = addrspacecast ptr addrspace(5) [[DEST5]] to ptr
  ;CHECK-NEXT: [[VR159:%.*]] = call ptr @__asan_memcpy(ptr [[VR158]], ptr [[SRC0]], i64 64)
  ;CHECK-NEXT: [[VR160:%.*]] = addrspacecast ptr addrspace(5) [[DEST5]] to ptr
  ;CHECK-NEXT: [[VR161:%.*]] = addrspacecast ptr addrspace(1) [[SRC1]] to ptr
  ;CHECK-NEXT: [[VR162:%.*]] = call ptr @__asan_memcpy(ptr [[VR160]], ptr [[VR161]], i64 64)
  ;CHECK-NEXT: [[VR163:%.*]] = addrspacecast ptr addrspace(5) [[DEST5]] to ptr
  ;CHECK-NEXT: [[VR164:%.*]] = addrspacecast ptr addrspace(2) [[SRC2]] to ptr
  ;CHECK-NEXT: [[VR165:%.*]] = call ptr @__asan_memcpy(ptr [[VR163]], ptr [[VR164]], i64 64)
  ;CHECK-NEXT: [[VR166:%.*]] = addrspacecast ptr addrspace(5) [[DEST5]] to ptr
  ;CHECK-NEXT: [[VR167:%.*]] = addrspacecast ptr addrspace(3) [[SRC3]] to ptr
  ;CHECK-NEXT: [[VR168:%.*]] = call ptr @__asan_memcpy(ptr [[VR166]], ptr [[VR167]], i64 64)
  ;CHECK-NEXT: [[VR169:%.*]] = addrspacecast ptr addrspace(5) [[DEST5]] to ptr
  ;CHECK-NEXT: [[VR170:%.*]] = addrspacecast ptr addrspace(4) [[SRC4]] to ptr
  ;CHECK-NEXT: [[VR171:%.*]] = call ptr @__asan_memcpy(ptr [[VR169]], ptr [[VR170]], i64 64)
  ;CHECK-NEXT: [[VR172:%.*]] = addrspacecast ptr addrspace(5) [[DEST5]] to ptr
  ;CHECK-NEXT: [[VR173:%.*]] = addrspacecast ptr addrspace(5) [[SRC5]] to ptr
  ;CHECK-NEXT: [[VR174:%.*]] = call ptr @__asan_memcpy(ptr [[VR172]], ptr [[VR173]], i64 64)
  ;CHECK-NEXT: [[VR175:%.*]] = addrspacecast ptr addrspace(5) [[DEST5]] to ptr
  ;CHECK-NEXT: [[VR176:%.*]] = call ptr @__asan_memcpy(ptr %175, ptr %src0, i64 32)
  ;CHECK-NEXT: [[VR177:%.*]] = addrspacecast ptr addrspace(5) [[DEST5]] to ptr
  ;CHECK-NEXT: [[VR178:%.*]] = addrspacecast ptr addrspace(1) [[SRC1]] to ptr
  ;CHECK-NEXT: [[VR179:%.*]] = call ptr @__asan_memcpy(ptr [[VR177]], ptr [[VR178]], i64 32)
  ;CHECK-NEXT: [[VR180:%.*]] = addrspacecast ptr addrspace(5) [[DEST5]] to ptr
  ;CHECK-NEXT: [[VR181:%.*]] = addrspacecast ptr addrspace(2) [[SRC2]] to ptr
  ;CHECK-NEXT: [[VR182:%.*]] = call ptr @__asan_memcpy(ptr [[VR180]], ptr [[VR181]], i64 32)
  ;CHECK-NEXT: [[VR183:%.*]] = addrspacecast ptr addrspace(5) [[DEST5]] to ptr
  ;CHECK-NEXT: [[VR184:%.*]] = addrspacecast ptr addrspace(3) [[SRC3]] to ptr
  ;CHECK-NEXT: [[VR185:%.*]] = call ptr @__asan_memcpy(ptr [[VR183]], ptr [[VR184]], i64 32)
  ;CHECK-NEXT: [[VR186:%.*]] = addrspacecast ptr addrspace(5) [[DEST5]] to ptr
  ;CHECK-NEXT: [[VR187:%.*]] = addrspacecast ptr addrspace(4) [[SRC4]] to ptr
  ;CHECK-NEXT: [[VR188:%.*]] = call ptr @__asan_memcpy(ptr [[VR186]], ptr [[VR187]], i64 32)
  ;CHECK-NEXT: [[VR189:%.*]] = addrspacecast ptr addrspace(5) [[DEST5]] to ptr
  ;CHECK-NEXT: [[VR190:%.*]] = addrspacecast ptr addrspace(5) [[SRC5]] to ptr
  ;CHECK-NEXT: [[VR191:%.*]] = call ptr @__asan_memcpy(ptr [[VR189]], ptr [[VR190]], i64 32)

  call void @llvm.memcpy.p0.p0.i64(ptr align 16 %dest0, ptr align 16 %src0, i64 64, i1 false)
  call void @llvm.memcpy.p0.p1.i64(ptr align 16 %dest0, ptr addrspace(1) align 16 %src1, i64 64, i1 false)
  call void @llvm.memcpy.p0.p2.i64(ptr align 16 %dest0, ptr addrspace(2) align 16 %src2, i64 64, i1 false)
  call void @llvm.memcpy.p0.p3.i64(ptr align 16 %dest0, ptr addrspace(3) align 16 %src3, i64 64, i1 false)
  call void @llvm.memcpy.p0.p4.i64(ptr align 16 %dest0, ptr addrspace(4) align 16 %src4, i64 64, i1 false)
  call void @llvm.memcpy.p0.p5.i64(ptr align 16 %dest0, ptr addrspace(5) align 16 %src5, i64 64, i1 false)

  call void @llvm.memcpy.p0.p0.i32(ptr align 16 %dest0, ptr align 16 %src0, i32 32, i1 false)
  call void @llvm.memcpy.p0.p1.i32(ptr align 16 %dest0, ptr addrspace(1) align 16 %src1, i32 32, i1 false)
  call void @llvm.memcpy.p0.p2.i32(ptr align 16 %dest0, ptr addrspace(2) align 16 %src2, i32 32, i1 false)
  call void @llvm.memcpy.p0.p3.i32(ptr align 16 %dest0, ptr addrspace(3) align 16 %src3, i32 32, i1 false)
  call void @llvm.memcpy.p0.p4.i32(ptr align 16 %dest0, ptr addrspace(4) align 16 %src4, i32 32, i1 false)
  call void @llvm.memcpy.p0.p5.i32(ptr align 16 %dest0, ptr addrspace(5) align 16 %src5, i32 32, i1 false)

  call void @llvm.memcpy.p1.p0.i64(ptr addrspace(1) align 16 %dest1, ptr align 16 %src0, i64 64, i1 false)
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) align 16 %dest1, ptr addrspace(1) align 16 %src1, i64 64, i1 false)
  call void @llvm.memcpy.p1.p2.i64(ptr addrspace(1) align 16 %dest1, ptr addrspace(2) align 16 %src2, i64 64, i1 false)
  call void @llvm.memcpy.p1.p3.i64(ptr addrspace(1) align 16 %dest1, ptr addrspace(3) align 16 %src3, i64 64, i1 false)
  call void @llvm.memcpy.p1.p4.i64(ptr addrspace(1) align 16 %dest1, ptr addrspace(4) align 16 %src4, i64 64, i1 false)
  call void @llvm.memcpy.p1.p5.i64(ptr addrspace(1) align 16 %dest1, ptr addrspace(5) align 16 %src5, i64 64, i1 false)

  call void @llvm.memcpy.p1.p0.i32(ptr addrspace(1) align 16 %dest1, ptr align 16 %src0, i32 32, i1 false)
  call void @llvm.memcpy.p1.p1.i32(ptr addrspace(1) align 16 %dest1, ptr addrspace(1) align 16 %src1, i32 32, i1 false)
  call void @llvm.memcpy.p1.p2.i32(ptr addrspace(1) align 16 %dest1, ptr addrspace(2) align 16 %src2, i32 32, i1 false)
  call void @llvm.memcpy.p1.p3.i32(ptr addrspace(1) align 16 %dest1, ptr addrspace(3) align 16 %src3, i32 32, i1 false)
  call void @llvm.memcpy.p1.p4.i32(ptr addrspace(1) align 16 %dest1, ptr addrspace(4) align 16 %src4, i32 32, i1 false)
  call void @llvm.memcpy.p1.p5.i32(ptr addrspace(1) align 16 %dest1, ptr addrspace(5) align 16 %src5, i32 32, i1 false)

  call void @llvm.memcpy.p2.p0.i64(ptr addrspace(2) align 16 %dest2, ptr align 16 %src0, i64 64, i1 false)
  call void @llvm.memcpy.p2.p1.i64(ptr addrspace(2) align 16 %dest2, ptr addrspace(1) align 16 %src1, i64 64, i1 false)
  call void @llvm.memcpy.p2.p2.i64(ptr addrspace(2) align 16 %dest2, ptr addrspace(2) align 16 %src2, i64 64, i1 false)
  call void @llvm.memcpy.p2.p3.i64(ptr addrspace(2) align 16 %dest2, ptr addrspace(3) align 16 %src3, i64 64, i1 false)
  call void @llvm.memcpy.p2.p4.i64(ptr addrspace(2) align 16 %dest2, ptr addrspace(4) align 16 %src4, i64 64, i1 false)
  call void @llvm.memcpy.p2.p5.i64(ptr addrspace(2) align 16 %dest2, ptr addrspace(5) align 16 %src5, i64 64, i1 false)

  call void @llvm.memcpy.p2.p0.i32(ptr addrspace(2) align 16 %dest2, ptr align 16 %src0, i32 32, i1 false)
  call void @llvm.memcpy.p2.p1.i32(ptr addrspace(2) align 16 %dest2, ptr addrspace(1) align 16 %src1, i32 32, i1 false)
  call void @llvm.memcpy.p2.p2.i32(ptr addrspace(2) align 16 %dest2, ptr addrspace(2) align 16 %src2, i32 32, i1 false)
  call void @llvm.memcpy.p2.p3.i32(ptr addrspace(2) align 16 %dest2, ptr addrspace(3) align 16 %src3, i32 32, i1 false)
  call void @llvm.memcpy.p2.p4.i32(ptr addrspace(2) align 16 %dest2, ptr addrspace(4) align 16 %src4, i32 32, i1 false)
  call void @llvm.memcpy.p2.p5.i32(ptr addrspace(2) align 16 %dest2, ptr addrspace(5) align 16 %src5, i32 32, i1 false)

  call void @llvm.memcpy.p3.p0.i64(ptr addrspace(3) align 16 %dest3, ptr align 16 %src0, i64 64, i1 false)
  call void @llvm.memcpy.p3.p1.i64(ptr addrspace(3) align 16 %dest3, ptr addrspace(1) align 16 %src1, i64 64, i1 false)
  call void @llvm.memcpy.p3.p2.i64(ptr addrspace(3) align 16 %dest3, ptr addrspace(2) align 16 %src2, i64 64, i1 false)
  call void @llvm.memcpy.p3.p3.i64(ptr addrspace(3) align 16 %dest3, ptr addrspace(3) align 16 %src3, i64 64, i1 false)
  call void @llvm.memcpy.p3.p4.i64(ptr addrspace(3) align 16 %dest3, ptr addrspace(4) align 16 %src4, i64 64, i1 false)
  call void @llvm.memcpy.p3.p5.i64(ptr addrspace(3) align 16 %dest3, ptr addrspace(5) align 16 %src5, i64 64, i1 false)

  call void @llvm.memcpy.p3.p0.i32(ptr addrspace(3) align 16 %dest3, ptr align 16 %src0, i32 32, i1 false)
  call void @llvm.memcpy.p3.p1.i32(ptr addrspace(3) align 16 %dest3, ptr addrspace(1) align 16 %src1, i32 32, i1 false)
  call void @llvm.memcpy.p3.p2.i32(ptr addrspace(3) align 16 %dest3, ptr addrspace(2) align 16 %src2, i32 32, i1 false)
  call void @llvm.memcpy.p3.p3.i32(ptr addrspace(3) align 16 %dest3, ptr addrspace(3) align 16 %src3, i32 32, i1 false)
  call void @llvm.memcpy.p3.p4.i32(ptr addrspace(3) align 16 %dest3, ptr addrspace(4) align 16 %src4, i32 32, i1 false)
  call void @llvm.memcpy.p3.p5.i32(ptr addrspace(3) align 16 %dest3, ptr addrspace(5) align 16 %src5, i32 32, i1 false)

  call void @llvm.memcpy.p4.p0.i64(ptr addrspace(4) align 16 %dest4, ptr align 16 %src0, i64 64, i1 false)
  call void @llvm.memcpy.p4.p1.i64(ptr addrspace(4) align 16 %dest4, ptr addrspace(1) align 16 %src1, i64 64, i1 false)
  call void @llvm.memcpy.p4.p2.i64(ptr addrspace(4) align 16 %dest4, ptr addrspace(2) align 16 %src2, i64 64, i1 false)
  call void @llvm.memcpy.p4.p3.i64(ptr addrspace(4) align 16 %dest4, ptr addrspace(3) align 16 %src3, i64 64, i1 false)
  call void @llvm.memcpy.p4.p4.i64(ptr addrspace(4) align 16 %dest4, ptr addrspace(4) align 16 %src4, i64 64, i1 false)
  call void @llvm.memcpy.p4.p5.i64(ptr addrspace(4) align 16 %dest4, ptr addrspace(5) align 16 %src5, i64 64, i1 false)

  call void @llvm.memcpy.p4.p0.i32(ptr addrspace(4) align 16 %dest4, ptr align 16 %src0, i32 32, i1 false)
  call void @llvm.memcpy.p4.p1.i32(ptr addrspace(4) align 16 %dest4, ptr addrspace(1) align 16 %src1, i32 32, i1 false)
  call void @llvm.memcpy.p4.p2.i32(ptr addrspace(4) align 16 %dest4, ptr addrspace(2) align 16 %src2, i32 32, i1 false)
  call void @llvm.memcpy.p4.p3.i32(ptr addrspace(4) align 16 %dest4, ptr addrspace(3) align 16 %src3, i32 32, i1 false)
  call void @llvm.memcpy.p4.p4.i32(ptr addrspace(4) align 16 %dest4, ptr addrspace(4) align 16 %src4, i32 32, i1 false)
  call void @llvm.memcpy.p4.p5.i32(ptr addrspace(4) align 16 %dest4, ptr addrspace(5) align 16 %src5, i32 32, i1 false)

  call void @llvm.memcpy.p5.p0.i64(ptr addrspace(5) align 16 %dest5, ptr align 16 %src0, i64 64, i1 false)
  call void @llvm.memcpy.p5.p1.i64(ptr addrspace(5) align 16 %dest5, ptr addrspace(1) align 16 %src1, i64 64, i1 false)
  call void @llvm.memcpy.p5.p2.i64(ptr addrspace(5) align 16 %dest5, ptr addrspace(2) align 16 %src2, i64 64, i1 false)
  call void @llvm.memcpy.p5.p3.i64(ptr addrspace(5) align 16 %dest5, ptr addrspace(3) align 16 %src3, i64 64, i1 false)
  call void @llvm.memcpy.p5.p4.i64(ptr addrspace(5) align 16 %dest5, ptr addrspace(4) align 16 %src4, i64 64, i1 false)
  call void @llvm.memcpy.p5.p5.i64(ptr addrspace(5) align 16 %dest5, ptr addrspace(5) align 16 %src5, i64 64, i1 false)

  call void @llvm.memcpy.p5.p0.i32(ptr addrspace(5) align 16 %dest5, ptr align 16 %src0, i32 32, i1 false)
  call void @llvm.memcpy.p5.p1.i32(ptr addrspace(5) align 16 %dest5, ptr addrspace(1) align 16 %src1, i32 32, i1 false)
  call void @llvm.memcpy.p5.p2.i32(ptr addrspace(5) align 16 %dest5, ptr addrspace(2) align 16 %src2, i32 32, i1 false)
  call void @llvm.memcpy.p5.p3.i32(ptr addrspace(5) align 16 %dest5, ptr addrspace(3) align 16 %src3, i32 32, i1 false)
  call void @llvm.memcpy.p5.p4.i32(ptr addrspace(5) align 16 %dest5, ptr addrspace(4) align 16 %src4, i32 32, i1 false)
  call void @llvm.memcpy.p5.p5.i32(ptr addrspace(5) align 16 %dest5, ptr addrspace(5) align 16 %src5, i32 32, i1 false)
  ret void
}

define void @test_mem_intrinsic_memmove(ptr %dest0,ptr %src0,ptr addrspace(1) %dest1,ptr addrspace(1) %src1,ptr addrspace(2) %dest2,ptr addrspace(2) %src2,ptr addrspace(3) %dest3,ptr addrspace(3) %src3,ptr addrspace(4) %dest4,ptr addrspace(4) %src4,ptr addrspace(5) %dest5,ptr addrspace(5) %src5) #0 {
entry:
  ;CHECK: define void @test_mem_intrinsic_memmove(ptr [[DEST0:%.*]], ptr [[SRC0:%.*]], ptr addrspace(1) [[DEST1:%.*]], ptr addrspace(1) [[SRC1:%.*]], ptr addrspace(2) [[DEST2:%.*]], ptr addrspace(2) [[SRC2:%.*]], ptr addrspace(3) [[DEST3:%.*]], ptr addrspace(3) [[SRC3:%.*]], ptr addrspace(4) [[DEST4:%.*]], ptr addrspace(4) [[SRC4:%.*]], ptr addrspace(5) [[DEST5:%.*]], ptr addrspace(5) [[SRC5:%.*]]) #2 {
  ;CHECK-NEXT: entry:
  ;CHECK-NEXT: [[VR0:%.*]] = call ptr @__asan_memmove(ptr [[DEST0]], ptr [[SRC0]], i64 64)
  ;CHECK-NEXT: [[VR1:%.*]] = addrspacecast ptr addrspace(1) [[SRC1]] to ptr
  ;CHECK-NEXT: [[VR2:%.*]] = call ptr @__asan_memmove(ptr [[DEST0]], ptr [[VR1]], i64 64)
  ;CHECK-NEXT: [[VR3:%.*]] = addrspacecast ptr addrspace(2) [[SRC2]] to ptr
  ;CHECK-NEXT: [[VR4:%.*]] = call ptr @__asan_memmove(ptr [[DEST0]], ptr [[VR3]], i64 64)
  ;CHECK-NEXT: [[VR5:%.*]] = addrspacecast ptr addrspace(3) [[SRC3]] to ptr
  ;CHECK-NEXT: [[VR6:%.*]] = call ptr @__asan_memmove(ptr [[DEST0]], ptr [[VR5]], i64 64)
  ;CHECK-NEXT: [[VR7:%.*]] = addrspacecast ptr addrspace(4) [[SRC4]] to ptr
  ;CHECK-NEXT: [[VR8:%.*]] = call ptr @__asan_memmove(ptr [[DEST0]], ptr [[VR7]], i64 64)
  ;CHECK-NEXT: [[VR9:%.*]] = addrspacecast ptr addrspace(5) [[SRC5]] to ptr
  ;CHECK-NEXT: [[VR10:%.*]] = call ptr @__asan_memmove(ptr [[DEST0]], ptr [[VR9]], i64 64)
  ;CHECK-NEXT: [[VR11:%.*]] = call ptr @__asan_memmove(ptr [[DEST0]], ptr [[SRC0]], i64 32)
  ;CHECK-NEXT: [[VR12:%.*]] = addrspacecast ptr addrspace(1) [[SRC1]] to ptr
  ;CHECK-NEXT: [[VR13:%.*]] = call ptr @__asan_memmove(ptr [[DEST0]], ptr [[VR12]], i64 32)
  ;CHECK-NEXT: [[VR14:%.*]] = addrspacecast ptr addrspace(2) [[SRC2]] to ptr
  ;CHECK-NEXT: [[VR15:%.*]] = call ptr @__asan_memmove(ptr [[DEST0]], ptr [[VR14]], i64 32)
  ;CHECK-NEXT: [[VR16:%.*]] = addrspacecast ptr addrspace(3) [[SRC3]] to ptr
  ;CHECK-NEXT: [[VR17:%.*]] = call ptr @__asan_memmove(ptr [[DEST0]], ptr [[VR16]], i64 32)
  ;CHECK-NEXT: [[VR18:%.*]] = addrspacecast ptr addrspace(4) [[SRC4]] to ptr
  ;CHECK-NEXT: [[VR19:%.*]] = call ptr @__asan_memmove(ptr [[DEST0]], ptr %18, i64 32)
  ;CHECK-NEXT: [[VR20:%.*]] = addrspacecast ptr addrspace(5) [[SRC5]] to ptr
  ;CHECK-NEXT: [[VR21:%.*]] = call ptr @__asan_memmove(ptr [[DEST0]], ptr [[VR20]], i64 32)
  ;CHECK-NEXT: [[VR22:%.*]] = addrspacecast ptr addrspace(1) [[DEST1]] to ptr
  ;CHECK-NEXT: [[VR23:%.*]] = call ptr @__asan_memmove(ptr [[VR22]], ptr [[SRC0]], i64 64)
  ;CHECK-NEXT: [[VR24:%.*]] = addrspacecast ptr addrspace(1) [[DEST1]] to ptr
  ;CHECK-NEXT: [[VR25:%.*]] = addrspacecast ptr addrspace(1) [[SRC1]] to ptr
  ;CHECK-NEXT: [[VR26:%.*]] = call ptr @__asan_memmove(ptr [[VR24]], ptr [[VR25]], i64 64)
  ;CHECK-NEXT: [[VR27:%.*]] = addrspacecast ptr addrspace(1) [[DEST1]] to ptr
  ;CHECK-NEXT: [[VR28:%.*]] = addrspacecast ptr addrspace(2) [[SRC2]] to ptr
  ;CHECK-NEXT: [[VR29:%.*]] = call ptr @__asan_memmove(ptr [[VR27]], ptr [[VR28]], i64 64)
  ;CHECK-NEXT: [[VR30:%.*]] = addrspacecast ptr addrspace(1) [[DEST1]] to ptr
  ;CHECK-NEXT: [[VR31:%.*]] = addrspacecast ptr addrspace(3) [[SRC3]] to ptr
  ;CHECK-NEXT: [[VR32:%.*]] = call ptr @__asan_memmove(ptr [[VR30]], ptr [[VR31]], i64 64)
  ;CHECK-NEXT: [[VR33:%.*]] = addrspacecast ptr addrspace(1) [[DEST1]] to ptr
  ;CHECK-NEXT: [[VR34:%.*]] = addrspacecast ptr addrspace(4) [[SRC4]] to ptr
  ;CHECK-NEXT: [[VR35:%.*]] = call ptr @__asan_memmove(ptr [[VR33]], ptr [[VR34]], i64 64)
  ;CHECK-NEXT: [[VR36:%.*]] = addrspacecast ptr addrspace(1) [[DEST1]] to ptr
  ;CHECK-NEXT: [[VR37:%.*]] = addrspacecast ptr addrspace(5) [[SRC5]] to ptr
  ;CHECK-NEXT: [[VR38:%.*]] = call ptr @__asan_memmove(ptr [[VR36]], ptr [[VR37]], i64 64)
  ;CHECK-NEXT: [[VR39:%.*]] = addrspacecast ptr addrspace(1) [[DEST1]] to ptr
  ;CHECK-NEXT: [[VR40:%.*]] = call ptr @__asan_memmove(ptr [[VR39]], ptr [[SRC0]], i64 32)
  ;CHECK-NEXT: [[VR41:%.*]] = addrspacecast ptr addrspace(1) [[DEST1]] to ptr
  ;CHECK-NEXT: [[VR42:%.*]] = addrspacecast ptr addrspace(1) [[SRC1]] to ptr
  ;CHECK-NEXT: [[VR43:%.*]] = call ptr @__asan_memmove(ptr [[VR41]], ptr [[VR42]], i64 32)
  ;CHECK-NEXT: [[VR44:%.*]] = addrspacecast ptr addrspace(1) [[DEST1]] to ptr
  ;CHECK-NEXT: [[VR45:%.*]] = addrspacecast ptr addrspace(2) [[SRC2]] to ptr
  ;CHECK-NEXT: [[VR46:%.*]] = call ptr @__asan_memmove(ptr [[VR44]], ptr [[VR45]], i64 32)
  ;CHECK-NEXT: [[VR47:%.*]] = addrspacecast ptr addrspace(1) [[DEST1]] to ptr
  ;CHECK-NEXT: [[VR48:%.*]] = addrspacecast ptr addrspace(3) [[SRC3]] to ptr
  ;CHECK-NEXT: [[VR49:%.*]] = call ptr @__asan_memmove(ptr [[VR47]], ptr [[VR48]], i64 32)
  ;CHECK-NEXT: [[VR50:%.*]] = addrspacecast ptr addrspace(1) [[DEST1]] to ptr
  ;CHECK-NEXT: [[VR51:%.*]] = addrspacecast ptr addrspace(4) [[SRC4]] to ptr
  ;CHECK-NEXT: [[VR52:%.*]] = call ptr @__asan_memmove(ptr [[VR50]], ptr [[VR51]], i64 32)
  ;CHECK-NEXT: [[VR53:%.*]] = addrspacecast ptr addrspace(1) [[DEST1]] to ptr
  ;CHECK-NEXT: [[VR54:%.*]] = addrspacecast ptr addrspace(5) [[SRC5]] to ptr
  ;CHECK-NEXT: [[VR55:%.*]] = call ptr @__asan_memmove(ptr [[VR53]], ptr [[VR54]], i64 32)
  ;CHECK-NEXT: [[VR56:%.*]] = addrspacecast ptr addrspace(2) [[DEST2]] to ptr
  ;CHECK-NEXT: [[VR57:%.*]] = call ptr @__asan_memmove(ptr [[VR56]], ptr [[SRC0]], i64 64)
  ;CHECK-NEXT: [[VR58:%.*]] = addrspacecast ptr addrspace(2) [[DEST2]] to ptr
  ;CHECK-NEXT: [[VR59:%.*]] = addrspacecast ptr addrspace(1) [[SRC1]] to ptr
  ;CHECK-NEXT: [[VR60:%.*]] = call ptr @__asan_memmove(ptr [[VR58]], ptr [[VR59]], i64 64)
  ;CHECK-NEXT: [[VR61:%.*]] = addrspacecast ptr addrspace(2) [[DEST2]] to ptr
  ;CHECK-NEXT: [[VR62:%.*]] = addrspacecast ptr addrspace(2) [[SRC2]] to ptr
  ;CHECK-NEXT: [[VR63:%.*]] = call ptr @__asan_memmove(ptr [[VR61]], ptr [[VR62]], i64 64)
  ;CHECK-NEXT: [[VR64:%.*]] = addrspacecast ptr addrspace(2) [[DEST2]] to ptr
  ;CHECK-NEXT: [[VR65:%.*]] = addrspacecast ptr addrspace(3) [[SRC3]] to ptr
  ;CHECK-NEXT: [[VR66:%.*]] = call ptr @__asan_memmove(ptr [[VR64]], ptr [[VR65]], i64 64)
  ;CHECK-NEXT: [[VR67:%.*]] = addrspacecast ptr addrspace(2) [[DEST2]] to ptr
  ;CHECK-NEXT: [[VR68:%.*]] = addrspacecast ptr addrspace(4) [[SRC4]] to ptr
  ;CHECK-NEXT: [[VR69:%.*]] = call ptr @__asan_memmove(ptr [[VR67]], ptr [[VR68]], i64 64)
  ;CHECK-NEXT: [[VR70:%.*]] = addrspacecast ptr addrspace(2) [[DEST2]] to ptr
  ;CHECK-NEXT: [[VR71:%.*]] = addrspacecast ptr addrspace(5) [[SRC5]] to ptr
  ;CHECK-NEXT: [[VR72:%.*]] = call ptr @__asan_memmove(ptr [[VR70]], ptr [[VR71]], i64 64)
  ;CHECK-NEXT: [[VR73:%.*]] = addrspacecast ptr addrspace(2) [[DEST2]] to ptr
  ;CHECK-NEXT: [[VR74:%.*]] = call ptr @__asan_memmove(ptr [[VR73]], ptr [[SRC0]], i64 32)
  ;CHECK-NEXT: [[VR75:%.*]] = addrspacecast ptr addrspace(2) [[DEST2]] to ptr
  ;CHECK-NEXT: [[VR76:%.*]] = addrspacecast ptr addrspace(1) [[SRC1]] to ptr
  ;CHECK-NEXT: [[VR77:%.*]] = call ptr @__asan_memmove(ptr [[VR75]], ptr [[VR76]], i64 32)
  ;CHECK-NEXT: [[VR78:%.*]] = addrspacecast ptr addrspace(2) [[DEST2]] to ptr
  ;CHECK-NEXT: [[VR79:%.*]] = addrspacecast ptr addrspace(2) [[SRC2]] to ptr
  ;CHECK-NEXT: [[VR80:%.*]] = call ptr @__asan_memmove(ptr [[VR78]], ptr [[VR79]], i64 32)
  ;CHECK-NEXT: [[VR81:%.*]] = addrspacecast ptr addrspace(2) [[DEST2]] to ptr
  ;CHECK-NEXT: [[VR82:%.*]] = addrspacecast ptr addrspace(3) [[SRC3]] to ptr
  ;CHECK-NEXT: [[VR83:%.*]] = call ptr @__asan_memmove(ptr [[VR81]], ptr [[VR82]], i64 32)
  ;CHECK-NEXT: [[VR84:%.*]] = addrspacecast ptr addrspace(2) [[DEST2]] to ptr
  ;CHECK-NEXT: [[VR85:%.*]] = addrspacecast ptr addrspace(4) [[SRC4]] to ptr
  ;CHECK-NEXT: [[VR86:%.*]] = call ptr @__asan_memmove(ptr [[VR84]], ptr [[VR85]], i64 32)
  ;CHECK-NEXT: [[VR87:%.*]] = addrspacecast ptr addrspace(2) [[DEST2]] to ptr
  ;CHECK-NEXT: [[VR88:%.*]] = addrspacecast ptr addrspace(5) [[SRC5]] to ptr
  ;CHECK-NEXT: [[VR89:%.*]] = call ptr @__asan_memmove(ptr [[VR87]], ptr [[VR88]], i64 32)
  ;CHECK-NEXT: [[VR90:%.*]] = addrspacecast ptr addrspace(3) [[DEST3]] to ptr
  ;CHECK-NEXT: [[VR91:%.*]] = call ptr @__asan_memmove(ptr [[VR90]], ptr [[SRC0]], i64 64)
  ;CHECK-NEXT: [[VR92:%.*]] = addrspacecast ptr addrspace(3) [[DEST3]] to ptr
  ;CHECK-NEXT: [[VR93:%.*]] = addrspacecast ptr addrspace(1) [[SRC1]] to ptr
  ;CHECK-NEXT: [[VR94:%.*]] = call ptr @__asan_memmove(ptr [[VR92]], ptr [[VR93]], i64 64)
  ;CHECK-NEXT: [[VR95:%.*]] = addrspacecast ptr addrspace(3) [[DEST3]] to ptr
  ;CHECK-NEXT: [[VR96:%.*]] = addrspacecast ptr addrspace(2) [[SRC2]] to ptr
  ;CHECK-NEXT: [[VR97:%.*]] = call ptr @__asan_memmove(ptr [[VR95]], ptr [[VR96]], i64 64)
  ;CHECK-NEXT: [[VR98:%.*]] = addrspacecast ptr addrspace(3) [[DEST3]] to ptr
  ;CHECK-NEXT: [[VR99:%.*]] = addrspacecast ptr addrspace(3) [[SRC3]] to ptr
  ;CHECK-NEXT: [[VR100:%.*]] = call ptr @__asan_memmove(ptr [[VR98]], ptr [[VR99]], i64 64)
  ;CHECK-NEXT: [[VR101:%.*]] = addrspacecast ptr addrspace(3) [[DEST3]] to ptr
  ;CHECK-NEXT: [[VR102:%.*]] = addrspacecast ptr addrspace(4) [[SRC4]] to ptr
  ;CHECK-NEXT: [[VR103:%.*]] = call ptr @__asan_memmove(ptr [[VR101]], ptr [[VR102]], i64 64)
  ;CHECK-NEXT: [[VR104:%.*]] = addrspacecast ptr addrspace(3) [[DEST3]] to ptr
  ;CHECK-NEXT: [[VR105:%.*]] = addrspacecast ptr addrspace(5) [[SRC5]] to ptr
  ;CHECK-NEXT: [[VR106:%.*]] = call ptr @__asan_memmove(ptr [[VR104]], ptr [[VR105]], i64 64)
  ;CHECK-NEXT: [[VR107:%.*]] = addrspacecast ptr addrspace(3) [[DEST3]] to ptr
  ;CHECK-NEXT: [[VR108:%.*]] = call ptr @__asan_memmove(ptr [[VR107]], ptr [[SRC0]], i64 32)
  ;CHECK-NEXT: [[VR109:%.*]] = addrspacecast ptr addrspace(3) [[DEST3]] to ptr
  ;CHECK-NEXT: [[VR110:%.*]] = addrspacecast ptr addrspace(1) [[SRC1]] to ptr
  ;CHECK-NEXT: [[VR111:%.*]] = call ptr @__asan_memmove(ptr [[VR109]], ptr [[VR110]], i64 32)
  ;CHECK-NEXT: [[VR112:%.*]] = addrspacecast ptr addrspace(3) [[DEST3]] to ptr
  ;CHECK-NEXT: [[VR113:%.*]] = addrspacecast ptr addrspace(2) [[SRC2]] to ptr
  ;CHECK-NEXT: [[VR114:%.*]] = call ptr @__asan_memmove(ptr [[VR112]], ptr [[VR113]], i64 32)
  ;CHECK-NEXT: [[VR115:%.*]] = addrspacecast ptr addrspace(3) [[DEST3]] to ptr
  ;CHECK-NEXT: [[VR116:%.*]] = addrspacecast ptr addrspace(3) [[SRC3]] to ptr
  ;CHECK-NEXT: [[VR117:%.*]] = call ptr @__asan_memmove(ptr [[VR115]], ptr [[VR116]], i64 32)
  ;CHECK-NEXT: [[VR118:%.*]] = addrspacecast ptr addrspace(3) [[DEST3]] to ptr
  ;CHECK-NEXT: [[VR119:%.*]] = addrspacecast ptr addrspace(4) [[SRC4]] to ptr
  ;CHECK-NEXT: [[VR120:%.*]] = call ptr @__asan_memmove(ptr [[VR118]], ptr [[VR119]], i64 32)
  ;CHECK-NEXT: [[VR121:%.*]] = addrspacecast ptr addrspace(3) [[DEST3]] to ptr
  ;CHECK-NEXT: [[VR122:%.*]] = addrspacecast ptr addrspace(5) [[SRC5]] to ptr
  ;CHECK-NEXT: [[VR123:%.*]] = call ptr @__asan_memmove(ptr [[VR121]], ptr [[VR122]], i64 32)
  ;CHECK-NEXT: [[VR124:%.*]] = addrspacecast ptr addrspace(4) [[DEST4]] to ptr
  ;CHECK-NEXT: [[VR125:%.*]] = call ptr @__asan_memmove(ptr [[VR124]], ptr [[SRC0]], i64 64)
  ;CHECK-NEXT: [[VR126:%.*]] = addrspacecast ptr addrspace(4) [[DEST4]] to ptr
  ;CHECK-NEXT: [[VR127:%.*]] = addrspacecast ptr addrspace(1) [[SRC1]] to ptr
  ;CHECK-NEXT: [[VR128:%.*]] = call ptr @__asan_memmove(ptr [[VR126]], ptr [[VR127]], i64 64)
  ;CHECK-NEXT: [[VR129:%.*]] = addrspacecast ptr addrspace(4) [[DEST4]] to ptr
  ;CHECK-NEXT: [[VR130:%.*]] = addrspacecast ptr addrspace(2) [[SRC2]] to ptr
  ;CHECK-NEXT: [[VR131:%.*]] = call ptr @__asan_memmove(ptr [[VR129]], ptr [[VR130]], i64 64)
  ;CHECK-NEXT: [[VR132:%.*]] = addrspacecast ptr addrspace(4) [[DEST4]] to ptr
  ;CHECK-NEXT: [[VR133:%.*]] = addrspacecast ptr addrspace(3) [[SRC3]] to ptr
  ;CHECK-NEXT: [[VR134:%.*]] = call ptr @__asan_memmove(ptr [[VR132]], ptr [[VR133]], i64 64)
  ;CHECK-NEXT: [[VR135:%.*]] = addrspacecast ptr addrspace(4) [[DEST4]] to ptr
  ;CHECK-NEXT: [[VR136:%.*]] = addrspacecast ptr addrspace(4) [[SRC4]] to ptr
  ;CHECK-NEXT: [[VR137:%.*]] = call ptr @__asan_memmove(ptr [[VR135]], ptr [[VR136]], i64 64)
  ;CHECK-NEXT: [[VR138:%.*]] = addrspacecast ptr addrspace(4) [[DEST4]] to ptr
  ;CHECK-NEXT: [[VR139:%.*]] = addrspacecast ptr addrspace(5) [[SRC5]] to ptr
  ;CHECK-NEXT: [[VR140:%.*]] = call ptr @__asan_memmove(ptr [[VR138]], ptr [[VR139]], i64 64)
  ;CHECK-NEXT: [[VR141:%.*]] = addrspacecast ptr addrspace(4) [[DEST4]] to ptr
  ;CHECK-NEXT: [[VR142:%.*]] = call ptr @__asan_memmove(ptr [[VR141]], ptr [[SRC0]], i64 32)
  ;CHECK-NEXT: [[VR143:%.*]] = addrspacecast ptr addrspace(4) [[DEST4]] to ptr
  ;CHECK-NEXT: [[VR144:%.*]] = addrspacecast ptr addrspace(1) [[SRC1]] to ptr
  ;CHECK-NEXT: [[VR145:%.*]] = call ptr @__asan_memmove(ptr [[VR143]], ptr [[VR144]], i64 32)
  ;CHECK-NEXT: [[VR146:%.*]] = addrspacecast ptr addrspace(4) [[DEST4]] to ptr
  ;CHECK-NEXT: [[VR147:%.*]] = addrspacecast ptr addrspace(2) [[SRC2]] to ptr
  ;CHECK-NEXT: [[VR148:%.*]] = call ptr @__asan_memmove(ptr [[VR146]], ptr [[VR147]], i64 32)
  ;CHECK-NEXT: [[VR149:%.*]] = addrspacecast ptr addrspace(4) [[DEST4]] to ptr
  ;CHECK-NEXT: [[VR150:%.*]] = addrspacecast ptr addrspace(3) [[SRC3]] to ptr
  ;CHECK-NEXT: [[VR151:%.*]] = call ptr @__asan_memmove(ptr [[VR149]], ptr [[VR150]], i64 32)
  ;CHECK-NEXT: [[VR152:%.*]] = addrspacecast ptr addrspace(4) [[DEST4]] to ptr
  ;CHECK-NEXT: [[VR153:%.*]] = addrspacecast ptr addrspace(4) [[SRC4]] to ptr
  ;CHECK-NEXT: [[VR154:%.*]] = call ptr @__asan_memmove(ptr [[VR152]], ptr [[VR153]], i64 32)
  ;CHECK-NEXT: [[VR155:%.*]] = addrspacecast ptr addrspace(4) [[DEST4]] to ptr
  ;CHECK-NEXT: [[VR156:%.*]] = addrspacecast ptr addrspace(5) [[SRC5]] to ptr
  ;CHECK-NEXT: [[VR157:%.*]] = call ptr @__asan_memmove(ptr [[VR155]], ptr [[VR156]], i64 32)
  ;CHECK-NEXT: [[VR158:%.*]] = addrspacecast ptr addrspace(5) [[DEST5]] to ptr
  ;CHECK-NEXT: [[VR159:%.*]] = call ptr @__asan_memmove(ptr [[VR158]], ptr [[SRC0]], i64 64)
  ;CHECK-NEXT: [[VR160:%.*]] = addrspacecast ptr addrspace(5) [[DEST5]] to ptr
  ;CHECK-NEXT: [[VR161:%.*]] = addrspacecast ptr addrspace(1) [[SRC1]] to ptr
  ;CHECK-NEXT: [[VR162:%.*]] = call ptr @__asan_memmove(ptr [[VR160]], ptr [[VR161]], i64 64)
  ;CHECK-NEXT: [[VR163:%.*]] = addrspacecast ptr addrspace(5) [[DEST5]] to ptr
  ;CHECK-NEXT: [[VR164:%.*]] = addrspacecast ptr addrspace(2) [[SRC2]] to ptr
  ;CHECK-NEXT: [[VR165:%.*]] = call ptr @__asan_memmove(ptr [[VR163]], ptr [[VR164]], i64 64)
  ;CHECK-NEXT: [[VR166:%.*]] = addrspacecast ptr addrspace(5) [[DEST5]] to ptr
  ;CHECK-NEXT: [[VR167:%.*]] = addrspacecast ptr addrspace(3) [[SRC3]] to ptr
  ;CHECK-NEXT: [[VR168:%.*]] = call ptr @__asan_memmove(ptr [[VR166]], ptr [[VR167]], i64 64)
  ;CHECK-NEXT: [[VR169:%.*]] = addrspacecast ptr addrspace(5) [[DEST5]] to ptr
  ;CHECK-NEXT: [[VR170:%.*]] = addrspacecast ptr addrspace(4) [[SRC4]] to ptr
  ;CHECK-NEXT: [[VR171:%.*]] = call ptr @__asan_memmove(ptr [[VR169]], ptr [[VR170]], i64 64)
  ;CHECK-NEXT: [[VR172:%.*]] = addrspacecast ptr addrspace(5) [[DEST5]] to ptr
  ;CHECK-NEXT: [[VR173:%.*]] = addrspacecast ptr addrspace(5) [[SRC5]] to ptr
  ;CHECK-NEXT: [[VR174:%.*]] = call ptr @__asan_memmove(ptr [[VR172]], ptr [[VR173]], i64 64)
  ;CHECK-NEXT: [[VR175:%.*]] = addrspacecast ptr addrspace(5) [[DEST5]] to ptr
  ;CHECK-NEXT: [[VR176:%.*]] = call ptr @__asan_memmove(ptr %175, ptr %src0, i64 32)
  ;CHECK-NEXT: [[VR177:%.*]] = addrspacecast ptr addrspace(5) [[DEST5]] to ptr
  ;CHECK-NEXT: [[VR178:%.*]] = addrspacecast ptr addrspace(1) [[SRC1]] to ptr
  ;CHECK-NEXT: [[VR179:%.*]] = call ptr @__asan_memmove(ptr [[VR177]], ptr [[VR178]], i64 32)
  ;CHECK-NEXT: [[VR180:%.*]] = addrspacecast ptr addrspace(5) [[DEST5]] to ptr
  ;CHECK-NEXT: [[VR181:%.*]] = addrspacecast ptr addrspace(2) [[SRC2]] to ptr
  ;CHECK-NEXT: [[VR182:%.*]] = call ptr @__asan_memmove(ptr [[VR180]], ptr [[VR181]], i64 32)
  ;CHECK-NEXT: [[VR183:%.*]] = addrspacecast ptr addrspace(5) [[DEST5]] to ptr
  ;CHECK-NEXT: [[VR184:%.*]] = addrspacecast ptr addrspace(3) [[SRC3]] to ptr
  ;CHECK-NEXT: [[VR185:%.*]] = call ptr @__asan_memmove(ptr [[VR183]], ptr [[VR184]], i64 32)
  ;CHECK-NEXT: [[VR186:%.*]] = addrspacecast ptr addrspace(5) [[DEST5]] to ptr
  ;CHECK-NEXT: [[VR187:%.*]] = addrspacecast ptr addrspace(4) [[SRC4]] to ptr
  ;CHECK-NEXT: [[VR188:%.*]] = call ptr @__asan_memmove(ptr [[VR186]], ptr [[VR187]], i64 32)
  ;CHECK-NEXT: [[VR189:%.*]] = addrspacecast ptr addrspace(5) [[DEST5]] to ptr
  ;CHECK-NEXT: [[VR190:%.*]] = addrspacecast ptr addrspace(5) [[SRC5]] to ptr
  ;CHECK-NEXT: [[VR191:%.*]] = call ptr @__asan_memmove(ptr [[VR189]], ptr [[VR190]], i64 32)

	call void @llvm.memmove.p0.p0.i64(ptr align 16 %dest0, ptr align 16 %src0, i64 64, i1 false)
  call void @llvm.memmove.p0.p1.i64(ptr align 16 %dest0, ptr addrspace(1) align 16 %src1, i64 64, i1 false)
  call void @llvm.memmove.p0.p2.i64(ptr align 16 %dest0, ptr addrspace(2) align 16 %src2, i64 64, i1 false)
  call void @llvm.memmove.p0.p3.i64(ptr align 16 %dest0, ptr addrspace(3) align 16 %src3, i64 64, i1 false)
  call void @llvm.memmove.p0.p4.i64(ptr align 16 %dest0, ptr addrspace(4) align 16 %src4, i64 64, i1 false)
  call void @llvm.memmove.p0.p5.i64(ptr align 16 %dest0, ptr addrspace(5) align 16 %src5, i64 64, i1 false)

  call void @llvm.memmove.p0.p0.i32(ptr align 16 %dest0, ptr align 16 %src0, i32 32, i1 false)
  call void @llvm.memmove.p0.p1.i32(ptr align 16 %dest0, ptr addrspace(1) align 16 %src1, i32 32, i1 false)
  call void @llvm.memmove.p0.p2.i32(ptr align 16 %dest0, ptr addrspace(2) align 16 %src2, i32 32, i1 false)
  call void @llvm.memmove.p0.p3.i32(ptr align 16 %dest0, ptr addrspace(3) align 16 %src3, i32 32, i1 false)
  call void @llvm.memmove.p0.p4.i32(ptr align 16 %dest0, ptr addrspace(4) align 16 %src4, i32 32, i1 false)
  call void @llvm.memmove.p0.p5.i32(ptr align 16 %dest0, ptr addrspace(5) align 16 %src5, i32 32, i1 false)

  call void @llvm.memmove.p1.p0.i64(ptr addrspace(1) align 16 %dest1, ptr align 16 %src0, i64 64, i1 false)
  call void @llvm.memmove.p1.p1.i64(ptr addrspace(1) align 16 %dest1, ptr addrspace(1) align 16 %src1, i64 64,  i1 false)
  call void @llvm.memmove.p1.p2.i64(ptr addrspace(1) align 16 %dest1, ptr addrspace(2) align 16 %src2, i64 64, i1 false)
  call void @llvm.memmove.p1.p3.i64(ptr addrspace(1) align 16 %dest1, ptr addrspace(3) align 16 %src3, i64 64, i1 false)
  call void @llvm.memmove.p1.p4.i64(ptr addrspace(1) align 16 %dest1, ptr addrspace(4) align 16 %src4, i64 64, i1 false)
  call void @llvm.memmove.p1.p5.i64(ptr addrspace(1) align 16 %dest1, ptr addrspace(5) align 16 %src5, i64 64, i1 false)

  call void @llvm.memmove.p1.p0.i32(ptr addrspace(1) align 16 %dest1, ptr align 16 %src0, i32 32, i1 false)
  call void @llvm.memmove.p1.p1.i32(ptr addrspace(1) align 16 %dest1, ptr addrspace(1) align 16 %src1, i32 32, i1 false)
  call void @llvm.memmove.p1.p2.i32(ptr addrspace(1) align 16 %dest1, ptr addrspace(2) align 16 %src2, i32 32, i1 false)
  call void @llvm.memmove.p1.p3.i32(ptr addrspace(1) align 16 %dest1, ptr addrspace(3) align 16 %src3, i32 32, i1 false)
  call void @llvm.memmove.p1.p4.i32(ptr addrspace(1) align 16 %dest1, ptr addrspace(4) align 16 %src4, i32 32, i1 false)
  call void @llvm.memmove.p1.p5.i32(ptr addrspace(1) align 16 %dest1, ptr addrspace(5) align 16 %src5, i32 32, i1 false)

  call void @llvm.memmove.p2.p0.i64(ptr addrspace(2) align 16 %dest2, ptr align 16 %src0, i64 64, i1 false)
  call void @llvm.memmove.p2.p1.i64(ptr addrspace(2) align 16 %dest2, ptr addrspace(1) align 16 %src1, i64 64, i1 false)
  call void @llvm.memmove.p2.p2.i64(ptr addrspace(2) align 16 %dest2, ptr addrspace(2) align 16 %src2, i64 64, i1 false)
  call void @llvm.memmove.p2.p3.i64(ptr addrspace(2) align 16 %dest2, ptr addrspace(3) align 16 %src3, i64 64, i1 false)
  call void @llvm.memmove.p2.p4.i64(ptr addrspace(2) align 16 %dest2, ptr addrspace(4) align 16 %src4, i64 64, i1 false)
  call void @llvm.memmove.p2.p5.i64(ptr addrspace(2) align 16 %dest2, ptr addrspace(5) align 16 %src5, i64 64, i1 false)

  call void @llvm.memmove.p2.p0.i32(ptr addrspace(2) align 16 %dest2, ptr align 16 %src0, i32 32, i1 false)
  call void @llvm.memmove.p2.p1.i32(ptr addrspace(2) align 16 %dest2, ptr addrspace(1) align 16 %src1, i32 32, i1 false)
  call void @llvm.memmove.p2.p2.i32(ptr addrspace(2) align 16 %dest2, ptr addrspace(2) align 16 %src2, i32 32, i1 false)
  call void @llvm.memmove.p2.p3.i32(ptr addrspace(2) align 16 %dest2, ptr addrspace(3) align 16 %src3, i32 32, i1 false)
  call void @llvm.memmove.p2.p4.i32(ptr addrspace(2) align 16 %dest2, ptr addrspace(4) align 16 %src4, i32 32, i1 false)
  call void @llvm.memmove.p2.p5.i32(ptr addrspace(2) align 16 %dest2, ptr addrspace(5) align 16 %src5, i32 32, i1 false)

  call void @llvm.memmove.p3.p0.i64(ptr addrspace(3) align 16 %dest3, ptr align 16 %src0, i64 64, i1 false)
  call void @llvm.memmove.p3.p1.i64(ptr addrspace(3) align 16 %dest3, ptr addrspace(1) align 16 %src1, i64 64, i1 false)
  call void @llvm.memmove.p3.p2.i64(ptr addrspace(3) align 16 %dest3, ptr addrspace(2) align 16 %src2, i64 64, i1 false)
  call void @llvm.memmove.p3.p3.i64(ptr addrspace(3) align 16 %dest3, ptr addrspace(3) align 16 %src3, i64 64, i1 false)
  call void @llvm.memmove.p3.p4.i64(ptr addrspace(3) align 16 %dest3, ptr addrspace(4) align 16 %src4, i64 64, i1 false)
  call void @llvm.memmove.p3.p5.i64(ptr addrspace(3) align 16 %dest3, ptr addrspace(5) align 16 %src5, i64 64, i1 false)

  call void @llvm.memmove.p3.p0.i32(ptr addrspace(3) align 16 %dest3, ptr align 16 %src0, i32 32, i1 false)
  call void @llvm.memmove.p3.p1.i32(ptr addrspace(3) align 16 %dest3, ptr addrspace(1) align 16 %src1, i32 32, i1 false)
  call void @llvm.memmove.p3.p2.i32(ptr addrspace(3) align 16 %dest3, ptr addrspace(2) align 16 %src2, i32 32, i1 false)
  call void @llvm.memmove.p3.p3.i32(ptr addrspace(3) align 16 %dest3, ptr addrspace(3) align 16 %src3, i32 32, i1 false)
  call void @llvm.memmove.p3.p4.i32(ptr addrspace(3) align 16 %dest3, ptr addrspace(4) align 16 %src4, i32 32, i1 false)
  call void @llvm.memmove.p3.p5.i32(ptr addrspace(3) align 16 %dest3, ptr addrspace(5) align 16 %src5, i32 32, i1 false)

  call void @llvm.memmove.p4.p0.i64(ptr addrspace(4) align 16 %dest4, ptr align 16 %src0, i64 64, i1 false)
  call void @llvm.memmove.p4.p1.i64(ptr addrspace(4) align 16 %dest4, ptr addrspace(1) align 16 %src1, i64 64, i1 false)
  call void @llvm.memmove.p4.p2.i64(ptr addrspace(4) align 16 %dest4, ptr addrspace(2) align 16 %src2, i64 64, i1 false)
  call void @llvm.memmove.p4.p3.i64(ptr addrspace(4) align 16 %dest4, ptr addrspace(3) align 16 %src3, i64 64, i1 false)
  call void @llvm.memmove.p4.p4.i64(ptr addrspace(4) align 16 %dest4, ptr addrspace(4) align 16 %src4, i64 64, i1 false)
  call void @llvm.memmove.p4.p5.i64(ptr addrspace(4) align 16 %dest4, ptr addrspace(5) align 16 %src5, i64 64, i1 false)

  call void @llvm.memmove.p4.p0.i32(ptr addrspace(4) align 16 %dest4, ptr align 16 %src0, i32 32, i1 false)
  call void @llvm.memmove.p4.p1.i32(ptr addrspace(4) align 16 %dest4, ptr addrspace(1) align 16 %src1, i32 32, i1 false)
  call void @llvm.memmove.p4.p2.i32(ptr addrspace(4) align 16 %dest4, ptr addrspace(2) align 16 %src2, i32 32, i1 false)
  call void @llvm.memmove.p4.p3.i32(ptr addrspace(4) align 16 %dest4, ptr addrspace(3) align 16 %src3, i32 32, i1 false)
  call void @llvm.memmove.p4.p4.i32(ptr addrspace(4) align 16 %dest4, ptr addrspace(4) align 16 %src4, i32 32, i1 false)
  call void @llvm.memmove.p4.p5.i32(ptr addrspace(4) align 16 %dest4, ptr addrspace(5) align 16 %src5, i32 32, i1 false)

  call void @llvm.memmove.p5.p0.i64(ptr addrspace(5) align 16 %dest5, ptr align 16 %src0, i64 64, i1 false)
  call void @llvm.memmove.p5.p1.i64(ptr addrspace(5) align 16 %dest5, ptr addrspace(1) align 16 %src1, i64 64, i1 false)
  call void @llvm.memmove.p5.p2.i64(ptr addrspace(5) align 16 %dest5, ptr addrspace(2) align 16 %src2, i64 64, i1 false)
  call void @llvm.memmove.p5.p3.i64(ptr addrspace(5) align 16 %dest5, ptr addrspace(3) align 16 %src3, i64 64, i1 false)
  call void @llvm.memmove.p5.p4.i64(ptr addrspace(5) align 16 %dest5, ptr addrspace(4) align 16 %src4, i64 64, i1 false)
  call void @llvm.memmove.p5.p5.i64(ptr addrspace(5) align 16 %dest5, ptr addrspace(5) align 16 %src5, i64 64, i1 false)

  call void @llvm.memmove.p5.p0.i32(ptr addrspace(5) align 16 %dest5, ptr align 16 %src0, i32 32, i1 false)
  call void @llvm.memmove.p5.p1.i32(ptr addrspace(5) align 16 %dest5, ptr addrspace(1) align 16 %src1, i32 32, i1 false)
  call void @llvm.memmove.p5.p2.i32(ptr addrspace(5) align 16 %dest5, ptr addrspace(2) align 16 %src2, i32 32, i1 false)
  call void @llvm.memmove.p5.p3.i32(ptr addrspace(5) align 16 %dest5, ptr addrspace(3) align 16 %src3, i32 32, i1 false)
  call void @llvm.memmove.p5.p4.i32(ptr addrspace(5) align 16 %dest5, ptr addrspace(4) align 16 %src4, i32 32, i1 false)
  call void @llvm.memmove.p5.p5.i32(ptr addrspace(5) align 16 %dest5, ptr addrspace(5) align 16 %src5, i32 32, i1 false)
  ret void
}


define void @test_mem_intrinsic_memset(ptr %ptr0,ptr addrspace(1) %ptr1,ptr addrspace(2) %ptr2,ptr addrspace(3) %ptr3,ptr addrspace(4) %ptr4,ptr addrspace(5) %ptr5) #0{
entry:
  ;CHECK: define void @test_mem_intrinsic_memset(ptr [[PTR0:%.*]], ptr addrspace(1) [[PTR1:%.*]], ptr addrspace(2) [[PTR2:%.*]], ptr addrspace(3) [[PTR3:%.*]], ptr addrspace(4) [[PTR4:%.*]], ptr addrspace(5) [[PTR5:%.*]]) #2 {
  ;CHECK-NEXT: entry:
  ;CHECK-NEXT: [[VR0:%.*]] = call ptr @__asan_memset(ptr [[PTR0]], i32 1, i64 128)
  ;CHECK-NEXT: [[VR1:%.*]] = addrspacecast ptr addrspace(1) [[PTR1]] to ptr
  ;CHECK-NEXT: [[VR2:%.*]] = call ptr @__asan_memset(ptr [[VR1]], i32 1, i64 128)
  ;CHECK-NEXT: [[VR3:%.*]] = addrspacecast ptr addrspace(2) [[PTR2]] to ptr
  ;CHECK-NEXT: [[VR4:%.*]] = call ptr @__asan_memset(ptr [[VR3]], i32 1, i64 128)
  ;CHECK-NEXT: [[VR5:%.*]] = addrspacecast ptr addrspace(3) [[PTR3]] to ptr
  ;CHECK-NEXT: [[VR6:%.*]] = call ptr @__asan_memset(ptr [[VR5]], i32 1, i64 128)
  ;CHECK-NEXT: [[VR7:%.*]] = addrspacecast ptr addrspace(4) [[PTR4]] to ptr
  ;CHECK-NEXT: [[VR8:%.*]] = call ptr @__asan_memset(ptr [[VR7]], i32 1, i64 128)
  ;CHECK-NEXT: [[VR9:%.*]] = addrspacecast ptr addrspace(5) [[PTR5]] to ptr
  ;CHECK-NEXT: [[VR10:%.*]] = call ptr @__asan_memset(ptr [[VR9]], i32 1, i64 128)
  ;CHECK-NEXT: [[VR11:%.*]] = call ptr @__asan_memset(ptr [[PTR0]], i32 1, i64 128)
  ;CHECK-NEXT: [[VR12:%.*]] = addrspacecast ptr addrspace(1) [[PTR1]] to ptr
  ;CHECK-NEXT: [[VR13:%.*]] = call ptr @__asan_memset(ptr [[VR12]], i32 1, i64 128)
  ;CHECK-NEXT: [[VR14:%.*]] = addrspacecast ptr addrspace(2) [[PTR2]] to ptr
  ;CHECK-NEXT: [[VR15:%.*]] = call ptr @__asan_memset(ptr [[VR14]], i32 1, i64 128)
  ;CHECK-NEXT: [[VR16:%.*]] = addrspacecast ptr addrspace(3) [[PTR3]] to ptr
  ;CHECK-NEXT: [[VR17:%.*]] = call ptr @__asan_memset(ptr [[VR16]], i32 1, i64 128)
  ;CHECK-NEXT: [[VR18:%.*]] = addrspacecast ptr addrspace(4) [[PTR4]] to ptr
  ;CHECK-NEXT: [[VR19:%.*]] = call ptr @__asan_memset(ptr [[VR18]], i32 1, i64 128)
  ;CHECK-NEXT: [[VR20:%.*]] = addrspacecast ptr addrspace(5) [[PTR5]] to ptr
  ;CHECK-NEXT: [[VR21:%.*]] = call ptr @__asan_memset(ptr [[VR20]], i32 1, i64 128)
  ;CHECK-NEXT: ret void
  call void @llvm.memset.p0.i64(ptr %ptr0, i8 1, i64 128, i1 false)
  call void @llvm.memset.p1.i64(ptr addrspace(1) %ptr1, i8 1, i64 128, i1 false)
  call void @llvm.memset.p2.i64(ptr addrspace(2) %ptr2, i8 1, i64 128, i1 false)
  call void @llvm.memset.p3.i64(ptr addrspace(3) %ptr3, i8 1, i64 128, i1 false)
  call void @llvm.memset.p4.i64(ptr addrspace(4) %ptr4, i8 1, i64 128, i1 false)
  call void @llvm.memset.p5.i64(ptr addrspace(5) %ptr5, i8 1, i64 128, i1 false)

  call void @llvm.memset.p0.i32(ptr %ptr0, i8 1, i32 128, i1 false)
  call void @llvm.memset.p1.i32(ptr addrspace(1) %ptr1, i8 1, i32 128, i1 false)
  call void @llvm.memset.p2.i32(ptr addrspace(2) %ptr2, i8 1, i32 128, i1 false)
  call void @llvm.memset.p3.i32(ptr addrspace(3) %ptr3, i8 1, i32 128, i1 false)
  call void @llvm.memset.p4.i32(ptr addrspace(4) %ptr4, i8 1, i32 128, i1 false)
  call void @llvm.memset.p5.i32(ptr addrspace(5) %ptr5, i8 1, i32 128, i1 false)
  ret void
}

attributes #0 = { sanitize_address }
