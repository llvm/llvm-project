; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -O3 %s -o - | FileCheck %s

; Two uniform buffer bases feed a byte store selected by a divergent branch.
; The VGPR move of a base must not run between s_or_saveexec and the matching
; s_xor, where exec covers both predecessors. That writes one base into the
; other path's lanes (ROCm/rocm-systems#11305).

; CHECK-LABEL: masker:
; CHECK: s_or_saveexec_b64 [[SAV:s\[[0-9]+:[0-9]+\]]], [[SAV]]
; CHECK-NOT: v_mov_b64
; CHECK: s_xor_b64 exec, exec, [[SAV]]

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"
define amdgpu_kernel void @masker(i32 %0, ptr addrspace(4) %2, ptr addrspace(4) %3, ptr addrspace(4) %4, ptr addrspace(4) %5) {
  %7 = load ptr, ptr addrspace(4) %2, align 8
  %8 = addrspacecast ptr %7 to ptr addrspace(1)
  %9 = load ptr, ptr addrspace(4) %3, align 8
  %10 = addrspacecast ptr %9 to ptr addrspace(1)
  %11 = load ptr, ptr addrspace(4) %4, align 8
  %12 = addrspacecast ptr %11 to ptr addrspace(1)
  %17 = load ptr, ptr addrspace(4) %5, align 8
  %18 = addrspacecast ptr %17 to ptr addrspace(1)
  %34 = tail call i32 @llvm.amdgcn.workitem.id.x()
  %35 = icmp slt i32 %34, %0
  br i1 %35, label %36, label %102
36:
  %39 = and i32 %34, 31
  %40 = lshr i32 %34, 5
  %41 = and i32 %40, 31
  %42 = icmp slt i32 %39, 32
  %43 = icmp slt i32 %41, 32
  %44 = select i1 %42, i1 %43, i1 false
  br i1 %44, label %45, label %102
45:
  %46 = sext i32 %34 to i64
  %47 = getelementptr inbounds i8, ptr addrspace(1) %8, i64 %46
  %50 = add nsw i32 %41, %39
  %51 = sext i32 %50 to i64
  %52 = getelementptr inbounds i8, ptr addrspace(1) %12, i64 %51
  %56 = getelementptr inbounds i8, ptr addrspace(1) %10, i64 %46
  %57 = add nuw nsw i32 %39, 1
  %58 = load i8, ptr addrspace(1) %47, align 1
  store i8 %58, ptr addrspace(1) %52, align 1
  %59 = icmp eq i32 %39, 0
  br i1 %59, label %60, label %64
60:
  %61 = sext i32 %50 to i64
  %62 = getelementptr inbounds i8, ptr addrspace(1) %18, i64 %61
  store i8 1, ptr addrspace(1) %62, align 1
  %63 = load i8, ptr addrspace(1) %47, align 1
  br label %77
64:
  %65 = load i8, ptr addrspace(1) %56, align 1
  %66 = icmp eq i8 %65, 0
  br i1 %66, label %77, label %67
67:
  %69 = add i32 %41, %57
  %70 = sext i32 %69 to i64
  %71 = getelementptr inbounds i8, ptr addrspace(1) %18, i64 %70
  store i8 1, ptr addrspace(1) %71, align 1
  %72 = load i8, ptr addrspace(1) %47, align 1
  %74 = add nsw i32 %57, %41
  %75 = sext i32 %74 to i64
  %76 = getelementptr inbounds i8, ptr addrspace(1) %12, i64 %75
  store i8 %72, ptr addrspace(1) %76, align 1
  br label %77
77:
  %78 = phi i8 [ %58, %64 ], [ %72, %67 ], [ %63, %60 ]
  %80 = add nsw i32 %39, -1
  store i8 %78, ptr addrspace(1) %52, align 1
  %81 = icmp samesign ult i32 %57, 32
  br i1 %81, label %84, label %82
82:
  %83 = add i32 %50, 1
  br label %96
84:
  %85 = load i8, ptr addrspace(1) %56, align 1
  %86 = icmp eq i8 %85, 0
  br i1 %86, label %102, label %87
87:
  %89 = add i32 1, %41
  %90 = add i32 %89, %80
  %91 = sext i32 %90 to i64
  %92 = getelementptr inbounds i8, ptr addrspace(1) %18, i64 %91
  store i8 1, ptr addrspace(1) %92, align 1
  %93 = load i8, ptr addrspace(1) %47, align 1
  %95 = add nsw i32 %80, %41
  br label %96
96:
  %97 = phi i32 [ %95, %87 ], [ %83, %82 ]
  %98 = phi ptr addrspace(1) [ %12, %87 ], [ %18, %82 ]
  %99 = phi i8 [ %93, %87 ], [ 1, %82 ]
  %100 = sext i32 %97 to i64
  %101 = getelementptr inbounds i8, ptr addrspace(1) %98, i64 %100
  store i8 %99, ptr addrspace(1) %101, align 1
  br label %102
102:
  ret void
}
declare i32 @llvm.amdgcn.workitem.id.x()
