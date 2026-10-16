; RUN: opt -S -mtriple=amdgpu12.50-amd-amdhsa -passes=slp-vectorizer < %s | FileCheck -check-prefixes=GCN,GFX1250 %s
; RUN: opt -S -mtriple=amdgpu12.51-amd-amdhsa -passes=slp-vectorizer < %s | FileCheck -check-prefixes=GCN,GFX1251 %s

; GCN-LABEL: @add_combine
; GFX1250: add i64
; GFX1250: add i64
; GFX1251: add <2 x i64>
define amdgpu_kernel void @add_combine(ptr addrspace(1) %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = getelementptr inbounds i64, ptr addrspace(1) %arg, i64 %tmp1
  %tmp3 = load i64, ptr addrspace(1) %tmp2, align 8
  %tmp4 = add i64 %tmp3, 1
  store i64 %tmp4, ptr addrspace(1) %tmp2, align 8
  %tmp5 = add nuw nsw i64 %tmp1, 1
  %tmp6 = getelementptr inbounds i64, ptr addrspace(1) %arg, i64 %tmp5
  %tmp7 = load i64, ptr addrspace(1) %tmp6, align 8
  %tmp8 = add i64 %tmp7, 1
  store i64 %tmp8, ptr addrspace(1) %tmp6, align 8
  ret void
}

; GCN-LABEL: @sub_combine
; GFX1250: sub i64
; GFX1250: sub i64
; GFX1251: sub <2 x i64>
define amdgpu_kernel void @sub_combine(ptr addrspace(1) %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = getelementptr inbounds i64, ptr addrspace(1) %arg, i64 %tmp1
  %tmp3 = load i64, ptr addrspace(1) %tmp2, align 8
  %tmp4 = sub i64 %tmp3, 1
  store i64 %tmp4, ptr addrspace(1) %tmp2, align 8
  %tmp5 = add nuw nsw i64 %tmp1, 1
  %tmp6 = getelementptr inbounds i64, ptr addrspace(1) %arg, i64 %tmp5
  %tmp7 = load i64, ptr addrspace(1) %tmp6, align 8
  %tmp8 = sub i64 %tmp7, 1
  store i64 %tmp8, ptr addrspace(1) %tmp6, align 8
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
