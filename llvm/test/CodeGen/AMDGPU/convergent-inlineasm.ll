; RUN: llc -mtriple=amdgcn--amdhsa -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

declare i32 @llvm.amdgcn.workitem.id.x() #0
; GCN-LABEL: {{^}}convergent_inlineasm:
; GCN: BB#0:
; GCN: v_cmp_ne_u32_e64
; GCN: ; mask branch
; GCN: BB{{[0-9]+_[0-9]+}}:
define amdgpu_kernel void @convergent_inlineasm(i64 addrspace(1)* nocapture %arg) {
bb:
  %tmp = call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = tail call i64 asm "v_cmp_ne_u32_e64 $0, 0, $1", "=s,v"(i32 1) #1
  %tmp2 = icmp eq i32 %tmp, 8
  br i1 %tmp2, label %bb3, label %bb5

bb3:                                              ; preds = %bb
  %tmp4 = getelementptr i64, i64 addrspace(1)* %arg, i32 %tmp
  store i64 %tmp1, i64 addrspace(1)* %arg, align 8
  br label %bb5

bb5:                                              ; preds = %bb3, %bb
  ret void
}

; GCN-LABEL: {{^}}nonconvergent_inlineasm:
; GCN: ; mask branch

; GCN: BB{{[0-9]+_[0-9]+}}:
; GCN: v_cmp_ne_u32_e64

; GCN: BB{{[0-9]+_[0-9]+}}:

define amdgpu_kernel void @nonconvergent_inlineasm(i64 addrspace(1)* nocapture %arg) {
bb:
  %tmp = call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = tail call i64 asm "v_cmp_ne_u32_e64 $0, 0, $1", "=s,v"(i32 1)
  %tmp2 = icmp eq i32 %tmp, 8
  br i1 %tmp2, label %bb3, label %bb5

bb3:                                              ; preds = %bb
  %tmp4 = getelementptr i64, i64 addrspace(1)* %arg, i32 %tmp
  store i64 %tmp1, i64 addrspace(1)* %arg, align 8
  br label %bb5

bb5:                                              ; preds = %bb3, %bb
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { convergent nounwind readnone }
