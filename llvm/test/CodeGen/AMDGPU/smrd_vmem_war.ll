; RUN: llc  -mtriple=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck %s -check-prefix=GCN

; GCN-LABEL: ; %bb.0:
; GCN: s_load_dword s{{[0-9]+}}, s[[[ADDR_LO:[0-9]+]]{{\:}}[[ADDR_HI:[0-9]+]]], 0x0
; GCN: s_waitcnt lgkmcnt(0)
; GCN: global_store_dword v

define amdgpu_kernel void @zot(ptr addrspace(1) nocapture %arg, ptr addrspace(1) nocapture %arg1) {
bb:
  %tmp = call i32 @llvm.amdgcn.workitem.id.x()
  %tmp2 = icmp eq i32 %tmp, 0
  br i1 %tmp2, label %bb3, label %bb8

bb3:                                              ; preds = %bb
  %tmp4 = load i32, ptr addrspace(1) %arg, align 4
  store i32 0, ptr addrspace(1) %arg, align 4
  %tmp5 = zext i32 %tmp4 to i64
  %tmp6 = load i64, ptr addrspace(1) %arg1, align 8
  %tmp7 = add i64 %tmp6, %tmp5
  store i64 %tmp7, ptr addrspace(1) %arg1, align 8
  br label %bb8

bb8:                                              ; preds = %bb3, %bb
  ret void
}
; Function Attrs: nounwind readnone speculatable
declare i32 @llvm.amdgcn.workitem.id.x() #0

attributes #0 = { nounwind readnone speculatable }
