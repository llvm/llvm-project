; RUN: llc -mtriple=amdgcn -stop-after=amdgpu-isel < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn -enable-new-pm -stop-after=amdgpu-isel < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: name:            s_ctpop_i32
; GCN: S_BCNT1_I32_B32
define amdgpu_kernel void @s_ctpop_i32(ptr addrspace(1) noalias %out, i32 %val) nounwind {
  %ctpop = call i32 @llvm.ctpop.i32(i32 %val) nounwind readnone
  store i32 %ctpop, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: name:            s_ctpop_i64
; GCN: %[[BCNT:[0-9]+]]:sreg_32 = S_BCNT1_I32_B64
; GCN: %[[SREG1:[0-9]+]]:sreg_32 = COPY killed %[[BCNT]]
; GCN: %[[SREG2:[0-9]+]]:sreg_32 = S_MOV_B32 0
; GCN: REG_SEQUENCE killed %[[SREG1]], %subreg.sub0, killed %[[SREG2]], %subreg.sub1
define amdgpu_kernel void @s_ctpop_i64(ptr addrspace(1) noalias %out, i64 %val) nounwind {
  %ctpop = call i64 @llvm.ctpop.i64(i64 %val) nounwind readnone
  %truncctpop = trunc i64 %ctpop to i32
  store i32 %truncctpop, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: name:            v_ctpop_i32
; GCN: V_BCNT_U32_B32_e64
define amdgpu_kernel void @v_ctpop_i32(ptr addrspace(1) noalias %out, ptr addrspace(1) noalias %in) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %in.gep = getelementptr i32, ptr addrspace(1) %in, i32 %tid
  %val = load i32, ptr addrspace(1) %in.gep, align 4
  %ctpop = call i32 @llvm.ctpop.i32(i32 %val) nounwind readnone
  store i32 %ctpop, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: name:            v_ctpop_i64
; GCN: %[[BCNT1:[0-9]+]]:vgpr_32 = V_BCNT_U32_B32_e64 killed %{{[0-9]+}}, 0, implicit $exec
; GCN: %[[BCNT2:[0-9]+]]:vgpr_32 = V_BCNT_U32_B32_e64 killed %{{[0-9]+}}, killed %[[BCNT1]], implicit $exec
; GCN: %[[VGPR1:[0-9]+]]:vgpr_32 = V_MOV_B32_e32 0, implicit $exec
; GCN: REG_SEQUENCE killed %[[BCNT2]], %subreg.sub0, killed %[[VGPR1]], %subreg.sub1
define amdgpu_kernel void @v_ctpop_i64(ptr addrspace(1) noalias %out, ptr addrspace(1) noalias %in) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %in.gep = getelementptr i64, ptr addrspace(1) %in, i32 %tid
  %val = load i64, ptr addrspace(1) %in.gep, align 8
  %ctpop = call i64 @llvm.ctpop.i64(i64 %val) nounwind readnone
  %truncctpop = trunc i64 %ctpop to i32
  store i32 %truncctpop, ptr addrspace(1) %out, align 4
  ret void
}

declare i64 @llvm.ctpop.i64(i64) nounwind readnone

declare i32 @llvm.ctpop.i32(i32) nounwind readnone

declare i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
