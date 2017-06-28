; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}setcc_sgt_true_sext:
; GCN:      v_cmp_le_u32_e{{32|64}} [[CC:[^,]+]], v{{[0-9]+}}, v{{[0-9]+}}
; GCN-NEXT: s_and_saveexec_b64 {{[^,]+}}, [[CC]]
; GCN-NOT:  v_cndmask_

define amdgpu_kernel void @setcc_sgt_true_sext(i32 addrspace(1)* nocapture %arg) {
bb:
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %y = tail call i32 @llvm.amdgcn.workitem.id.y()
  %cmp = icmp ugt i32 %x, %y
  %ext = sext i1 %cmp to i32
  %cond = icmp sgt i32 %ext, -1
  br i1 %cond, label %then, label %endif

then:
  store i32 1, i32 addrspace(1)* %arg, align 4
  br label %endif

endif:
  ret void
}

; GCN-LABEL: {{^}}setcc_sgt_true_sext_swap:
; GCN:      v_cmp_le_u32_e{{32|64}} [[CC:[^,]+]], v{{[0-9]+}}, v{{[0-9]+}}
; GCN-NEXT: s_and_saveexec_b64 {{[^,]+}}, [[CC]]
; GCN-NOT:  v_cndmask_

define amdgpu_kernel void @setcc_sgt_true_sext_swap(i32 addrspace(1)* nocapture %arg) {
bb:
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %y = tail call i32 @llvm.amdgcn.workitem.id.y()
  %cmp = icmp ugt i32 %x, %y
  %ext = sext i1 %cmp to i32
  %cond = icmp slt i32 -1, %ext
  br i1 %cond, label %then, label %endif

then:
  store i32 1, i32 addrspace(1)* %arg, align 4
  br label %endif

endif:
  ret void
}

; GCN-LABEL: {{^}}setcc_ne_true_sext:
; GCN:      v_cmp_le_u32_e{{32|64}} [[CC:[^,]+]], v{{[0-9]+}}, v{{[0-9]+}}
; GCN-NEXT: s_and_saveexec_b64 {{[^,]+}}, [[CC]]
; GCN-NOT:  v_cndmask_

define amdgpu_kernel void @setcc_ne_true_sext(i32 addrspace(1)* nocapture %arg) {
bb:
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %y = tail call i32 @llvm.amdgcn.workitem.id.y()
  %cmp = icmp ugt i32 %x, %y
  %ext = sext i1 %cmp to i32
  %cond = icmp ne i32 %ext, -1
  br i1 %cond, label %then, label %endif

then:
  store i32 1, i32 addrspace(1)* %arg, align 4
  br label %endif

endif:
  ret void
}

; GCN-LABEL: {{^}}setcc_ult_true_sext:
; GCN:      v_cmp_le_u32_e{{32|64}} [[CC:[^,]+]], v{{[0-9]+}}, v{{[0-9]+}}
; GCN-NEXT: s_and_saveexec_b64 {{[^,]+}}, [[CC]]
; GCN-NOT:  v_cndmask_

define amdgpu_kernel void @setcc_ult_true_sext(i32 addrspace(1)* nocapture %arg) {
bb:
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %y = tail call i32 @llvm.amdgcn.workitem.id.y()
  %cmp = icmp ugt i32 %x, %y
  %ext = sext i1 %cmp to i32
  %cond = icmp ult i32 %ext, -1
  br i1 %cond, label %then, label %endif

then:
  store i32 1, i32 addrspace(1)* %arg, align 4
  br label %endif

endif:
  ret void
}

; GCN-LABEL: {{^}}setcc_eq_true_sext:
; GCN:      v_cmp_gt_u32_e{{32|64}} [[CC:[^,]+]], v{{[0-9]+}}, v{{[0-9]+}}
; GCN-NEXT: s_and_saveexec_b64 {{[^,]+}}, [[CC]]
; GCN-NOT:  v_cndmask_

define amdgpu_kernel void @setcc_eq_true_sext(i32 addrspace(1)* nocapture %arg) {
bb:
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %y = tail call i32 @llvm.amdgcn.workitem.id.y()
  %cmp = icmp ugt i32 %x, %y
  %ext = sext i1 %cmp to i32
  %cond = icmp eq i32 %ext, -1
  br i1 %cond, label %then, label %endif

then:
  store i32 1, i32 addrspace(1)* %arg, align 4
  br label %endif

endif:
  ret void
}

; GCN-LABEL: {{^}}setcc_sle_true_sext:
; GCN:      v_cmp_gt_u32_e{{32|64}} [[CC:[^,]+]], v{{[0-9]+}}, v{{[0-9]+}}
; GCN-NEXT: s_and_saveexec_b64 {{[^,]+}}, [[CC]]
; GCN-NOT:  v_cndmask_

define amdgpu_kernel void @setcc_sle_true_sext(i32 addrspace(1)* nocapture %arg) {
bb:
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %y = tail call i32 @llvm.amdgcn.workitem.id.y()
  %cmp = icmp ugt i32 %x, %y
  %ext = sext i1 %cmp to i32
  %cond = icmp sle i32 %ext, -1
  br i1 %cond, label %then, label %endif

then:
  store i32 1, i32 addrspace(1)* %arg, align 4
  br label %endif

endif:
  ret void
}

; GCN-LABEL: {{^}}setcc_uge_true_sext:
; GCN:      v_cmp_gt_u32_e{{32|64}} [[CC:[^,]+]], v{{[0-9]+}}, v{{[0-9]+}}
; GCN-NEXT: s_and_saveexec_b64 {{[^,]+}}, [[CC]]
; GCN-NOT:  v_cndmask_

define amdgpu_kernel void @setcc_uge_true_sext(i32 addrspace(1)* nocapture %arg) {
bb:
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %y = tail call i32 @llvm.amdgcn.workitem.id.y()
  %cmp = icmp ugt i32 %x, %y
  %ext = sext i1 %cmp to i32
  %cond = icmp uge i32 %ext, -1
  br i1 %cond, label %then, label %endif

then:
  store i32 1, i32 addrspace(1)* %arg, align 4
  br label %endif

endif:
  ret void
}

; GCN-LABEL: {{^}}setcc_eq_false_sext:
; GCN:      v_cmp_le_u32_e{{32|64}} [[CC:[^,]+]], v{{[0-9]+}}, v{{[0-9]+}}
; GCN-NEXT: s_and_saveexec_b64 {{[^,]+}}, [[CC]]
; GCN-NOT:  v_cndmask_

define amdgpu_kernel void @setcc_eq_false_sext(i32 addrspace(1)* nocapture %arg) {
bb:
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %y = tail call i32 @llvm.amdgcn.workitem.id.y()
  %cmp = icmp ugt i32 %x, %y
  %ext = sext i1 %cmp to i32
  %cond = icmp eq i32 %ext, 0
  br i1 %cond, label %then, label %endif

then:
  store i32 1, i32 addrspace(1)* %arg, align 4
  br label %endif

endif:
  ret void
}

; GCN-LABEL: {{^}}setcc_sge_false_sext:
; GCN:      v_cmp_le_u32_e{{32|64}} [[CC:[^,]+]], v{{[0-9]+}}, v{{[0-9]+}}
; GCN-NEXT: s_and_saveexec_b64 {{[^,]+}}, [[CC]]
; GCN-NOT:  v_cndmask_

define amdgpu_kernel void @setcc_sge_false_sext(i32 addrspace(1)* nocapture %arg) {
bb:
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %y = tail call i32 @llvm.amdgcn.workitem.id.y()
  %cmp = icmp ugt i32 %x, %y
  %ext = sext i1 %cmp to i32
  %cond = icmp sge i32 %ext, 0
  br i1 %cond, label %then, label %endif

then:
  store i32 1, i32 addrspace(1)* %arg, align 4
  br label %endif

endif:
  ret void
}

; GCN-LABEL: {{^}}setcc_ule_false_sext:
; GCN:      v_cmp_le_u32_e{{32|64}} [[CC:[^,]+]], v{{[0-9]+}}, v{{[0-9]+}}
; GCN-NEXT: s_and_saveexec_b64 {{[^,]+}}, [[CC]]
; GCN-NOT:  v_cndmask_

define amdgpu_kernel void @setcc_ule_false_sext(i32 addrspace(1)* nocapture %arg) {
bb:
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %y = tail call i32 @llvm.amdgcn.workitem.id.y()
  %cmp = icmp ugt i32 %x, %y
  %ext = sext i1 %cmp to i32
  %cond = icmp ule i32 %ext, 0
  br i1 %cond, label %then, label %endif

then:
  store i32 1, i32 addrspace(1)* %arg, align 4
  br label %endif

endif:
  ret void
}

; GCN-LABEL: {{^}}setcc_ne_false_sext:
; GCN:      v_cmp_gt_u32_e{{32|64}} [[CC:[^,]+]], v{{[0-9]+}}, v{{[0-9]+}}
; GCN-NEXT: s_and_saveexec_b64 {{[^,]+}}, [[CC]]
; GCN-NOT:  v_cndmask_

define amdgpu_kernel void @setcc_ne_false_sext(i32 addrspace(1)* nocapture %arg) {
bb:
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %y = tail call i32 @llvm.amdgcn.workitem.id.y()
  %cmp = icmp ugt i32 %x, %y
  %ext = sext i1 %cmp to i32
  %cond = icmp ne i32 %ext, 0
  br i1 %cond, label %then, label %endif

then:
  store i32 1, i32 addrspace(1)* %arg, align 4
  br label %endif

endif:
  ret void
}
; GCN-LABEL: {{^}}setcc_ugt_false_sext:
; GCN:      v_cmp_gt_u32_e{{32|64}} [[CC:[^,]+]], v{{[0-9]+}}, v{{[0-9]+}}
; GCN-NEXT: s_and_saveexec_b64 {{[^,]+}}, [[CC]]
; GCN-NOT:  v_cndmask_

define amdgpu_kernel void @setcc_ugt_false_sext(i32 addrspace(1)* nocapture %arg) {
bb:
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %y = tail call i32 @llvm.amdgcn.workitem.id.y()
  %cmp = icmp ugt i32 %x, %y
  %ext = sext i1 %cmp to i32
  %cond = icmp ugt i32 %ext, 0
  br i1 %cond, label %then, label %endif

then:
  store i32 1, i32 addrspace(1)* %arg, align 4
  br label %endif

endif:
  ret void
}
; GCN-LABEL: {{^}}setcc_slt_false_sext:
; GCN:      v_cmp_gt_u32_e{{32|64}} [[CC:[^,]+]], v{{[0-9]+}}, v{{[0-9]+}}
; GCN-NEXT: s_and_saveexec_b64 {{[^,]+}}, [[CC]]
; GCN-NOT:  v_cndmask_

define amdgpu_kernel void @setcc_slt_false_sext(i32 addrspace(1)* nocapture %arg) {
bb:
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %y = tail call i32 @llvm.amdgcn.workitem.id.y()
  %cmp = icmp ugt i32 %x, %y
  %ext = sext i1 %cmp to i32
  %cond = icmp slt i32 %ext, 0
  br i1 %cond, label %then, label %endif

then:
  store i32 1, i32 addrspace(1)* %arg, align 4
  br label %endif

endif:
  ret void
}


declare i32 @llvm.amdgcn.workitem.id.x() #0

declare i32 @llvm.amdgcn.workitem.id.y() #0

attributes #0 = { nounwind readnone speculatable }
