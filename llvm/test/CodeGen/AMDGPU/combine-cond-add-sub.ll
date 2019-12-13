; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=gfx902  -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GFX9 %s

; GCN-LABEL: {{^}}add1:
; GCN: v_cmp_gt_u32_e{{32|64}} [[CC:[^,]+]], v{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_addc_u32_e{{32|64}} v{{[0-9]+}}, {{[^,]+}}, 0, v{{[0-9]+}}, [[CC]]
; GCN-NOT: v_cndmask

; GFX9-LABEL: {{^}}add1:
; GFX9: v_addc_co_u32_e{{32|64}} v{{[0-9]+}}, vcc
define amdgpu_kernel void @add1(i32 addrspace(1)* nocapture %arg) {
bb:
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %y = tail call i32 @llvm.amdgcn.workitem.id.y()
  %gep = getelementptr inbounds i32, i32 addrspace(1)* %arg, i32 %x
  %v = load i32, i32 addrspace(1)* %gep, align 4
  %cmp = icmp ugt i32 %x, %y
  %ext = zext i1 %cmp to i32
  %add = add i32 %v, %ext
  store i32 %add, i32 addrspace(1)* %gep, align 4
  ret void
}

; GCN-LABEL: {{^}}add1_i16:
; GCN: v_cmp_gt_u32_e{{32|64}} [[CC:[^,]+]], v{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_addc_u32_e{{32|64}} v{{[0-9]+}}, {{[^,]+}}, 0, v{{[0-9]+}}, [[CC]]
; GCN-NOT: v_cndmask

; GFX9-LABEL: {{^}}add1_i16:
; GFX9: v_addc_co_u32_e{{32|64}} v{{[0-9]+}}, vcc
define i16 @add1_i16(i32 addrspace(1)* nocapture %arg, i16 addrspace(1)* nocapture %dst) {
bb:
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %y = tail call i32 @llvm.amdgcn.workitem.id.y()
  %gep = getelementptr inbounds i32, i32 addrspace(1)* %arg, i32 %x
  %v = load i32, i32 addrspace(1)* %gep, align 4
  %cmp = icmp ugt i32 %x, %y
  %ext = zext i1 %cmp to i32
  %add = add i32 %v, %ext
  %trunc = trunc i32 %add to i16
  ret i16 %trunc
}

; GCN-LABEL: {{^}}sub1:
; GCN: v_cmp_gt_u32_e32 vcc, v{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_subbrev_u32_e32 v{{[0-9]+}}, vcc, 0, v{{[0-9]+}}, vcc
; GCN-NOT: v_cndmask

; GFX9-LABEL: {{^}}sub1:
; GFX9: v_subbrev_co_u32_e{{32|64}} v{{[0-9]+}}, vcc
define amdgpu_kernel void @sub1(i32 addrspace(1)* nocapture %arg) {
bb:
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %y = tail call i32 @llvm.amdgcn.workitem.id.y()
  %gep = getelementptr inbounds i32, i32 addrspace(1)* %arg, i32 %x
  %v = load i32, i32 addrspace(1)* %gep, align 4
  %cmp = icmp ugt i32 %x, %y
  %ext = sext i1 %cmp to i32
  %add = add i32 %v, %ext
  store i32 %add, i32 addrspace(1)* %gep, align 4
  ret void
}

; GCN-LABEL: {{^}}add_adde:
; GCN: v_cmp_gt_u32_e{{32|64}} [[CC:[^,]+]], v{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_addc_u32_e{{32|64}} v{{[0-9]+}}, {{[^,]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[CC]]
; GCN-NOT: v_cndmask
; GCN-NOT: v_add

; GFX9-LABEL: {{^}}add_adde:
; GFX9: v_addc_co_u32_e{{32|64}} v{{[0-9]+}}, vcc
define amdgpu_kernel void @add_adde(i32 addrspace(1)* nocapture %arg, i32 %a) {
bb:
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %y = tail call i32 @llvm.amdgcn.workitem.id.y()
  %gep = getelementptr inbounds i32, i32 addrspace(1)* %arg, i32 %x
  %v = load i32, i32 addrspace(1)* %gep, align 4
  %cmp = icmp ugt i32 %x, %y
  %ext = zext i1 %cmp to i32
  %adde = add i32 %v, %ext
  %add2 = add i32 %adde, %a
  store i32 %add2, i32 addrspace(1)* %gep, align 4
  ret void
}

; GCN-LABEL: {{^}}adde_add:
; GCN: v_cmp_gt_u32_e{{32|64}} [[CC:[^,]+]], v{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_addc_u32_e{{32|64}} v{{[0-9]+}}, {{[^,]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[CC]]
; GCN-NOT: v_cndmask
; GCN-NOT: v_add

; GFX9-LABEL: {{^}}adde_add:
; GFX9: v_addc_co_u32_e{{32|64}} v{{[0-9]+}}, vcc
define amdgpu_kernel void @adde_add(i32 addrspace(1)* nocapture %arg, i32 %a) {
bb:
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %y = tail call i32 @llvm.amdgcn.workitem.id.y()
  %gep = getelementptr inbounds i32, i32 addrspace(1)* %arg, i32 %x
  %v = load i32, i32 addrspace(1)* %gep, align 4
  %cmp = icmp ugt i32 %x, %y
  %ext = zext i1 %cmp to i32
  %add = add i32 %v, %a
  %adde = add i32 %add, %ext
  store i32 %adde, i32 addrspace(1)* %gep, align 4
  ret void
}

; GCN-LABEL: {{^}}sub_sube:
; GCN: v_cmp_gt_u32_e{{32|64}} [[CC:[^,]+]], v{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_subb_u32_e{{32|64}} v{{[0-9]+}}, {{[^,]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[CC]]
; GCN-NOT: v_cndmask
; GCN-NOT: v_sub

; GFX9-LABEL: {{^}}sub_sube:
; GFX9: v_subb_co_u32_e{{32|64}} v{{[0-9]+}}, vcc
define amdgpu_kernel void @sub_sube(i32 addrspace(1)* nocapture %arg, i32 %a) {
bb:
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %y = tail call i32 @llvm.amdgcn.workitem.id.y()
  %gep = getelementptr inbounds i32, i32 addrspace(1)* %arg, i32 %x
  %v = load i32, i32 addrspace(1)* %gep, align 4
  %cmp = icmp ugt i32 %x, %y
  %ext = sext i1 %cmp to i32
  %adde = add i32 %v, %ext
  %sub = sub i32 %adde, %a
  store i32 %sub, i32 addrspace(1)* %gep, align 4
  ret void
}

; GCN-LABEL: {{^}}sub_sube_commuted:
; GCN-DAG: v_cmp_gt_u32_e{{32|64}} [[CC:[^,]+]], v{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: buffer_load_dword [[V:v[0-9]+]],
; GCN: v_cndmask_b32_e64 [[CCZEXT:v[0-9]+]], 0, 1, [[CC]]
; GCN: v_sub_i32_e32 [[SUB:v[0-9]+]], vcc, [[CCZEXT]], v4
; GCN: v_add_i32_e32 [[ADD:v[0-9]+]], vcc, {{.*}}, [[SUB]]
; GCN: v_add_i32_e32 {{.*}}, 0x64, [[ADD]]

; GFX9-LABEL: {{^}}sub_sube_commuted:
; GFX9-DAG: v_cmp_gt_u32_e{{32|64}} [[CC:[^,]+]], v{{[0-9]+}}, v{{[0-9]+}}
; GFX9-DAG: global_load_dword [[V:v[0-9]+]],
; GFX9-DAG: v_cndmask_b32_e64 [[CCZEXT:v[0-9]+]], 0, 1, [[CC]]
; GFX9: v_sub_u32_e32 {{.*}}, [[CCZEXT]]
; GFX9: v_add_u32_e32
; GFX9: v_add_u32_e32 {{.*}}, 0x64,
define amdgpu_kernel void @sub_sube_commuted(i32 addrspace(1)* nocapture %arg, i32 %a) {
bb:
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %y = tail call i32 @llvm.amdgcn.workitem.id.y()
  %gep = getelementptr inbounds i32, i32 addrspace(1)* %arg, i32 %x
  %v = load i32, i32 addrspace(1)* %gep, align 4
  %cmp = icmp ugt i32 %x, %y
  %ext = sext i1 %cmp to i32
  %adde = add i32 %v, %ext
  %sub = sub i32 %adde, %a
  %sub2 = sub i32 100, %sub
  store i32 %sub2, i32 addrspace(1)* %gep, align 4
  ret void
}

; GCN-LABEL: {{^}}sube_sub:
; GCN: v_cmp_gt_u32_e{{32|64}} [[CC:[^,]+]], v{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_subb_u32_e{{32|64}} v{{[0-9]+}}, {{[^,]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[CC]]
; GCN-NOT: v_cndmask
; GCN-NOT: v_sub

; GFX9-LABEL: {{^}}sube_sub:
; GFX9: v_subb_co_u32_e{{32|64}} v{{[0-9]+}}, vcc
define amdgpu_kernel void @sube_sub(i32 addrspace(1)* nocapture %arg, i32 %a) {
bb:
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %y = tail call i32 @llvm.amdgcn.workitem.id.y()
  %gep = getelementptr inbounds i32, i32 addrspace(1)* %arg, i32 %x
  %v = load i32, i32 addrspace(1)* %gep, align 4
  %cmp = icmp ugt i32 %x, %y
  %ext = sext i1 %cmp to i32
  %sub = sub i32 %v, %a
  %adde = add i32 %sub, %ext
  store i32 %adde, i32 addrspace(1)* %gep, align 4
  ret void
}

; GCN-LABEL: {{^}}zext_flclass:
; GCN: v_cmp_class_f32_e{{32|64}} [[CC:[^,]+]],
; GCN: v_addc_u32_e{{32|64}} v{{[0-9]+}}, {{[^,]+}}, 0, v{{[0-9]+}}, [[CC]]
; GCN-NOT: v_cndmask

; GFX9-LABEL: {{^}}zext_flclass:
; GFX9: v_addc_co_u32_e{{32|64}} v{{[0-9]+}}, vcc
define amdgpu_kernel void @zext_flclass(i32 addrspace(1)* nocapture %arg, float %x) {
bb:
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i32, i32 addrspace(1)* %arg, i32 %id
  %v = load i32, i32 addrspace(1)* %gep, align 4
  %cmp = tail call zeroext i1 @llvm.amdgcn.class.f32(float %x, i32 608)
  %ext = zext i1 %cmp to i32
  %add = add i32 %v, %ext
  store i32 %add, i32 addrspace(1)* %gep, align 4
  ret void
}

; GCN-LABEL: {{^}}sext_flclass:
; GCN: v_cmp_class_f32_e32 vcc,
; GCN: v_subbrev_u32_e32 v{{[0-9]+}}, vcc, 0, v{{[0-9]+}}, vcc
; GCN-NOT: v_cndmask

; GFX9-LABEL: {{^}}sext_flclass:
; GFX9: v_subbrev_co_u32_e32 v{{[0-9]+}}, vcc
define amdgpu_kernel void @sext_flclass(i32 addrspace(1)* nocapture %arg, float %x) {
bb:
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i32, i32 addrspace(1)* %arg, i32 %id
  %v = load i32, i32 addrspace(1)* %gep, align 4
  %cmp = tail call zeroext i1 @llvm.amdgcn.class.f32(float %x, i32 608)
  %ext = sext i1 %cmp to i32
  %add = add i32 %v, %ext
  store i32 %add, i32 addrspace(1)* %gep, align 4
  ret void
}

; GCN-LABEL: {{^}}add_and:
; GCN: s_and_b64 [[CC:[^,]+]],
; GCN: v_addc_u32_e{{32|64}} v{{[0-9]+}}, {{[^,]+}}, 0, v{{[0-9]+}}, [[CC]]
; GCN-NOT: v_cndmask

; GFX9-LABEL: {{^}}add_and:
; GFX9: v_addc_co_u32_e{{32|64}} v{{[0-9]+}}, vcc
define amdgpu_kernel void @add_and(i32 addrspace(1)* nocapture %arg) {
bb:
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %y = tail call i32 @llvm.amdgcn.workitem.id.y()
  %gep = getelementptr inbounds i32, i32 addrspace(1)* %arg, i32 %x
  %v = load i32, i32 addrspace(1)* %gep, align 4
  %cmp1 = icmp ugt i32 %x, %y
  %cmp2 = icmp ugt i32 %x, 1
  %cmp = and i1 %cmp1, %cmp2
  %ext = zext i1 %cmp to i32
  %add = add i32 %v, %ext
  store i32 %add, i32 addrspace(1)* %gep, align 4
  ret void
}

; sub x, sext (setcc) => addcarry x, 0, setcc
; GCN-LABEL: {{^}}cmp_sub_sext:
; GCN: v_cmp_gt_u32_e32 vcc, v
; GCN-NOT: vcc
; GCN: v_addc_u32_e32 [[RESULT:v[0-9]+]], vcc, 0, v{{[0-9]+}}, vcc
define amdgpu_kernel void @cmp_sub_sext(i32 addrspace(1)* nocapture %arg) {
bb:
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %y = tail call i32 @llvm.amdgcn.workitem.id.y()
  %gep = getelementptr inbounds i32, i32 addrspace(1)* %arg, i32 %x
  %v = load i32, i32 addrspace(1)* %gep, align 4
  %cmp = icmp ugt i32 %x, %y
  %ext = sext i1 %cmp to i32
  %add = sub i32 %v, %ext
  store i32 %add, i32 addrspace(1)* %gep, align 4
  ret void
}

; sub x, zext (setcc) => subcarry x, 0, setcc
; GCN-LABEL: {{^}}cmp_sub_zext:
; GCN: v_cmp_gt_u32_e32 vcc, v
; GCN-NOT: vcc
; GCN: v_subbrev_u32_e32 [[RESULT:v[0-9]+]], vcc, 0, v{{[0-9]+}}, vcc
define amdgpu_kernel void @cmp_sub_zext(i32 addrspace(1)* nocapture %arg) {
bb:
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %y = tail call i32 @llvm.amdgcn.workitem.id.y()
  %gep = getelementptr inbounds i32, i32 addrspace(1)* %arg, i32 %x
  %v = load i32, i32 addrspace(1)* %gep, align 4
  %cmp = icmp ugt i32 %x, %y
  %ext = zext i1 %cmp to i32
  %add = sub i32 %v, %ext
  store i32 %add, i32 addrspace(1)* %gep, align 4
  ret void
}

; GCN-LABEL: {{^}}sub_addcarry:
; GCN: v_cmp_gt_u32_e32 vcc, v
; GCN-NOT: vcc
; GCN: v_addc_u32_e32 [[ADDC:v[0-9]+]], vcc, 0, v{{[0-9]+}}, vcc
; GCN-NOT: vcc
; GCN: v_subrev_i32_e32 [[RESULT:v[0-9]+]], vcc,
define amdgpu_kernel void @sub_addcarry(i32 addrspace(1)* nocapture %arg, i32 %a) {
bb:
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %y = tail call i32 @llvm.amdgcn.workitem.id.y()
  %gep = getelementptr inbounds i32, i32 addrspace(1)* %arg, i32 %x
  %v = load i32, i32 addrspace(1)* %gep, align 4
  %cmp = icmp ugt i32 %x, %y
  %ext = zext i1 %cmp to i32
  %adde = add i32 %v, %ext
  %add2 = sub i32 %adde, %a
  store i32 %add2, i32 addrspace(1)* %gep, align 4
  ret void
}

; GCN-LABEL: {{^}}sub_subcarry:
; GCN: v_cmp_gt_u32_e32 vcc, v
; GCN-NOT: vcc
; GCN: v_subb_u32_e32 [[RESULT:v[0-9]+]], vcc, v{{[0-9]+}}, v{{[0-9]+}}, vcc
define amdgpu_kernel void @sub_subcarry(i32 addrspace(1)* nocapture %arg, i32 %a) {
bb:
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %y = tail call i32 @llvm.amdgcn.workitem.id.y()
  %gep = getelementptr inbounds i32, i32 addrspace(1)* %arg, i32 %x
  %v = load i32, i32 addrspace(1)* %gep, align 4
  %cmp = icmp ugt i32 %x, %y
  %ext = zext i1 %cmp to i32
  %adde = sub i32 %v, %ext
  %add2 = sub i32 %adde, %a
  store i32 %add2, i32 addrspace(1)* %gep, align 4
  ret void
}

; Check case where sub is commuted with zext
; GCN-LABEL: {{^}}sub_zext_setcc_commute:
; GCN: v_cmp_gt_u32_e32 vcc, v
; GCN: v_cndmask
; GCN: v_sub_i32_e32
; GCN: v_add_i32_e32 [[ADD:v[0-9]+]], vcc,
; GCN: v_subrev_i32_e32 [[RESULT:v[0-9]+]], vcc, s{{[0-9]+}}, [[ADD]]
define amdgpu_kernel void @sub_zext_setcc_commute(i32 addrspace(1)* nocapture %arg, i32 %a, i32%b) {
bb:
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %y = tail call i32 @llvm.amdgcn.workitem.id.y()
  %gep = getelementptr inbounds i32, i32 addrspace(1)* %arg, i32 %x
  %v = load i32, i32 addrspace(1)* %gep, align 4
  %cmp = icmp ugt i32 %x, %y
  %ext = zext i1 %cmp to i32
  %adde = sub i32 %v, %ext
  %sub = sub i32 %a, %adde
  %sub2 = sub i32 %sub, %b
  store i32 %sub2, i32 addrspace(1)* %gep, align 4
  ret void
}

; Check case where sub is commuted with sext
; GCN-LABEL: {{^}}sub_sext_setcc_commute:
; GCN: v_cmp_gt_u32_e32 vcc, v
; GCN: v_cndmask
; GCN: v_sub_i32_e32
; GCN: v_add_i32_e32 [[ADD:v[0-9]+]], vcc,
; GCN: v_subrev_i32_e32 [[RESULT:v[0-9]+]], vcc, s{{[0-9]+}}, [[ADD]]
define amdgpu_kernel void @sub_sext_setcc_commute(i32 addrspace(1)* nocapture %arg, i32 %a, i32%b) {
bb:
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %y = tail call i32 @llvm.amdgcn.workitem.id.y()
  %gep = getelementptr inbounds i32, i32 addrspace(1)* %arg, i32 %x
  %v = load i32, i32 addrspace(1)* %gep, align 4
  %cmp = icmp ugt i32 %x, %y
  %ext = sext i1 %cmp to i32
  %adde = sub i32 %v, %ext
  %sub = sub i32 %a, %adde
  %sub2 = sub i32 %sub, %b
  store i32 %sub2, i32 addrspace(1)* %gep, align 4
  ret void
}

declare i1 @llvm.amdgcn.class.f32(float, i32) #0

declare i32 @llvm.amdgcn.workitem.id.x() #0

declare i32 @llvm.amdgcn.workitem.id.y() #0

attributes #0 = { nounwind readnone speculatable }
