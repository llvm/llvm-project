; RUN: llc -march=amdgcn -mcpu=fiji -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI -check-prefix=VIPLUS %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GFX9 -check-prefix=VIPLUS %s

; FIXME: Need to handle non-uniform case for function below (load without gep).
; GCN-LABEL: {{^}}v_test_imax_sge_i16:
; VIPLUS: v_max_i16_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @v_test_imax_sge_i16(i16 addrspace(1)* %out, i16 addrspace(1)* %aptr, i16 addrspace(1)* %bptr) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep0 = getelementptr i16, i16 addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr i16, i16 addrspace(1)* %bptr, i32 %tid
  %outgep = getelementptr i16, i16 addrspace(1)* %out, i32 %tid
  %a = load i16, i16 addrspace(1)* %gep0, align 4
  %b = load i16, i16 addrspace(1)* %gep1, align 4
  %cmp = icmp sge i16 %a, %b
  %val = select i1 %cmp, i16 %a, i16 %b
  store i16 %val, i16 addrspace(1)* %outgep, align 4
  ret void
}

; FIXME: Need to handle non-uniform case for function below (load without gep).
; GCN-LABEL: {{^}}v_test_imax_sge_v2i16:
; VI: v_max_i16_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; VI: v_max_i16_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1

; GFX9: v_pk_max_i16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @v_test_imax_sge_v2i16(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %aptr, <2 x i16> addrspace(1)* %bptr) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep0 = getelementptr <2 x i16>, <2 x i16> addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr <2 x i16>, <2 x i16> addrspace(1)* %bptr, i32 %tid
  %outgep = getelementptr <2 x i16>, <2 x i16> addrspace(1)* %out, i32 %tid
  %a = load <2 x i16>, <2 x i16> addrspace(1)* %gep0, align 4
  %b = load <2 x i16>, <2 x i16> addrspace(1)* %gep1, align 4
  %cmp = icmp sge <2 x i16> %a, %b
  %val = select <2 x i1> %cmp, <2 x i16> %a, <2 x i16> %b
  store <2 x i16> %val, <2 x i16> addrspace(1)* %outgep, align 4
  ret void
}

; FIXME: Need to handle non-uniform case for function below (load without gep).
; GCN-LABEL: {{^}}v_test_imax_sge_v3i16:
; VI: v_max_i16_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; VI: v_max_i16_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; VI: v_max_i16_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; VI-NOT: v_max_i16

; GFX9: v_pk_max_i16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GFX9: v_pk_max_i16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @v_test_imax_sge_v3i16(<3 x i16> addrspace(1)* %out, <3 x i16> addrspace(1)* %aptr, <3 x i16> addrspace(1)* %bptr) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep0 = getelementptr <3 x i16>, <3 x i16> addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr <3 x i16>, <3 x i16> addrspace(1)* %bptr, i32 %tid
  %outgep = getelementptr <3 x i16>, <3 x i16> addrspace(1)* %out, i32 %tid
  %a = load <3 x i16>, <3 x i16> addrspace(1)* %gep0, align 4
  %b = load <3 x i16>, <3 x i16> addrspace(1)* %gep1, align 4
  %cmp = icmp sge <3 x i16> %a, %b
  %val = select <3 x i1> %cmp, <3 x i16> %a, <3 x i16> %b
  store <3 x i16> %val, <3 x i16> addrspace(1)* %outgep, align 4
  ret void
}

; FIXME: Need to handle non-uniform case for function below (load without gep).
; GCN-LABEL: {{^}}v_test_imax_sge_v4i16:
; VI: v_max_i16_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; VI: v_max_i16_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; VI: v_max_i16_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; VI: v_max_i16_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1

; GFX9: v_pk_max_i16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GFX9: v_pk_max_i16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @v_test_imax_sge_v4i16(<4 x i16> addrspace(1)* %out, <4 x i16> addrspace(1)* %aptr, <4 x i16> addrspace(1)* %bptr) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep0 = getelementptr <4 x i16>, <4 x i16> addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr <4 x i16>, <4 x i16> addrspace(1)* %bptr, i32 %tid
  %outgep = getelementptr <4 x i16>, <4 x i16> addrspace(1)* %out, i32 %tid
  %a = load <4 x i16>, <4 x i16> addrspace(1)* %gep0, align 4
  %b = load <4 x i16>, <4 x i16> addrspace(1)* %gep1, align 4
  %cmp = icmp sge <4 x i16> %a, %b
  %val = select <4 x i1> %cmp, <4 x i16> %a, <4 x i16> %b
  store <4 x i16> %val, <4 x i16> addrspace(1)* %outgep, align 4
  ret void
}

; FIXME: Need to handle non-uniform case for function below (load without gep).
; GCN-LABEL: {{^}}v_test_imax_sgt_i16:
; VIPLUS: v_max_i16_e32
define amdgpu_kernel void @v_test_imax_sgt_i16(i16 addrspace(1)* %out, i16 addrspace(1)* %aptr, i16 addrspace(1)* %bptr) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep0 = getelementptr i16, i16 addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr i16, i16 addrspace(1)* %bptr, i32 %tid
  %outgep = getelementptr i16, i16 addrspace(1)* %out, i32 %tid
  %a = load i16, i16 addrspace(1)* %gep0, align 4
  %b = load i16, i16 addrspace(1)* %gep1, align 4
  %cmp = icmp sgt i16 %a, %b
  %val = select i1 %cmp, i16 %a, i16 %b
  store i16 %val, i16 addrspace(1)* %outgep, align 4
  ret void
}

; FIXME: Need to handle non-uniform case for function below (load without gep).
; GCN-LABEL: {{^}}v_test_umax_uge_i16:
; VIPLUS: v_max_u16_e32
define amdgpu_kernel void @v_test_umax_uge_i16(i16 addrspace(1)* %out, i16 addrspace(1)* %aptr, i16 addrspace(1)* %bptr) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep0 = getelementptr i16, i16 addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr i16, i16 addrspace(1)* %bptr, i32 %tid
  %outgep = getelementptr i16, i16 addrspace(1)* %out, i32 %tid
  %a = load i16, i16 addrspace(1)* %gep0, align 4
  %b = load i16, i16 addrspace(1)* %gep1, align 4
  %cmp = icmp uge i16 %a, %b
  %val = select i1 %cmp, i16 %a, i16 %b
  store i16 %val, i16 addrspace(1)* %outgep, align 4
  ret void
}

; FIXME: Need to handle non-uniform case for function below (load without gep).
; GCN-LABEL: {{^}}v_test_umax_ugt_i16:
; VIPLUS: v_max_u16_e32
define amdgpu_kernel void @v_test_umax_ugt_i16(i16 addrspace(1)* %out, i16 addrspace(1)* %aptr, i16 addrspace(1)* %bptr) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep0 = getelementptr i16, i16 addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr i16, i16 addrspace(1)* %bptr, i32 %tid
  %outgep = getelementptr i16, i16 addrspace(1)* %out, i32 %tid
  %a = load i16, i16 addrspace(1)* %gep0, align 4
  %b = load i16, i16 addrspace(1)* %gep1, align 4
  %cmp = icmp ugt i16 %a, %b
  %val = select i1 %cmp, i16 %a, i16 %b
  store i16 %val, i16 addrspace(1)* %outgep, align 4
  ret void
}

; GCN-LABEL: {{^}}v_test_umax_ugt_v2i16:
; VI: v_max_u16_e32
; VI: v_max_u16_sdwa

; GFX9: v_pk_max_u16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @v_test_umax_ugt_v2i16(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %aptr, <2 x i16> addrspace(1)* %bptr) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep0 = getelementptr <2 x i16>, <2 x i16> addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr <2 x i16>, <2 x i16> addrspace(1)* %bptr, i32 %tid
  %outgep = getelementptr <2 x i16>, <2 x i16> addrspace(1)* %out, i32 %tid
  %a = load <2 x i16>, <2 x i16> addrspace(1)* %gep0, align 4
  %b = load <2 x i16>, <2 x i16> addrspace(1)* %gep1, align 4
  %cmp = icmp ugt <2 x i16> %a, %b
  %val = select <2 x i1> %cmp, <2 x i16> %a, <2 x i16> %b
  store <2 x i16> %val, <2 x i16> addrspace(1)* %outgep, align 4
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
