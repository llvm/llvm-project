; RUN: llc < %s -march=amdgcn -mcpu=fiji -mattr=-flat-for-global -verify-machineinstrs | FileCheck -check-prefix=GCN -check-prefix=VI %s


declare i32 @llvm.amdgcn.workitem.id.x() nounwind readnone

; FIXME: Need to handle non-uniform case for function below (load without gep).
; GCN-LABEL: {{^}}v_test_imax_sge_i16:
; VI: v_max_i16_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define void @v_test_imax_sge_i16(i16 addrspace(1)* %out, i16 addrspace(1)* %aptr, i16 addrspace(1)* %bptr) nounwind {
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
; GCN-LABEL: {{^}}v_test_imax_sge_v4i16:
; VI: v_max_i16_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; VI: v_max_i16_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; VI: v_max_i16_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; VI: v_max_i16_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define void @v_test_imax_sge_v4i16(<4 x i16> addrspace(1)* %out, <4 x i16> addrspace(1)* %aptr, <4 x i16> addrspace(1)* %bptr) nounwind {
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
; VI: v_max_i16_e32
define void @v_test_imax_sgt_i16(i16 addrspace(1)* %out, i16 addrspace(1)* %aptr, i16 addrspace(1)* %bptr) nounwind {
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
; VI: v_max_u16_e32
define void @v_test_umax_uge_i16(i16 addrspace(1)* %out, i16 addrspace(1)* %aptr, i16 addrspace(1)* %bptr) nounwind {
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
; VI: v_max_u16_e32
define void @v_test_umax_ugt_i16(i16 addrspace(1)* %out, i16 addrspace(1)* %aptr, i16 addrspace(1)* %bptr) nounwind {
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
