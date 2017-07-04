; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}any_extend_vector_inreg_v16i8_to_v4i32:
; GCN: s_load_dwordx4
; GCN-DAG: s_load_dwordx4
; GCN-DAG: s_load_dword

; GCN: {{buffer|flat}}_store_byte
; GCN: {{buffer|flat}}_store_byte
; GCN: {{buffer|flat}}_store_byte
; GCN: {{buffer|flat}}_store_byte

; GCN: {{buffer|flat}}_store_byte
; GCN: {{buffer|flat}}_store_byte
; GCN: {{buffer|flat}}_store_byte
; GCN: {{buffer|flat}}_store_byte

; GCN: {{buffer|flat}}_store_byte
; GCN: {{buffer|flat}}_store_byte
; GCN: {{buffer|flat}}_store_byte
; GCN: {{buffer|flat}}_store_byte

; GCN: {{buffer|flat}}_store_byte
; GCN: {{buffer|flat}}_store_byte
; GCN: {{buffer|flat}}_store_byte
; GCN: {{buffer|flat}}_store_byte
define amdgpu_kernel void @any_extend_vector_inreg_v16i8_to_v4i32(<8 x i8> addrspace(1)* nocapture readonly %arg, <16 x i8> addrspace(1)* %arg1) local_unnamed_addr #0 {
bb:
  %tmp = bitcast <8 x i8> addrspace(1)* %arg to <16 x i8> addrspace(1)*
  %tmp2 = load <16 x i8>, <16 x i8> addrspace(1)* %tmp, align 16
  %tmp3 = extractelement <16 x i8> %tmp2, i64 4
  %tmp6 = extractelement <16 x i8> %tmp2, i64 11
  %tmp10 = getelementptr inbounds <8 x i8>, <8 x i8> addrspace(1)* %arg, i64 2
  %tmp11 = bitcast <8 x i8> addrspace(1)* %tmp10 to <16 x i8> addrspace(1)*
  %tmp12 = load <16 x i8>, <16 x i8> addrspace(1)* %tmp11, align 16
  %tmp13 = extractelement <16 x i8> %tmp12, i64 7
  %tmp17 = extractelement <16 x i8> %tmp12, i64 12
  %tmp21 = getelementptr inbounds <8 x i8>, <8 x i8> addrspace(1)* %arg, i64 4
  %tmp22 = bitcast <8 x i8> addrspace(1)* %tmp21 to <16 x i8> addrspace(1)*
  %tmp23 = load <16 x i8>, <16 x i8> addrspace(1)* %tmp22, align 16
  %tmp24 = extractelement <16 x i8> %tmp23, i64 3
  %tmp1 = insertelement <16 x i8> undef, i8 %tmp3, i32 2
  %tmp4 = insertelement <16 x i8> %tmp1, i8 0, i32 3
  %tmp5 = insertelement <16 x i8> %tmp4, i8 0, i32 4
  %tmp7 = insertelement <16 x i8> %tmp5, i8 %tmp6, i32 5
  %tmp8 = insertelement <16 x i8> %tmp7, i8 0, i32 6
  %tmp9 = insertelement <16 x i8> %tmp8, i8 %tmp13, i32 7
  %tmp14 = insertelement <16 x i8> %tmp9, i8 0, i32 8
  %tmp15 = insertelement <16 x i8> %tmp14, i8 %tmp17, i32 9
  %tmp16 = insertelement <16 x i8> %tmp15, i8 0, i32 10
  %tmp18 = insertelement <16 x i8> %tmp16, i8 0, i32 11
  %tmp19 = insertelement <16 x i8> %tmp18, i8 %tmp24, i32 12
  store <16 x i8> %tmp19, <16 x i8> addrspace(1)* %arg1, align 1
  ret void
}

attributes #0 = { nounwind }
