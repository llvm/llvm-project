; RUN: llc -verify-machineinstrs -march=amdgcn < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -verify-machineinstrs -march=amdgcn -mcpu=tonga < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

; Test expansion of scalar selects on vectors.
; Evergreen not enabled since it seems to be having problems with doubles.


; FUNC-LABEL: {{^}}select_v4i8:
; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
define void @select_v4i8(<4 x i8> addrspace(1)* %out, <4 x i8> %a, <4 x i8> %b, i8 %c) nounwind {
  %cmp = icmp eq i8 %c, 0
  %select = select i1 %cmp, <4 x i8> %a, <4 x i8> %b
  store <4 x i8> %select, <4 x i8> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}select_v4i16:
; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
define void @select_v4i16(<4 x i16> addrspace(1)* %out, <4 x i16> %a, <4 x i16> %b, i32 %c) nounwind {
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <4 x i16> %a, <4 x i16> %b
  store <4 x i16> %select, <4 x i16> addrspace(1)* %out, align 4
  ret void
}

; FIXME: Expansion with bitwise operations may be better if doing a
; vector select with SGPR inputs.

; FUNC-LABEL: {{^}}s_select_v2i32:
; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
; SI: buffer_store_dwordx2
define void @s_select_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> %a, <2 x i32> %b, i32 %c) nounwind {
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <2 x i32> %a, <2 x i32> %b
  store <2 x i32> %select, <2 x i32> addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}s_select_v4i32:
; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
; SI: buffer_store_dwordx4
define void @s_select_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> %a, <4 x i32> %b, i32 %c) nounwind {
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <4 x i32> %a, <4 x i32> %b
  store <4 x i32> %select, <4 x i32> addrspace(1)* %out, align 16
  ret void
}

; FUNC-LABEL: {{^}}v_select_v4i32:
; SI: buffer_load_dwordx4
; SI: v_cmp_gt_u32_e64 vcc, 32, s{{[0-9]+}}
; SI: v_cndmask_b32_e32 v{{[0-9]+}}, 0, v{{[0-9]+}}
; SI: v_cndmask_b32_e32 v{{[0-9]+}}, 0, v{{[0-9]+}}
; SI: v_cndmask_b32_e32 v{{[0-9]+}}, 0, v{{[0-9]+}}
; SI: v_cndmask_b32_e32 v{{[0-9]+}}, 0, v{{[0-9]+}}
; SI: buffer_store_dwordx4
define void @v_select_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %in, i32 %cond) #0 {
bb:
  %tmp2 = icmp ult i32 %cond, 32
  %val = load <4 x i32>, <4 x i32> addrspace(1)* %in
  %tmp3 = select i1 %tmp2, <4 x i32> %val, <4 x i32> zeroinitializer
  store <4 x i32> %tmp3, <4 x i32> addrspace(1)* %out, align 16
  ret void
}

; FUNC-LABEL: {{^}}select_v8i32:
; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
define void @select_v8i32(<8 x i32> addrspace(1)* %out, <8 x i32> %a, <8 x i32> %b, i32 %c) nounwind {
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <8 x i32> %a, <8 x i32> %b
  store <8 x i32> %select, <8 x i32> addrspace(1)* %out, align 16
  ret void
}

; FUNC-LABEL: {{^}}s_select_v2f32:
; SI-DAG: s_load_dwordx2 s{{\[}}[[ALO:[0-9]+]]:[[AHI:[0-9]+]]{{\]}}, s{{\[[0-9]+:[0-9]+\]}}, {{0xb|0x2c}}
; SI-DAG: s_load_dwordx2 s{{\[}}[[BLO:[0-9]+]]:[[BHI:[0-9]+]]{{\]}}, s{{\[[0-9]+:[0-9]+\]}}, {{0xd|0x34}}

; SI-DAG: v_mov_b32_e32 v{{[0-9]+}}, s[[ALO]]
; SI-DAG: v_mov_b32_e32 v{{[0-9]+}}, s[[AHI]]
; SI-DAG: v_mov_b32_e32 v{{[0-9]+}}, s[[BLO]]
; SI-DAG: v_mov_b32_e32 v{{[0-9]+}}, s[[BHI]]
; SI-DAG: v_cmp_eq_i32_e64 vcc, 0, s{{[0-9]+}}

; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
; SI: buffer_store_dwordx2
define void @s_select_v2f32(<2 x float> addrspace(1)* %out, <2 x float> %a, <2 x float> %b, i32 %c) nounwind {
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <2 x float> %a, <2 x float> %b
  store <2 x float> %select, <2 x float> addrspace(1)* %out, align 16
  ret void
}

; FUNC-LABEL: {{^}}s_select_v4f32:
; SI: s_load_dwordx4
; SI: s_load_dwordx4
; SI: v_cmp_eq_i32_e64 vcc, 0, s{{[0-9]+}}

; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32

; SI: buffer_store_dwordx4
define void @s_select_v4f32(<4 x float> addrspace(1)* %out, <4 x float> %a, <4 x float> %b, i32 %c) nounwind {
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <4 x float> %a, <4 x float> %b
  store <4 x float> %select, <4 x float> addrspace(1)* %out, align 16
  ret void
}

; FUNC-LABEL: {{^}}v_select_v4f32:
; SI: buffer_load_dwordx4
; SI: v_cmp_gt_u32_e64 vcc, 32, s{{[0-9]+}}
; SI: v_cndmask_b32_e32 v{{[0-9]+}}, 0, v{{[0-9]+}}
; SI: v_cndmask_b32_e32 v{{[0-9]+}}, 0, v{{[0-9]+}}
; SI: v_cndmask_b32_e32 v{{[0-9]+}}, 0, v{{[0-9]+}}
; SI: v_cndmask_b32_e32 v{{[0-9]+}}, 0, v{{[0-9]+}}
; SI: buffer_store_dwordx4
define void @v_select_v4f32(<4 x float> addrspace(1)* %out, <4 x float> addrspace(1)* %in, i32 %cond) #0 {
bb:
  %tmp2 = icmp ult i32 %cond, 32
  %val = load <4 x float>, <4 x float> addrspace(1)* %in
  %tmp3 = select i1 %tmp2, <4 x float> %val, <4 x float> zeroinitializer
  store <4 x float> %tmp3, <4 x float> addrspace(1)* %out, align 16
  ret void
}

; FUNC-LABEL: {{^}}select_v8f32:
; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
define void @select_v8f32(<8 x float> addrspace(1)* %out, <8 x float> %a, <8 x float> %b, i32 %c) nounwind {
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <8 x float> %a, <8 x float> %b
  store <8 x float> %select, <8 x float> addrspace(1)* %out, align 16
  ret void
}

; FUNC-LABEL: {{^}}select_v2f64:
; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
define void @select_v2f64(<2 x double> addrspace(1)* %out, <2 x double> %a, <2 x double> %b, i32 %c) nounwind {
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <2 x double> %a, <2 x double> %b
  store <2 x double> %select, <2 x double> addrspace(1)* %out, align 16
  ret void
}

; FUNC-LABEL: {{^}}select_v4f64:
; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
define void @select_v4f64(<4 x double> addrspace(1)* %out, <4 x double> %a, <4 x double> %b, i32 %c) nounwind {
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <4 x double> %a, <4 x double> %b
  store <4 x double> %select, <4 x double> addrspace(1)* %out, align 16
  ret void
}

; FUNC-LABEL: {{^}}select_v8f64:
; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
define void @select_v8f64(<8 x double> addrspace(1)* %out, <8 x double> %a, <8 x double> %b, i32 %c) nounwind {
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <8 x double> %a, <8 x double> %b
  store <8 x double> %select, <8 x double> addrspace(1)* %out, align 16
  ret void
}

; Function Attrs: nounwind readnone
declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
