; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX900 %s
; RUN: llc -global-isel=0 -march=amdgcn -mcpu=gfx90a -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,PACKED,PACKED-SDAG %s
; RUN: llc -global-isel=1 -march=amdgcn -mcpu=gfx90a -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,PACKED,PACKED-GISEL %s
; RUN: llc -global-isel=0 -march=amdgcn -mcpu=gfx940 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,PACKED,PACKED-SDAG %s
; RUN: llc -global-isel=1 -march=amdgcn -mcpu=gfx940 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,PACKED,PACKED-GISEL %s

; GCN-LABEL: {{^}}fadd_v2_vv:
; GFX900-COUNT-2: v_add_f32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; PACKED:         v_pk_add_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}]
define amdgpu_kernel void @fadd_v2_vv(ptr addrspace(1) %a) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <2 x float>, ptr addrspace(1) %a, i32 %id
  %load = load <2 x float>, ptr addrspace(1) %gep, align 8
  %add = fadd <2 x float> %load, %load
  store <2 x float> %add, ptr addrspace(1) %gep, align 8
  ret void
}

; GCN-LABEL: {{^}}fadd_v2_vs:
; GFX900-COUNT-2: v_add_f32_e32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}
; PACKED:         v_pk_add_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], s[{{[0-9:]+}}]{{$}}
define amdgpu_kernel void @fadd_v2_vs(ptr addrspace(1) %a, <2 x float> %x) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <2 x float>, ptr addrspace(1) %a, i32 %id
  %load = load <2 x float>, ptr addrspace(1) %gep, align 8
  %add = fadd <2 x float> %load, %x
  store <2 x float> %add, ptr addrspace(1) %gep, align 8
  ret void
}

; GCN-LABEL: {{^}}fadd_v4_vs:
; GFX900-COUNT-4: v_add_f32_e32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}
; PACKED-COUNT-2: v_pk_add_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], s[{{[0-9:]+}}]{{$}}
define amdgpu_kernel void @fadd_v4_vs(ptr addrspace(1) %a, <4 x float> %x) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <4 x float>, ptr addrspace(1) %a, i32 %id
  %load = load <4 x float>, ptr addrspace(1) %gep, align 16
  %add = fadd <4 x float> %load, %x
  store <4 x float> %add, ptr addrspace(1) %gep, align 16
  ret void
}

; GCN-LABEL: {{^}}fadd_v32_vs:
; GFX900-COUNT-32: v_add_f32_e32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}
; PACKED-COUNT-16: v_pk_add_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], s[{{[0-9:]+}}]{{$}}
define amdgpu_kernel void @fadd_v32_vs(ptr addrspace(1) %a, <32 x float> %x) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <32 x float>, ptr addrspace(1) %a, i32 %id
  %load = load <32 x float>, ptr addrspace(1) %gep, align 128
  %add = fadd <32 x float> %load, %x
  store <32 x float> %add, ptr addrspace(1) %gep, align 128
  ret void
}

; FIXME: GISel does not use op_sel for splat constants.

; GCN-LABEL: {{^}}fadd_v2_v_imm:
; PACKED:         s_mov_b32 s[[K:[0-9]+]], 0x42c80000
; GFX900-COUNT-2: v_add_f32_e32 v{{[0-9]+}}, 0x42c80000, v{{[0-9]+}}
; PACKED-SDAG:    v_pk_add_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], s[[[K]]:{{[0-9:]+}}] op_sel_hi:[1,0]{{$}}
; PACKED-GISEL:   v_pk_add_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], s[[[K]]:{{[0-9:]+}}]{{$}}
define amdgpu_kernel void @fadd_v2_v_imm(ptr addrspace(1) %a) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <2 x float>, ptr addrspace(1) %a, i32 %id
  %load = load <2 x float>, ptr addrspace(1) %gep, align 8
  %add = fadd <2 x float> %load, <float 100.0, float 100.0>
  store <2 x float> %add, ptr addrspace(1) %gep, align 8
  ret void
}

; GCN-LABEL: {{^}}fadd_v2_v_v_splat:
; GFX900-COUNT-2: v_add_f32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v0
; PACKED-SDAG:    v_pk_add_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], v[0:1] op_sel_hi:[1,0]{{$}}
; PACKED-GISEL:   v_pk_add_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], v[0:1]{{$}}
define amdgpu_kernel void @fadd_v2_v_v_splat(ptr addrspace(1) %a) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <2 x float>, ptr addrspace(1) %a, i32 %id
  %load = load <2 x float>, ptr addrspace(1) %gep, align 8
  %fid = bitcast i32 %id to float
  %tmp1 = insertelement <2 x float> undef, float %fid, i64 0
  %k = insertelement <2 x float> %tmp1, float %fid, i64 1
  %add = fadd <2 x float> %load, %k
  store <2 x float> %add, ptr addrspace(1) %gep, align 8
  ret void
}

; GCN-LABEL: {{^}}fadd_v2_v_lit_splat:
; GFX900-COUNT-2: v_add_f32_e32 v{{[0-9]+}}, 1.0, v{{[0-9]+}}
; PACKED-SDAG:    v_pk_add_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], 1.0 op_sel_hi:[1,0]{{$}}
; PACKED-GISEL:   v_pk_add_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], s[{{[0-9:]+}}]{{$}}
define amdgpu_kernel void @fadd_v2_v_lit_splat(ptr addrspace(1) %a) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <2 x float>, ptr addrspace(1) %a, i32 %id
  %load = load <2 x float>, ptr addrspace(1) %gep, align 8
  %add = fadd <2 x float> %load, <float 1.0, float 1.0>
  store <2 x float> %add, ptr addrspace(1) %gep, align 8
  ret void
}

; GCN-LABEL: {{^}}fadd_v2_v_lit_hi0:
; GFX900-DAG: v_add_f32_e32 v{{[0-9]+}}, 0, v{{[0-9]+}}
; GFX900-DAG: v_add_f32_e32 v{{[0-9]+}}, 1.0, v{{[0-9]+}}
; PACKED-DAG: s_mov_b64 [[K:s\[[0-9:]+\]]], 0x3f800000
; PACKED:     v_pk_add_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], [[K]]
define amdgpu_kernel void @fadd_v2_v_lit_hi0(ptr addrspace(1) %a) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <2 x float>, ptr addrspace(1) %a, i32 %id
  %load = load <2 x float>, ptr addrspace(1) %gep, align 8
  %add = fadd <2 x float> %load, <float 1.0, float 0.0>
  store <2 x float> %add, ptr addrspace(1) %gep, align 8
  ret void
}

; GCN-LABEL: {{^}}fadd_v2_v_lit_lo0:
; GFX900-DAG: v_add_f32_e32 v{{[0-9]+}}, 0, v{{[0-9]+}}
; GFX900-DAG: v_add_f32_e32 v{{[0-9]+}}, 1.0, v{{[0-9]+}}
; PACKED-DAG: s_mov_b32 s[[LO:[0-9]+]], 0
; PACKED-DAG: s_mov_b32 s[[HI:[0-9]+]], 1.0
; PACKED:     v_pk_add_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], s[[[LO]]:[[HI]]]{{$}}
define amdgpu_kernel void @fadd_v2_v_lit_lo0(ptr addrspace(1) %a) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <2 x float>, ptr addrspace(1) %a, i32 %id
  %load = load <2 x float>, ptr addrspace(1) %gep, align 8
  %add = fadd <2 x float> %load, <float 0.0, float 1.0>
  store <2 x float> %add, ptr addrspace(1) %gep, align 8
  ret void
}

; GCN-LABEL: {{^}}fadd_v2_v_unfoldable_lit:
; GFX900-DAG: v_add_f32_e32 v{{[0-9]+}}, 1.0, v{{[0-9]+}}
; GFX900-DAG: v_add_f32_e32 v{{[0-9]+}}, 2.0, v{{[0-9]+}}
; PACKED-DAG: s_mov_b32 s{{[0-9]+}}, 1.0
; PACKED-DAG: s_mov_b32 s{{[0-9]+}}, 2.0
; PACKED:     v_pk_add_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], s[{{[0-9:]+}}]{{$}}
define amdgpu_kernel void @fadd_v2_v_unfoldable_lit(ptr addrspace(1) %a) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <2 x float>, ptr addrspace(1) %a, i32 %id
  %load = load <2 x float>, ptr addrspace(1) %gep, align 8
  %add = fadd <2 x float> %load, <float 1.0, float 2.0>
  store <2 x float> %add, ptr addrspace(1) %gep, align 8
  ret void
}

; FIXME: Fold fneg into v_pk_add_f32 with Global ISel.

; GCN-LABEL: {{^}}fadd_v2_v_fneg:
; GFX900-COUNT-2: v_subrev_f32_e32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}
; PACKED-SDAG:    v_pk_add_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], s[{{[0-9:]+}}] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]{{$}}
; PACKED-GISEL:   v_pk_add_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}]{{$}}
define amdgpu_kernel void @fadd_v2_v_fneg(ptr addrspace(1) %a, float %x) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <2 x float>, ptr addrspace(1) %a, i32 %id
  %load = load <2 x float>, ptr addrspace(1) %gep, align 8
  %fneg = fsub float -0.0, %x
  %tmp1 = insertelement <2 x float> undef, float %fneg, i64 0
  %k = insertelement <2 x float> %tmp1, float %fneg, i64 1
  %add = fadd <2 x float> %load, %k
  store <2 x float> %add, ptr addrspace(1) %gep, align 8
  ret void
}

; GCN-LABEL: {{^}}fadd_v2_v_fneg_lo:
; GFX900-DAG:   v_add_f32_e32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}
; GFX900-DAG:   v_subrev_f32_e32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}
; PACKED-SDAG:  v_pk_add_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], s[{{[0-9:]+}}] op_sel_hi:[1,0] neg_lo:[0,1]{{$}}
; PACKED-GISEL: v_pk_add_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}]{{$}}
define amdgpu_kernel void @fadd_v2_v_fneg_lo(ptr addrspace(1) %a, float %x) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <2 x float>, ptr addrspace(1) %a, i32 %id
  %load = load <2 x float>, ptr addrspace(1) %gep, align 8
  %fneg = fsub float -0.0, %x
  %tmp1 = insertelement <2 x float> undef, float %fneg, i64 0
  %k = insertelement <2 x float> %tmp1, float %x, i64 1
  %add = fadd <2 x float> %load, %k
  store <2 x float> %add, ptr addrspace(1) %gep, align 8
  ret void
}

; GCN-LABEL: {{^}}fadd_v2_v_fneg_hi:
; GFX900-DAG:   v_add_f32_e32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}
; GFX900-DAG:   v_subrev_f32_e32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}
; PACKED-SDAG:  v_pk_add_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], s[{{[0-9:]+}}] op_sel_hi:[1,0] neg_hi:[0,1]{{$}}
; PACKED-GISEL: v_pk_add_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}]{{$}}
define amdgpu_kernel void @fadd_v2_v_fneg_hi(ptr addrspace(1) %a, float %x) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <2 x float>, ptr addrspace(1) %a, i32 %id
  %load = load <2 x float>, ptr addrspace(1) %gep, align 8
  %fneg = fsub float -0.0, %x
  %tmp1 = insertelement <2 x float> undef, float %x, i64 0
  %k = insertelement <2 x float> %tmp1, float %fneg, i64 1
  %add = fadd <2 x float> %load, %k
  store <2 x float> %add, ptr addrspace(1) %gep, align 8
  ret void
}

; GCN-LABEL: {{^}}fadd_v2_v_fneg_lo2:
; GFX900-DAG:   v_add_f32_e32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}
; GFX900-DAG:   v_subrev_f32_e32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}
; PACKED-SDAG:  v_pk_add_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], s[{{[0-9:]+}}] neg_lo:[0,1]{{$}}
; PACKED-GISEL: v_pk_add_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}]{{$}}
define amdgpu_kernel void @fadd_v2_v_fneg_lo2(ptr addrspace(1) %a, float %x, float %y) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <2 x float>, ptr addrspace(1) %a, i32 %id
  %load = load <2 x float>, ptr addrspace(1) %gep, align 8
  %fneg = fsub float -0.0, %x
  %tmp1 = insertelement <2 x float> undef, float %fneg, i64 0
  %k = insertelement <2 x float> %tmp1, float %y, i64 1
  %add = fadd <2 x float> %load, %k
  store <2 x float> %add, ptr addrspace(1) %gep, align 8
  ret void
}

; GCN-LABEL: {{^}}fadd_v2_v_fneg_hi2:
; GFX900-DAG:   v_add_f32_e32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}
; GFX900-DAG:   v_subrev_f32_e32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}
; PACKED-SDAG:  v_pk_add_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], s[{{[0-9:]+}}] op_sel:[0,1] op_sel_hi:[1,0] neg_hi:[0,1]{{$}}
; PACKED-GISEL: v_pk_add_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}]{{$}}
define amdgpu_kernel void @fadd_v2_v_fneg_hi2(ptr addrspace(1) %a, float %x, float %y) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <2 x float>, ptr addrspace(1) %a, i32 %id
  %load = load <2 x float>, ptr addrspace(1) %gep, align 8
  %fneg = fsub float -0.0, %x
  %tmp1 = insertelement <2 x float> undef, float %y, i64 0
  %k = insertelement <2 x float> %tmp1, float %fneg, i64 1
  %add = fadd <2 x float> %load, %k
  store <2 x float> %add, ptr addrspace(1) %gep, align 8
  ret void
}

; GCN-LABEL: {{^}}fmul_v2_vv:
; GFX900-COUNT-2: v_mul_f32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; PACKED:         v_pk_mul_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}]
define amdgpu_kernel void @fmul_v2_vv(ptr addrspace(1) %a) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <2 x float>, ptr addrspace(1) %a, i32 %id
  %load = load <2 x float>, ptr addrspace(1) %gep, align 8
  %mul = fmul <2 x float> %load, %load
  store <2 x float> %mul, ptr addrspace(1) %gep, align 8
  ret void
}

; GCN-LABEL: {{^}}fmul_v2_vs:
; GFX900-COUNT-2: v_mul_f32_e32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}
; PACKED:         v_pk_mul_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], s[{{[0-9:]+}}]{{$}}
define amdgpu_kernel void @fmul_v2_vs(ptr addrspace(1) %a, <2 x float> %x) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <2 x float>, ptr addrspace(1) %a, i32 %id
  %load = load <2 x float>, ptr addrspace(1) %gep, align 8
  %mul = fmul <2 x float> %load, %x
  store <2 x float> %mul, ptr addrspace(1) %gep, align 8
  ret void
}

; GCN-LABEL: {{^}}fmul_v4_vs:
; GFX900-COUNT-4: v_mul_f32_e32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}
; PACKED-COUNT-2: v_pk_mul_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], s[{{[0-9:]+}}]{{$}}
define amdgpu_kernel void @fmul_v4_vs(ptr addrspace(1) %a, <4 x float> %x) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <4 x float>, ptr addrspace(1) %a, i32 %id
  %load = load <4 x float>, ptr addrspace(1) %gep, align 16
  %mul = fmul <4 x float> %load, %x
  store <4 x float> %mul, ptr addrspace(1) %gep, align 16
  ret void
}

; GCN-LABEL: {{^}}fmul_v32_vs:
; GFX900-COUNT-32: v_mul_f32_e32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}
; PACKED-COUNT-16: v_pk_mul_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], s[{{[0-9:]+}}]{{$}}
define amdgpu_kernel void @fmul_v32_vs(ptr addrspace(1) %a, <32 x float> %x) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <32 x float>, ptr addrspace(1) %a, i32 %id
  %load = load <32 x float>, ptr addrspace(1) %gep, align 128
  %mul = fmul <32 x float> %load, %x
  store <32 x float> %mul, ptr addrspace(1) %gep, align 128
  ret void
}

; GCN-LABEL: {{^}}fmul_v2_v_imm:
; PACKED:         s_mov_b32 s[[K:[0-9]+]], 0x42c80000
; GFX900-COUNT-2: v_mul_f32_e32 v{{[0-9]+}}, 0x42c80000, v{{[0-9]+}}
; PACKED-SDAG:    v_pk_mul_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], s[[[K]]:{{[0-9:]+}}] op_sel_hi:[1,0]{{$}}
; PACKED-GISEL:   v_pk_mul_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], s[[[K]]:{{[0-9:]+}}]{{$}}
define amdgpu_kernel void @fmul_v2_v_imm(ptr addrspace(1) %a) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <2 x float>, ptr addrspace(1) %a, i32 %id
  %load = load <2 x float>, ptr addrspace(1) %gep, align 8
  %mul = fmul <2 x float> %load, <float 100.0, float 100.0>
  store <2 x float> %mul, ptr addrspace(1) %gep, align 8
  ret void
}

; GCN-LABEL: {{^}}fmul_v2_v_v_splat:
; GFX900-COUNT-2: v_mul_f32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v0
; PACKED-SDAG:    v_pk_mul_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], v[0:1] op_sel_hi:[1,0]{{$}}
; PACKED-GISEL:   v_pk_mul_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], v[0:1]{{$}}
define amdgpu_kernel void @fmul_v2_v_v_splat(ptr addrspace(1) %a) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <2 x float>, ptr addrspace(1) %a, i32 %id
  %load = load <2 x float>, ptr addrspace(1) %gep, align 8
  %fid = bitcast i32 %id to float
  %tmp1 = insertelement <2 x float> undef, float %fid, i64 0
  %k = insertelement <2 x float> %tmp1, float %fid, i64 1
  %mul = fmul <2 x float> %load, %k
  store <2 x float> %mul, ptr addrspace(1) %gep, align 8
  ret void
}

; GCN-LABEL: {{^}}fmul_v2_v_lit_splat:
; GFX900-COUNT-2: v_mul_f32_e32 v{{[0-9]+}}, 4.0, v{{[0-9]+}}
; PACKED-SDAG:    v_pk_mul_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], 4.0 op_sel_hi:[1,0]{{$}}
; PACKED-GISEL:   v_pk_mul_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], s[{{[0-9:]+}}]{{$}}
define amdgpu_kernel void @fmul_v2_v_lit_splat(ptr addrspace(1) %a) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <2 x float>, ptr addrspace(1) %a, i32 %id
  %load = load <2 x float>, ptr addrspace(1) %gep, align 8
  %mul = fmul <2 x float> %load, <float 4.0, float 4.0>
  store <2 x float> %mul, ptr addrspace(1) %gep, align 8
  ret void
}

; GCN-LABEL: {{^}}fmul_v2_v_unfoldable_lit:
; GFX900-DAG: v_mul_f32_e32 v{{[0-9]+}}, 4.0, v{{[0-9]+}}
; GFX900-DAG: v_mul_f32_e32 v{{[0-9]+}}, 0x40400000, v{{[0-9]+}}
; PACKED-DAG: s_mov_b32 s{{[0-9]+}}, 4.0
; PACKED-DAG: s_mov_b32 s{{[0-9]+}}, 0x40400000
; PACKED:     v_pk_mul_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], s[{{[0-9:]+}}]{{$}}
define amdgpu_kernel void @fmul_v2_v_unfoldable_lit(ptr addrspace(1) %a) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <2 x float>, ptr addrspace(1) %a, i32 %id
  %load = load <2 x float>, ptr addrspace(1) %gep, align 8
  %mul = fmul <2 x float> %load, <float 4.0, float 3.0>
  store <2 x float> %mul, ptr addrspace(1) %gep, align 8
  ret void
}

; GCN-LABEL: {{^}}fmul_v2_v_fneg:
; GFX900-COUNT-2: v_mul_f32_e64 v{{[0-9]+}}, v{{[0-9]+}}, -s{{[0-9]+}}
; PACKED-SDAG:    v_pk_mul_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], s[{{[0-9:]+}}] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]{{$}}
; PACKED-GISEL:   v_pk_mul_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}]{{$}}
define amdgpu_kernel void @fmul_v2_v_fneg(ptr addrspace(1) %a, float %x) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <2 x float>, ptr addrspace(1) %a, i32 %id
  %load = load <2 x float>, ptr addrspace(1) %gep, align 8
  %fneg = fsub float -0.0, %x
  %tmp1 = insertelement <2 x float> undef, float %fneg, i64 0
  %k = insertelement <2 x float> %tmp1, float %fneg, i64 1
  %mul = fmul <2 x float> %load, %k
  store <2 x float> %mul, ptr addrspace(1) %gep, align 8
  ret void
}

; GCN-LABEL: {{^}}fma_v2_vv:
; GFX900-COUNT-2: v_fma_f32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; PACKED:         v_pk_fma_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}]
define amdgpu_kernel void @fma_v2_vv(ptr addrspace(1) %a) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <2 x float>, ptr addrspace(1) %a, i32 %id
  %load = load <2 x float>, ptr addrspace(1) %gep, align 8
  %fma = tail call <2 x float> @llvm.fma.v2f32(<2 x float> %load, <2 x float> %load, <2 x float> %load)
  store <2 x float> %fma, ptr addrspace(1) %gep, align 8
  ret void
}

; GCN-LABEL: {{^}}fma_v2_vs:
; GFX900-COUNT-2: v_fma_f32 v{{[0-9]+}}, v{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}
; PACKED:         v_pk_fma_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], s[{{[0-9:]+}}], s[{{[0-9:]+}}]{{$}}
define amdgpu_kernel void @fma_v2_vs(ptr addrspace(1) %a, <2 x float> %x) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <2 x float>, ptr addrspace(1) %a, i32 %id
  %load = load <2 x float>, ptr addrspace(1) %gep, align 8
  %fma = tail call <2 x float> @llvm.fma.v2f32(<2 x float> %load, <2 x float> %x, <2 x float> %x)
  store <2 x float> %fma, ptr addrspace(1) %gep, align 8
  ret void
}

; GCN-LABEL: {{^}}fma_v4_vs:
; GFX900-COUNT-4: v_fma_f32 v{{[0-9]+}}, v{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}
; PACKED-COUNT-2: v_pk_fma_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], s[{{[0-9:]+}}], s[{{[0-9:]+}}]{{$}}
define amdgpu_kernel void @fma_v4_vs(ptr addrspace(1) %a, <4 x float> %x) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <4 x float>, ptr addrspace(1) %a, i32 %id
  %load = load <4 x float>, ptr addrspace(1) %gep, align 16
  %fma = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %load, <4 x float> %x, <4 x float> %x)
  store <4 x float> %fma, ptr addrspace(1) %gep, align 16
  ret void
}

; GCN-LABEL: {{^}}fma_v32_vs:
; GFX900-COUNT-32: v_fma_f32 v{{[0-9]+}}, v{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}
; PACKED-COUNT-16: v_pk_fma_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], s[{{[0-9:]+}}], s[{{[0-9:]+}}]{{$}}
define amdgpu_kernel void @fma_v32_vs(ptr addrspace(1) %a, <32 x float> %x) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <32 x float>, ptr addrspace(1) %a, i32 %id
  %load = load <32 x float>, ptr addrspace(1) %gep, align 128
  %fma = tail call <32 x float> @llvm.fma.v32f32(<32 x float> %load, <32 x float> %x, <32 x float> %x)
  store <32 x float> %fma, ptr addrspace(1) %gep, align 128
  ret void
}

; GCN-LABEL: {{^}}fma_v2_v_imm:
; GCN-DAG:         s_mov_b32 s[[K1:[0-9]+]], 0x42c80000
; GFX900-DAG:      v_mov_b32_e32 v[[K2:[0-9]+]], 0x43480000
; PACKED-SDAG-DAG: v_mov_b32_e32 v[[K2:[0-9]+]], 0x43480000
; GFX900-COUNT-2:  v_fma_f32 v{{[0-9]+}}, v{{[0-9]+}}, s[[K1]], v[[K2]]
; PACKED-SDAG:     v_pk_fma_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], s[[[K1]]:{{[0-9:]+}}], v[[[K2]]:{{[0-9:]+}}] op_sel_hi:[1,0,0]{{$}}
; PACKED-GISEL:    v_pk_fma_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], s[[[K1]]:{{[0-9:]+}}], v[{{[0-9:]+}}]{{$}}
define amdgpu_kernel void @fma_v2_v_imm(ptr addrspace(1) %a) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <2 x float>, ptr addrspace(1) %a, i32 %id
  %load = load <2 x float>, ptr addrspace(1) %gep, align 8
  %fma = tail call <2 x float> @llvm.fma.v2f32(<2 x float> %load, <2 x float> <float 100.0, float 100.0>, <2 x float> <float 200.0, float 200.0>)
  store <2 x float> %fma, ptr addrspace(1) %gep, align 8
  ret void
}

; GCN-LABEL: {{^}}fma_v2_v_v_splat:
; GFX900-COUNT-2: v_fma_f32 v{{[0-9]+}}, v{{[0-9]+}}, v0, v0
; PACKED-SDAG:    v_pk_fma_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], v[0:1], v[0:1] op_sel_hi:[1,0,0]{{$}}
; PACKED-GISEL:   v_pk_fma_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], v[0:1], v[0:1]{{$}}
define amdgpu_kernel void @fma_v2_v_v_splat(ptr addrspace(1) %a) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <2 x float>, ptr addrspace(1) %a, i32 %id
  %load = load <2 x float>, ptr addrspace(1) %gep, align 8
  %fid = bitcast i32 %id to float
  %tmp1 = insertelement <2 x float> undef, float %fid, i64 0
  %k = insertelement <2 x float> %tmp1, float %fid, i64 1
  %fma = tail call <2 x float> @llvm.fma.v2f32(<2 x float> %load, <2 x float> %k, <2 x float> %k)
  store <2 x float> %fma, ptr addrspace(1) %gep, align 8
  ret void
}

; GCN-LABEL: {{^}}fma_v2_v_lit_splat:
; GFX900-COUNT-2: v_fma_f32 v{{[0-9]+}}, v{{[0-9]+}}, 4.0, 1.0
; PACKED-SDAG:    v_pk_fma_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], 4.0, 1.0 op_sel_hi:[1,0,0]{{$}}
; PACKED-GISEL:   v_pk_fma_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], s[{{[0-9:]+}}], v[{{[0-9:]+}}]{{$}}
define amdgpu_kernel void @fma_v2_v_lit_splat(ptr addrspace(1) %a) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <2 x float>, ptr addrspace(1) %a, i32 %id
  %load = load <2 x float>, ptr addrspace(1) %gep, align 8
  %fma = tail call <2 x float> @llvm.fma.v2f32(<2 x float> %load, <2 x float> <float 4.0, float 4.0>, <2 x float> <float 1.0, float 1.0>)
  store <2 x float> %fma, ptr addrspace(1) %gep, align 8
  ret void
}

; GCN-LABEL: {{^}}fma_v2_v_unfoldable_lit:
; GCN-DAG:         s_mov_b32 s{{[0-9]+}}, 0x40400000
; GFX900-DAG:      v_fma_f32 v{{[0-9]+}}, v{{[0-9]+}}, 4.0, 1.0
; GFX900-DAG:      v_fma_f32 v{{[0-9]+}}, v{{[0-9]+}}, s{{[0-9]+}}, 2.0
; PACKED-SDAG-DAG: s_mov_b32 s{{[0-9]+}}, 4.0
; PACKED-SDAG-DAG: v_mov_b32_e32 v{{[0-9]+}}, 1.0
; PACKED-SDAG-DAG: v_mov_b32_e32 v{{[0-9]+}}, 2.0
; PACKED:          v_pk_fma_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], s[{{[0-9:]+}}], v[{{[0-9:]+}}]{{$}}
define amdgpu_kernel void @fma_v2_v_unfoldable_lit(ptr addrspace(1) %a) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <2 x float>, ptr addrspace(1) %a, i32 %id
  %load = load <2 x float>, ptr addrspace(1) %gep, align 8
  %fma = tail call <2 x float> @llvm.fma.v2f32(<2 x float> %load, <2 x float> <float 4.0, float 3.0>, <2 x float> <float 1.0, float 2.0>)
  store <2 x float> %fma, ptr addrspace(1) %gep, align 8
  ret void
}

; GCN-LABEL: {{^}}fma_v2_v_fneg:
; GFX900-COUNT-2: v_fma_f32 v{{[0-9]+}}, v{{[0-9]+}}, -s{{[0-9]+}}, -s{{[0-9]+}}
; PACKED-SDAG:    v_pk_fma_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], s[{{[0-9:]+}}], s[{{[0-9:]+}}] op_sel_hi:[1,0,0] neg_lo:[0,1,1] neg_hi:[0,1,1]{{$}}
; PACKED-GISEL:   v_pk_fma_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}]{{$}}
define amdgpu_kernel void @fma_v2_v_fneg(ptr addrspace(1) %a, float %x) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <2 x float>, ptr addrspace(1) %a, i32 %id
  %load = load <2 x float>, ptr addrspace(1) %gep, align 8
  %fneg = fsub float -0.0, %x
  %tmp1 = insertelement <2 x float> undef, float %fneg, i64 0
  %k = insertelement <2 x float> %tmp1, float %fneg, i64 1
  %fma = tail call <2 x float> @llvm.fma.v2f32(<2 x float> %load, <2 x float> %k, <2 x float> %k)
  store <2 x float> %fma, ptr addrspace(1) %gep, align 8
  ret void
}

; GCN-LABEL: {{^}}add_vector_neg_bitcast_scalar_lo:
; GFX900-COUNT-2: v_sub_f32_e32
; PACKED-SDAG:    v_pk_add_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]{{$}}
; PACKED-GISEL:   v_pk_add_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}]{{$}}
define amdgpu_kernel void @add_vector_neg_bitcast_scalar_lo(ptr addrspace(1) %out, ptr addrspace(3) %lds, ptr addrspace(3) %arg2) {
bb:
  %vec0 = load volatile <2 x float>, ptr addrspace(3) %lds, align 4
  %scalar0 = load volatile float, ptr addrspace(3) %arg2, align 4
  %neg.scalar0 = fsub float -0.0, %scalar0

  %neg.scalar0.vec = insertelement <2 x float> undef, float %neg.scalar0, i32 0
  %neg.scalar0.broadcast = shufflevector <2 x float> %neg.scalar0.vec, <2 x float> undef, <2 x i32> zeroinitializer

  %result = fadd <2 x float> %vec0, %neg.scalar0.broadcast
  store <2 x float> %result, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}fma_vector_vector_neg_scalar_lo_scalar_hi:
; GFX900-COUNT-2: v_fma_f32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, -v{{[0-9]+}}
; PACKED-SDAG:    v_pk_fma_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}] neg_lo:[0,0,1] neg_hi:[0,0,1]{{$}}
; PACKED-GISEL:   v_pk_fma_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}]{{$}}
define amdgpu_kernel void @fma_vector_vector_neg_scalar_lo_scalar_hi(ptr addrspace(1) %out, ptr addrspace(3) %lds, ptr addrspace(3) %arg2) {
bb:
  %lds.gep1 = getelementptr inbounds <2 x float>, ptr addrspace(3) %lds, i32 1
  %arg2.gep = getelementptr inbounds float, ptr addrspace(3) %arg2, i32 2

  %vec0 = load volatile <2 x float>, ptr addrspace(3) %lds, align 4
  %vec1 = load volatile <2 x float>, ptr addrspace(3) %lds.gep1, align 4

  %scalar0 = load volatile float, ptr addrspace(3) %arg2, align 4
  %scalar1 = load volatile float, ptr addrspace(3) %arg2.gep, align 4

  %vec.ins0 = insertelement <2 x float> undef, float %scalar0, i32 0
  %vec2 = insertelement <2 x float> %vec.ins0, float %scalar1, i32 1
  %neg.vec2 = fsub <2 x float> <float -0.0, float -0.0>, %vec2

  %result = tail call <2 x float> @llvm.fma.v2f32(<2 x float> %vec0, <2 x float> %vec1, <2 x float> %neg.vec2)
  store <2 x float> %result, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}shuffle_add_f32:
; GFX900-COUNT-2: v_add_f32_e32
; PACKED-SDAG:    v_pk_add_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}] op_sel:[0,1] op_sel_hi:[1,0]{{$}}
; PACKED-GISEL:   v_pk_add_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}]{{$}}
define amdgpu_kernel void @shuffle_add_f32(ptr addrspace(1) %out, ptr addrspace(3) %lds) #0 {
bb:
  %vec0 = load volatile <2 x float>, ptr addrspace(3) %lds, align 8
  %lds.gep1 = getelementptr inbounds <2 x float>, ptr addrspace(3) %lds, i32 1
  %vec1 = load volatile <2 x float>, ptr addrspace(3) %lds.gep1, align 8
  %vec1.swap = shufflevector <2 x float> %vec1, <2 x float> undef, <2 x i32> <i32 1, i32 0>
  %result = fadd <2 x float> %vec0, %vec1.swap
  store <2 x float> %result, ptr addrspace(1) %out, align 8
  ret void
}

; GCN-LABEL: {{^}}shuffle_neg_add_f32:
; GFX900-COUNT-2: v_sub_f32_e32
; PACKED-SDAG:    v_pk_add_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}] op_sel:[0,1] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]{{$}}
; PACKED-GISEL:   v_pk_add_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}]{{$}}
define amdgpu_kernel void @shuffle_neg_add_f32(ptr addrspace(1) %out, ptr addrspace(3) %lds) #0 {
bb:
  %vec0 = load volatile <2 x float>, ptr addrspace(3) %lds, align 8
  %lds.gep1 = getelementptr inbounds <2 x float>, ptr addrspace(3) %lds, i32 1
  %f32 = load volatile float, ptr addrspace(3) undef, align 8
  %vec1 = load volatile <2 x float>, ptr addrspace(3) %lds.gep1, align 8
  %vec1.neg = fsub <2 x float> <float -0.0, float -0.0>, %vec1
  %vec1.neg.swap = shufflevector <2 x float> %vec1.neg, <2 x float> undef, <2 x i32> <i32 1, i32 0>
  %result = fadd <2 x float> %vec0, %vec1.neg.swap
  store <2 x float> %result, ptr addrspace(1) %out, align 8
  ret void
}

; GCN-LABEL: {{^}}fadd_fadd_fsub:
; GFX900:       v_add_f32_e64 v{{[0-9]+}}, s{{[0-9]+}}, 0
; GFX900:       v_add_f32_e32 v{{[0-9]+}}, 0, v{{[0-9]+}}
; PACKED-SDAG:  v_pk_add_f32 v[{{[0-9:]+}}], s[{{[0-9:]+}}], 0 op_sel_hi:[1,0]{{$}}
; PACKED-SDAG:  v_pk_add_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], 0 op_sel_hi:[1,0]{{$}}
; PACKED-GISEL: v_pk_add_f32 v[{{[0-9:]+}}], s[{{[0-9:]+}}], v[{{[0-9:]+}}]{{$}}
; PACKED-GISEL: v_pk_add_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], s[{{[0-9:]+}}]{{$}}
define amdgpu_kernel void @fadd_fadd_fsub(<2 x float> %arg) {
bb:
  %i12 = fadd <2 x float> zeroinitializer, %arg
  %shift8 = shufflevector <2 x float> %i12, <2 x float> undef, <2 x i32> <i32 1, i32 undef>
  %i13 = fadd <2 x float> zeroinitializer, %shift8
  %i14 = shufflevector <2 x float> %arg, <2 x float> %i13, <2 x i32> <i32 0, i32 2>
  %i15 = fsub <2 x float> %i14, zeroinitializer
  store <2 x float> %i15, ptr undef
  ret void
}

; GCN-LABEL: {{^}}fadd_shuffle_v4:
; GFX900-COUNT-4:       v_add_f32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; PACKED-SDAG-COUNT-2:  v_pk_add_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}] op_sel_hi:[1,0]{{$}}
; PACKED-GISEL-COUNT-2: v_pk_add_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}]{{$}}
define amdgpu_kernel void @fadd_shuffle_v4(ptr addrspace(1) %arg) {
bb:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <4 x float>, ptr addrspace(1) %arg, i32 %tid
  %in.1 = load <4 x float>, ptr addrspace(1) %gep
  %shuf = shufflevector <4 x float> %in.1, <4 x float> undef, <4 x i32> zeroinitializer
  %add.1 = fadd <4 x float> %in.1, %shuf
  store <4 x float> %add.1, ptr addrspace(1) %gep
  ret void
}

; GCN-LABEL: {{^}}fneg_v2f32_vec:
; GFX900-COUNT-2:       v_xor_b32_e32 v{{[0-9]+}}, 0x80000000, v{{[0-9]+}}
; PACKED-SDAG:          v_pk_add_f32 v[{{[0-9:]+}}], v[{{[0-9:]+}}], 0 neg_lo:[1,1] neg_hi:[1,1]{{$}}
; PACKED-GISEL-COUNT-2: v_xor_b32_e32 v{{[0-9]+}}, 0x80000000, v{{[0-9]+}}
; PACKED-GISEL:         v_pk_mul_f32 v[{{[0-9:]+}}], 1.0, v[{{[0-9:]+}}] op_sel_hi:[0,1]{{$}}
define amdgpu_kernel void @fneg_v2f32_vec(ptr addrspace(1) %a) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <2 x float>, ptr addrspace(1) %a, i32 %id
  %load = load <2 x float>, ptr addrspace(1) %gep, align 8
  %fneg = fsub <2 x float> <float -0.0, float -0.0>, %load
  store <2 x float> %fneg, ptr addrspace(1) %gep, align 8
  ret void
}

; GCN-LABEL: {{^}}fneg_v2f32_scalar:
; GCN-COUNT-2: s_xor_b32 s{{[0-9]+}}, s{{[0-9]+}}, 0x80000000
define amdgpu_kernel void @fneg_v2f32_scalar(ptr addrspace(1) %a, <2 x float> %x) {
  %fneg = fsub <2 x float> <float -0.0, float -0.0>, %x
  store <2 x float> %fneg, ptr addrspace(1) %a, align 8
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
declare <2 x float> @llvm.fma.v2f32(<2 x float>, <2 x float>, <2 x float>)
declare <4 x float> @llvm.fma.v4f32(<4 x float>, <4 x float>, <4 x float>)
declare <32 x float> @llvm.fma.v32f32(<32 x float>, <32 x float>, <32 x float>)
