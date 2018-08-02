; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,SI %s
; RUN: llc -march=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX89 %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX89,GFX9 %s

; GCN-LABEL: {{^}}v_clamp_add_src_f32:
; GCN: {{buffer|flat|global}}_load_dword [[A:v[0-9]+]]
; GCN-NOT: [[A]]
; GCN: v_add_f32_e64 v{{[0-9]+}}, [[A]], 1.0 clamp{{$}}
define amdgpu_kernel void @v_clamp_add_src_f32(float addrspace(1)* %out, float addrspace(1)* %aptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load float, float addrspace(1)* %gep0
  %add = fadd float %a, 1.0
  %max = call float @llvm.maxnum.f32(float %add, float 0.0)
  %clamp = call float @llvm.minnum.f32(float %max, float 1.0)
  store float %clamp, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_multi_use_src_f32:
; GCN: {{buffer|flat|global}}_load_dword [[A:v[0-9]+]]
; GCN: v_add_f32_e32 [[ADD:v[0-9]+]], 1.0, [[A]]{{$}}
; GCN: v_max_f32_e64 v{{[0-9]+}}, [[ADD]], [[ADD]] clamp{{$}}
define amdgpu_kernel void @v_clamp_multi_use_src_f32(float addrspace(1)* %out, float addrspace(1)* %aptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load float, float addrspace(1)* %gep0
  %add = fadd float %a, 1.0
  %max = call float @llvm.maxnum.f32(float %add, float 0.0)
  %clamp = call float @llvm.minnum.f32(float %max, float 1.0)
  store float %clamp, float addrspace(1)* %out.gep
  store volatile float %add, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}v_clamp_dbg_use_src_f32:
; GCN: {{buffer|flat|global}}_load_dword [[A:v[0-9]+]]
; GCN-NOT: [[A]]
; GCN: v_add_f32_e64 v{{[0-9]+}}, [[A]], 1.0 clamp{{$}}
define amdgpu_kernel void @v_clamp_dbg_use_src_f32(float addrspace(1)* %out, float addrspace(1)* %aptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load float, float addrspace(1)* %gep0
  %add = fadd float %a, 1.0
  call void @llvm.dbg.value(metadata float %add, i64 0, metadata !4, metadata !9), !dbg !10
  %max = call float @llvm.maxnum.f32(float %add, float 0.0)
  %clamp = call float @llvm.minnum.f32(float %max, float 1.0)
  store float %clamp, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_add_neg_src_f32:
; GCN: {{buffer|flat|global}}_load_dword [[A:v[0-9]+]]
; GCN: v_floor_f32_e32 [[FLOOR:v[0-9]+]], [[A]]
; GCN: v_max_f32_e64 v{{[0-9]+}}, -[[FLOOR]], -[[FLOOR]] clamp{{$}}
define amdgpu_kernel void @v_clamp_add_neg_src_f32(float addrspace(1)* %out, float addrspace(1)* %aptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load float, float addrspace(1)* %gep0
  %floor = call float @llvm.floor.f32(float %a)
  %neg.floor = fsub float -0.0, %floor
  %max = call float @llvm.maxnum.f32(float %neg.floor, float 0.0)
  %clamp = call float @llvm.minnum.f32(float %max, float 1.0)
  store float %clamp, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_non_clamp_max_f32:
; GCN: {{buffer|flat|global}}_load_dword [[A:v[0-9]+]]
; GCN: v_add_f32_e32 [[ADD:v[0-9]+]], 1.0, [[A]]{{$}}
; GCN: v_max_f32_e32 v{{[0-9]+}}, 0, [[ADD]]{{$}}
define amdgpu_kernel void @v_non_clamp_max_f32(float addrspace(1)* %out, float addrspace(1)* %aptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load float, float addrspace(1)* %gep0
  %add = fadd float %a, 1.0
  %max = call float @llvm.maxnum.f32(float %add, float 0.0)
  store float %max, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_add_src_f32_denormals:
; GCN: {{buffer|flat|global}}_load_dword [[A:v[0-9]+]]
; GCN: v_add_f32_e64 [[ADD:v[0-9]+]], [[A]], 1.0 clamp{{$}}
define amdgpu_kernel void @v_clamp_add_src_f32_denormals(float addrspace(1)* %out, float addrspace(1)* %aptr) #2 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load float, float addrspace(1)* %gep0
  %add = fadd float %a, 1.0
  %max = call float @llvm.maxnum.f32(float %add, float 0.0)
  %clamp = call float @llvm.minnum.f32(float %max, float 1.0)
  store float %clamp, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_add_src_f16_denorm:
; GCN: {{buffer|flat|global}}_load_ushort [[A:v[0-9]+]]
; GFX89: v_add_f16_e64 [[ADD:v[0-9]+]], [[A]], 1.0 clamp{{$}}

; SI: v_cvt_f32_f16_e32 [[CVT:v[0-9]+]], [[A]]
; SI: v_add_f32_e64 [[ADD:v[0-9]+]], [[CVT]], 1.0 clamp{{$}}
; SI: v_cvt_f16_f32_e32 v{{[0-9]+}}, [[ADD]]
define amdgpu_kernel void @v_clamp_add_src_f16_denorm(half addrspace(1)* %out, half addrspace(1)* %aptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr half, half addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr half, half addrspace(1)* %out, i32 %tid
  %a = load half, half addrspace(1)* %gep0
  %add = fadd half %a, 1.0
  %max = call half @llvm.maxnum.f16(half %add, half 0.0)
  %clamp = call half @llvm.minnum.f16(half %max, half 1.0)
  store half %clamp, half addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_add_src_f16_no_denormals:
; GCN: {{buffer|flat|global}}_load_ushort [[A:v[0-9]+]]
; GFX89-NOT: [[A]]
; GFX89: v_add_f16_e64 v{{[0-9]+}}, [[A]], 1.0 clamp{{$}}

; SI: v_cvt_f32_f16_e32 [[CVT:v[0-9]+]], [[A]]
; SI: v_add_f32_e64 [[ADD:v[0-9]+]], [[CVT]], 1.0 clamp{{$}}
; SI: v_cvt_f16_f32_e32 v{{[0-9]+}}, [[ADD]]
define amdgpu_kernel void @v_clamp_add_src_f16_no_denormals(half addrspace(1)* %out, half addrspace(1)* %aptr) #3 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr half, half addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr half, half addrspace(1)* %out, i32 %tid
  %a = load half, half addrspace(1)* %gep0
  %add = fadd half %a, 1.0
  %max = call half @llvm.maxnum.f16(half %add, half 0.0)
  %clamp = call half @llvm.minnum.f16(half %max, half 1.0)
  store half %clamp, half addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_add_src_v2f32:
; GCN: {{buffer|flat|global}}_load_dwordx2 v{{\[}}[[A:[0-9]+]]:[[B:[0-9]+]]{{\]}}
; GCN-DAG: v_add_f32_e64 v{{[0-9]+}}, v[[A]], 1.0 clamp{{$}}
; GCN-DAG: v_add_f32_e64 v{{[0-9]+}}, v[[B]], 1.0 clamp{{$}}
define amdgpu_kernel void @v_clamp_add_src_v2f32(<2 x float> addrspace(1)* %out, <2 x float> addrspace(1)* %aptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr <2 x float>, <2 x float> addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr <2 x float>, <2 x float> addrspace(1)* %out, i32 %tid
  %a = load <2 x float>, <2 x float> addrspace(1)* %gep0
  %add = fadd <2 x float> %a, <float 1.0, float 1.0>
  %max = call <2 x float> @llvm.maxnum.v2f32(<2 x float> %add, <2 x float> zeroinitializer)
  %clamp = call <2 x float> @llvm.minnum.v2f32(<2 x float> %max, <2 x float> <float 1.0, float 1.0>)
  store <2 x float> %clamp, <2 x float> addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_add_src_f64:
; GCN: {{buffer|flat|global}}_load_dwordx2 [[A:v\[[0-9]+:[0-9]+\]]]
; GCN: v_add_f64 v{{\[[0-9]+:[0-9]+\]}}, [[A]], 1.0 clamp{{$}}
define amdgpu_kernel void @v_clamp_add_src_f64(double addrspace(1)* %out, double addrspace(1)* %aptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr double, double addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr double, double addrspace(1)* %out, i32 %tid
  %a = load double, double addrspace(1)* %gep0
  %add = fadd double %a, 1.0
  %max = call double @llvm.maxnum.f64(double %add, double 0.0)
  %clamp = call double @llvm.minnum.f64(double %max, double 1.0)
  store double %clamp, double addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_mac_to_mad:
; GCN: v_mad_f32 v{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]}} clamp{{$}}
define amdgpu_kernel void @v_clamp_mac_to_mad(float addrspace(1)* %out, float addrspace(1)* %aptr, float %a) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %b = load float, float addrspace(1)* %gep0

  %mul = fmul float %a, %a
  %add = fadd float %mul, %b
  %max = call float @llvm.maxnum.f32(float %add, float 0.0)
  %clamp = call float @llvm.minnum.f32(float %max, float 1.0)
  %res = fadd float %clamp, %b
  store float %res, float addrspace(1)* %out.gep
  ret void
}


; GCN-LABEL: {{^}}v_clamp_add_src_v2f16_denorm:
; GCN:  {{buffer|flat|global}}_load_dword [[A:v[0-9]+]]
; GFX9: v_pk_add_f16 [[ADD:v[0-9]+]], [[A]], 1.0 op_sel_hi:[1,0] clamp{{$}}
define amdgpu_kernel void @v_clamp_add_src_v2f16_denorm(<2 x half> addrspace(1)* %out, <2 x half> addrspace(1)* %aptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr <2 x half>, <2 x half> addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr <2 x half>, <2 x half> addrspace(1)* %out, i32 %tid
  %a = load <2 x half>, <2 x half> addrspace(1)* %gep0
  %add = fadd <2 x half> %a, <half 1.0, half 1.0>
  %max = call <2 x half> @llvm.maxnum.v2f16(<2 x half> %add, <2 x half> zeroinitializer)
  %clamp = call <2 x half> @llvm.minnum.v2f16(<2 x half> %max, <2 x half> <half 1.0, half 1.0>)
  store <2 x half> %clamp, <2 x half> addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_add_src_v2f16_no_denormals:
; GCN:  {{buffer|flat|global}}_load_dword [[A:v[0-9]+]]
; GFX9: v_pk_add_f16 [[ADD:v[0-9]+]], [[A]], 1.0 op_sel_hi:[1,0] clamp{{$}}
define amdgpu_kernel void @v_clamp_add_src_v2f16_no_denormals(<2 x half> addrspace(1)* %out, <2 x half> addrspace(1)* %aptr) #3 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr <2 x half>, <2 x half> addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr <2 x half>, <2 x half> addrspace(1)* %out, i32 %tid
  %a = load <2 x half>, <2 x half> addrspace(1)* %gep0
  %add = fadd <2 x half> %a, <half 1.0, half 1.0>
  %max = call <2 x half> @llvm.maxnum.v2f16(<2 x half> %add, <2 x half> zeroinitializer)
  %clamp = call <2 x half> @llvm.minnum.v2f16(<2 x half> %max, <2 x half> <half 1.0, half 1.0>)
  store <2 x half> %clamp, <2 x half> addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_add_src_v2f16_denorm_neg:
; GCN:  {{buffer|flat|global}}_load_dword [[A:v[0-9]+]]
; GFX9: v_pk_add_f16 [[ADD:v[0-9]+]], [[A]], 1.0 op_sel_hi:[1,0]{{$}}
; GFX9: v_pk_max_f16 [[MAX:v[0-9]+]], [[ADD]], [[ADD]] neg_lo:[1,1] neg_hi:[1,1] clamp{{$}}
define amdgpu_kernel void @v_clamp_add_src_v2f16_denorm_neg(<2 x half> addrspace(1)* %out, <2 x half> addrspace(1)* %aptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr <2 x half>, <2 x half> addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr <2 x half>, <2 x half> addrspace(1)* %out, i32 %tid
  %a = load <2 x half>, <2 x half> addrspace(1)* %gep0
  %add = fadd <2 x half> %a, <half 1.0, half 1.0>
  %neg.add = fsub <2 x half> <half -0.0, half -0.0>, %add
  %max = call <2 x half> @llvm.maxnum.v2f16(<2 x half> %neg.add, <2 x half> zeroinitializer)
  %clamp = call <2 x half> @llvm.minnum.v2f16(<2 x half> %max, <2 x half> <half 1.0, half 1.0>)
  store <2 x half> %clamp, <2 x half> addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_add_src_v2f16_denorm_neg_lo:
; GCN:  {{buffer|flat|global}}_load_dword [[A:v[0-9]+]]
; GFX9: v_pk_add_f16 [[ADD:v[0-9]+]], [[A]], 1.0 op_sel_hi:[1,0]{{$}}
; GFX9: v_pk_max_f16 [[MAX:v[0-9]+]], [[ADD]], [[ADD]] neg_lo:[1,1] clamp{{$}}
define amdgpu_kernel void @v_clamp_add_src_v2f16_denorm_neg_lo(<2 x half> addrspace(1)* %out, <2 x half> addrspace(1)* %aptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr <2 x half>, <2 x half> addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr <2 x half>, <2 x half> addrspace(1)* %out, i32 %tid
  %a = load <2 x half>, <2 x half> addrspace(1)* %gep0
  %add = fadd <2 x half> %a, <half 1.0, half 1.0>
  %lo = extractelement <2 x half> %add, i32 0
  %neg.lo = fsub half -0.0, %lo
  %neg.lo.add = insertelement <2 x half> %add, half %neg.lo, i32 0
  %max = call <2 x half> @llvm.maxnum.v2f16(<2 x half> %neg.lo.add, <2 x half> zeroinitializer)
  %clamp = call <2 x half> @llvm.minnum.v2f16(<2 x half> %max, <2 x half> <half 1.0, half 1.0>)
  store <2 x half> %clamp, <2 x half> addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_add_src_v2f16_denorm_neg_hi:
; GCN:  {{buffer|flat|global}}_load_dword [[A:v[0-9]+]]
; GFX9: v_pk_add_f16 [[ADD:v[0-9]+]], [[A]], 1.0 op_sel_hi:[1,0]{{$}}
; GFX9: v_pk_max_f16 [[MAX:v[0-9]+]], [[ADD]], [[ADD]] neg_hi:[1,1] clamp{{$}}
define amdgpu_kernel void @v_clamp_add_src_v2f16_denorm_neg_hi(<2 x half> addrspace(1)* %out, <2 x half> addrspace(1)* %aptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr <2 x half>, <2 x half> addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr <2 x half>, <2 x half> addrspace(1)* %out, i32 %tid
  %a = load <2 x half>, <2 x half> addrspace(1)* %gep0
  %add = fadd <2 x half> %a, <half 1.0, half 1.0>
  %hi = extractelement <2 x half> %add, i32 1
  %neg.hi = fsub half -0.0, %hi
  %neg.hi.add = insertelement <2 x half> %add, half %neg.hi, i32 1
  %max = call <2 x half> @llvm.maxnum.v2f16(<2 x half> %neg.hi.add, <2 x half> zeroinitializer)
  %clamp = call <2 x half> @llvm.minnum.v2f16(<2 x half> %max, <2 x half> <half 1.0, half 1.0>)
  store <2 x half> %clamp, <2 x half> addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_add_src_v2f16_denorm_shuf:
; GCN:  {{buffer|flat|global}}_load_dword [[A:v[0-9]+]]
; GFX9: v_pk_add_f16 [[ADD:v[0-9]+]], [[A]], 1.0 op_sel_hi:[1,0]{{$}}
; GFX9: v_pk_max_f16 [[MAX:v[0-9]+]], [[ADD]], [[ADD]] op_sel:[1,1] op_sel_hi:[0,0] clamp{{$}}
define amdgpu_kernel void @v_clamp_add_src_v2f16_denorm_shuf(<2 x half> addrspace(1)* %out, <2 x half> addrspace(1)* %aptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr <2 x half>, <2 x half> addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr <2 x half>, <2 x half> addrspace(1)* %out, i32 %tid
  %a = load <2 x half>, <2 x half> addrspace(1)* %gep0
  %add = fadd <2 x half> %a, <half 1.0, half 1.0>
  %shuf = shufflevector <2 x half> %add, <2 x half> undef, <2 x i32> <i32 1, i32 0>

  %max = call <2 x half> @llvm.maxnum.v2f16(<2 x half> %shuf, <2 x half> zeroinitializer)
  %clamp = call <2 x half> @llvm.minnum.v2f16(<2 x half> %max, <2 x half> <half 1.0, half 1.0>)
  store <2 x half> %clamp, <2 x half> addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_no_clamp_add_src_v2f16_f32_src:
; GCN-DAG: {{buffer|flat|global}}_load_dword [[A:v[0-9]+]]
; GFX9: v_add_f32_e32 [[ADD:v[0-9]+]], 1.0, [[A]]{{$}}
; GFX9: v_pk_max_f16 [[CLAMP:v[0-9]+]], [[ADD]], [[ADD]] clamp{{$}}
define amdgpu_kernel void @v_no_clamp_add_src_v2f16_f32_src(<2 x half> addrspace(1)* %out, <2 x half> addrspace(1)* %aptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr <2 x half>, <2 x half> addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr <2 x half>, <2 x half> addrspace(1)* %out, i32 %tid
  %a = load <2 x half>, <2 x half> addrspace(1)* %gep0
  %bc = bitcast <2 x half> %a to float
  %f32.op = fadd float %bc, 1.0
  %f32.op.cast = bitcast float %f32.op to <2 x half>
  %max = call <2 x half> @llvm.maxnum.v2f16(<2 x half> %f32.op.cast, <2 x half> zeroinitializer)
  %clamp = call <2 x half> @llvm.minnum.v2f16(<2 x half> %max, <2 x half> <half 1.0, half 1.0>)
  store <2 x half> %clamp, <2 x half> addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_no_clamp_add_packed_src_f32:
; GCN:  {{buffer|flat|global}}_load_dword [[A:v[0-9]+]]
; GFX9: v_pk_add_f16 [[ADD:v[0-9]+]], [[A]], 1.0 op_sel_hi:[1,0]{{$}}
; GFX9: v_max_f32_e64 [[CLAMP:v[0-9]+]], [[ADD]], [[ADD]] clamp{{$}}
define amdgpu_kernel void @v_no_clamp_add_packed_src_f32(float addrspace(1)* %out, <2 x half> addrspace(1)* %aptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr <2 x half>, <2 x half> addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load <2 x half>, <2 x half> addrspace(1)* %gep0
  %add = fadd <2 x half> %a, <half 1.0, half 1.0>
  %bc.add = bitcast <2 x half> %add to float
  %max = call float @llvm.maxnum.f32(float %bc.add, float 0.0)
  %clamp = call float @llvm.minnum.f32(float %max, float 1.0)
  store float %clamp, float addrspace(1)* %out.gep
  ret void
}

; Since the high bits are zeroed, it probably would be OK in this case
; to use clamp.
; GCN-LABEL: {{^}}v_no_clamp_add_src_v2f16_f16_src:
; GCN-DAG: {{buffer|flat|global}}_load_ushort [[A:v[0-9]+]]
; GFX9: v_add_f16_e32 [[ADD:v[0-9]+]], 1.0, [[A]]{{$}}
; GFX9: v_pk_max_f16 [[CLAMP:v[0-9]+]], [[ADD]], [[ADD]] clamp{{$}}
define amdgpu_kernel void @v_no_clamp_add_src_v2f16_f16_src(<2 x half> addrspace(1)* %out, half addrspace(1)* %aptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr half, half addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr <2 x half>, <2 x half> addrspace(1)* %out, i32 %tid
  %a = load half, half addrspace(1)* %gep0
  %add = fadd half %a, 1.0
  %bc = bitcast half %add to i16
  %zext = zext i16 %bc to i32
  %v2f16 = bitcast i32 %zext to <2 x half>
  %max = call <2 x half> @llvm.maxnum.v2f16(<2 x half> %v2f16, <2 x half> zeroinitializer)
  %clamp = call <2 x half> @llvm.minnum.v2f16(<2 x half> %max, <2 x half> <half 1.0, half 1.0>)
  store <2 x half> %clamp, <2 x half> addrspace(1)* %out.gep
  ret void
}

; FIXME: Worse code pre-gfx9

; GCN-LABEL: {{^}}v_clamp_cvt_pkrtz_src_v2f16_denorm:
; GFX9: s_waitcnt
; GFX9-NEXT: v_cvt_pkrtz_f16_f32 v0, v0, v1 clamp{{$}}
; GFX9-NEXT: s_setpc_b64

; VI: v_cvt_pkrtz_f16_f32 v0, v0, v1{{$}}
; VI: v_max_f16_sdwa
; VI: v_max_f16_e64
; VI: v_or_b32

; SI: v_cvt_pkrtz_f16_f32_e32 v0, v0, v1{{$}}
; SI-DAG: v_cvt_f32_f16_e64 v0, v0 clamp
; SI-DAG: v_cvt_f32_f16_e64 v1, v1 clamp
define <2 x half> @v_clamp_cvt_pkrtz_src_v2f16_denorm(float %a, float %b) #0 {
  %add = call <2 x half> @llvm.amdgcn.cvt.pkrtz(float %a, float %b)
  %max = call <2 x half> @llvm.maxnum.v2f16(<2 x half> %add, <2 x half> zeroinitializer)
  %clamp = call <2 x half> @llvm.minnum.v2f16(<2 x half> %max, <2 x half> <half 1.0, half 1.0>)
  ret <2 x half> %clamp
}

declare i32 @llvm.amdgcn.workitem.id.x() #1
declare float @llvm.fabs.f32(float) #1
declare float @llvm.floor.f32(float) #1
declare float @llvm.minnum.f32(float, float) #1
declare float @llvm.maxnum.f32(float, float) #1
declare float @llvm.amdgcn.fmed3.f32(float, float, float) #1
declare double @llvm.fabs.f64(double) #1
declare double @llvm.minnum.f64(double, double) #1
declare double @llvm.maxnum.f64(double, double) #1
declare half @llvm.fabs.f16(half) #1
declare half @llvm.minnum.f16(half, half) #1
declare half @llvm.maxnum.f16(half, half) #1
declare <2 x half> @llvm.minnum.v2f16(<2 x half>, <2 x half>) #1
declare <2 x half> @llvm.maxnum.v2f16(<2 x half>, <2 x half>) #1
declare <2 x float> @llvm.minnum.v2f32(<2 x float>, <2 x float>) #1
declare <2 x float> @llvm.maxnum.v2f32(<2 x float>, <2 x float>) #1
declare <2 x half> @llvm.amdgcn.cvt.pkrtz(float, float) #1


declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind "target-features"="+fp32-denormals" }
attributes #3 = { nounwind "target-features"="-fp64-fp16-denormals" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: NoDebug)
!1 = !DIFile(filename: "/tmp/foo.cl", directory: "/dev/null")
!2 = !{i32 2, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DILocalVariable(name: "add", arg: 1, scope: !5, file: !1, line: 1)
!5 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !6, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: true, unit: !0)
!6 = !DISubroutineType(types: !7)
!7 = !{null, !8}
!8 = !DIBasicType(name: "float", size: 32, align: 32)
!9 = !DIExpression()
!10 = !DILocation(line: 1, column: 42, scope: !5)
