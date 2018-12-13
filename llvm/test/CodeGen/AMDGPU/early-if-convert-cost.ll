; RUN:  llc -amdgpu-scalarize-global-loads=false  -stress-early-ifcvt -amdgpu-early-ifcvt=1 -march=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; FIXME: Most of these cases that don't trigger because of broken cost
; heuristics. Should not need -stress-early-ifcvt

; GCN-LABEL: {{^}}test_vccnz_ifcvt_triangle64:
; GCN: buffer_load_dwordx2 v{{\[}}[[VAL_LO:[0-9]+]]:[[VAL_HI:[0-9]+]]{{\]}}
; GCN: v_cmp_neq_f64_e32 vcc, 1.0, v{{\[}}[[VAL_LO]]:[[VAL_HI]]{{\]}}
; GCN: v_add_f64 v{{\[}}[[ADD_LO:[0-9]+]]:[[ADD_HI:[0-9]+]]{{\]}}, v{{\[}}[[VAL_LO]]:[[VAL_HI]]{{\]}}, v{{\[}}[[VAL_LO]]:[[VAL_HI]]{{\]}}
; GCN-DAG: v_cndmask_b32_e32 v[[RESULT_LO:[0-9]+]], v[[ADD_LO]], v[[VAL_LO]], vcc
; GCN-DAG: v_cndmask_b32_e32 v[[RESULT_HI:[0-9]+]], v[[ADD_HI]], v[[VAL_HI]], vcc
; GCN: buffer_store_dwordx2 v{{\[}}[[RESULT_LO]]:[[RESULT_HI]]{{\]}}
define amdgpu_kernel void @test_vccnz_ifcvt_triangle64(double addrspace(1)* %out, double addrspace(1)* %in) #0 {
entry:
  %v = load double, double addrspace(1)* %in
  %cc = fcmp oeq double %v, 1.000000e+00
  br i1 %cc, label %if, label %endif

if:
  %u = fadd double %v, %v
  br label %endif

endif:
  %r = phi double [ %v, %entry ], [ %u, %if ]
  store double %r, double addrspace(1)* %out
  ret void
}

; vcc branch with SGPR inputs
; GCN-LABEL: {{^}}test_vccnz_sgpr_ifcvt_triangle64:
; GCN: v_cmp_neq_f64
; GCN: v_add_f64
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
define amdgpu_kernel void @test_vccnz_sgpr_ifcvt_triangle64(double addrspace(1)* %out, double addrspace(4)* %in) #0 {
entry:
  %v = load double, double addrspace(4)* %in
  %cc = fcmp oeq double %v, 1.000000e+00
  br i1 %cc, label %if, label %endif

if:
  %u = fadd double %v, %v
  br label %endif

endif:
  %r = phi double [ %v, %entry ], [ %u, %if ]
  store double %r, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_vccnz_ifcvt_triangle96:
; GCN: v_cmp_neq_f32_e64 [[CMP:s\[[0-9]+:[0-9]+\]]], s{{[0-9]+}}, 1.0

; GCN: v_add_i32_e32
; GCN: v_add_i32_e32
; GCN: v_add_i32_e32
; GCN: s_mov_b64 vcc, [[CMP]]

; GCN: v_cndmask_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, vcc
; GCN: v_cndmask_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, vcc
; GCN: v_cndmask_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, vcc

; GCN-DAG: buffer_store_dwordx3
define amdgpu_kernel void @test_vccnz_ifcvt_triangle96(<3 x i32> addrspace(1)* %out, <3 x i32> addrspace(1)* %in, float %cnd) #0 {
entry:
  %v = load <3 x i32>, <3 x i32> addrspace(1)* %in
  %cc = fcmp oeq float %cnd, 1.000000e+00
  br i1 %cc, label %if, label %endif

if:
  %u = add <3 x i32> %v, %v
  br label %endif

endif:
  %r = phi <3 x i32> [ %v, %entry ], [ %u, %if ]
  store <3 x i32> %r, <3 x i32> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_vccnz_ifcvt_triangle128:
; GCN: v_cmp_neq_f32_e64 [[CMP:s\[[0-9]+:[0-9]+\]]], s{{[0-9]+}}, 1.0

; GCN: v_add_i32_e32
; GCN: v_add_i32_e32
; GCN: v_add_i32_e32
; GCN: v_add_i32_e32
; GCN: s_mov_b64 vcc, [[CMP]]

; GCN: v_cndmask_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, vcc
; GCN: v_cndmask_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, vcc
; GCN: v_cndmask_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, vcc
; GCN: v_cndmask_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, vcc

; GCN: buffer_store_dwordx4
define amdgpu_kernel void @test_vccnz_ifcvt_triangle128(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %in, float %cnd) #0 {
entry:
  %v = load <4 x i32>, <4 x i32> addrspace(1)* %in
  %cc = fcmp oeq float %cnd, 1.000000e+00
  br i1 %cc, label %if, label %endif

if:
  %u = add <4 x i32> %v, %v
  br label %endif

endif:
  %r = phi <4 x i32> [ %v, %entry ], [ %u, %if ]
  store <4 x i32> %r, <4 x i32> addrspace(1)* %out
  ret void
}
