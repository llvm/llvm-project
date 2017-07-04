; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -mcpu=verde -amdgpu-early-ifcvt=1 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; XUN: llc -march=amdgcn -mcpu=tonga -amdgpu-early-ifcvt=1 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; FIXME: This leaves behind a now unnecessary and with exec

; GCN-LABEL: {{^}}test_vccnz_ifcvt_triangle:
; GCN: buffer_load_dword [[VAL:v[0-9]+]]
; GCN: v_cmp_neq_f32_e32 vcc, 1.0, [[VAL]]
; GCN: v_add_f32_e32 [[ADD:v[0-9]+]], [[VAL]], [[VAL]]
; GCN: v_cndmask_b32_e32 [[RESULT:v[0-9]+]], [[ADD]], [[VAL]], vcc
; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @test_vccnz_ifcvt_triangle(float addrspace(1)* %out, float addrspace(1)* %in) #0 {
entry:
  %v = load float, float addrspace(1)* %in
  %cc = fcmp oeq float %v, 1.000000e+00
  br i1 %cc, label %if, label %endif

if:
  %u = fadd float %v, %v
  br label %endif

endif:
  %r = phi float [ %v, %entry ], [ %u, %if ]
  store float %r, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_vccnz_ifcvt_diamond:
; GCN: buffer_load_dword [[VAL:v[0-9]+]]
; GCN: v_cmp_neq_f32_e32 vcc, 1.0, [[VAL]]
; GCN-DAG: v_add_f32_e32 [[ADD:v[0-9]+]], [[VAL]], [[VAL]]
; GCN-DAG: v_mul_f32_e32 [[MUL:v[0-9]+]], [[VAL]], [[VAL]]
; GCN: v_cndmask_b32_e32 [[RESULT:v[0-9]+]], [[ADD]], [[MUL]], vcc
; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @test_vccnz_ifcvt_diamond(float addrspace(1)* %out, float addrspace(1)* %in) #0 {
entry:
  %v = load float, float addrspace(1)* %in
  %cc = fcmp oeq float %v, 1.000000e+00
  br i1 %cc, label %if, label %else

if:
  %u0 = fadd float %v, %v
  br label %endif

else:
  %u1 = fmul float %v, %v
  br label %endif

endif:
  %r = phi float [ %u0, %if ], [ %u1, %else ]
  store float %r, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_vccnz_ifcvt_triangle_vcc_clobber:
; GCN: ; clobber vcc
; GCN: v_cmp_neq_f32_e64 [[CMP:s\[[0-9]+:[0-9]+\]]], s{{[0-9]+}}, 1.0
; GCN: v_add_i32_e32 v{{[0-9]+}}, vcc
; GCN: s_mov_b64 vcc, [[CMP]]
; GCN: v_cndmask_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, vcc
define amdgpu_kernel void @test_vccnz_ifcvt_triangle_vcc_clobber(i32 addrspace(1)* %out, i32 addrspace(1)* %in, float %k) #0 {
entry:
  %v = load i32, i32 addrspace(1)* %in
  %cc = fcmp oeq float %k, 1.000000e+00
  br i1 %cc, label %if, label %endif

if:
  call void asm "; clobber $0", "~{VCC}"() #0
  %u = add i32 %v, %v
  br label %endif

endif:
  %r = phi i32 [ %v, %entry ], [ %u, %if ]
  store i32 %r, i32 addrspace(1)* %out
  ret void
}

; Longest chain of cheap instructions to convert
; GCN-LABEL: {{^}}test_vccnz_ifcvt_triangle_max_cheap:
; GCN: v_mul_f32
; GCN: v_mul_f32
; GCN: v_mul_f32
; GCN: v_mul_f32
; GCN: v_mul_f32
; GCN: v_mul_f32
; GCN: v_mul_f32
; GCN: v_mul_f32
; GCN: v_mul_f32
; GCN: v_cndmask_b32_e32
define amdgpu_kernel void @test_vccnz_ifcvt_triangle_max_cheap(float addrspace(1)* %out, float addrspace(1)* %in) #0 {
entry:
  %v = load float, float addrspace(1)* %in
  %cc = fcmp oeq float %v, 1.000000e+00
  br i1 %cc, label %if, label %endif

if:
  %u.0 = fmul float %v, %v
  %u.1 = fmul float %v, %u.0
  %u.2 = fmul float %v, %u.1
  %u.3 = fmul float %v, %u.2
  %u.4 = fmul float %v, %u.3
  %u.5 = fmul float %v, %u.4
  %u.6 = fmul float %v, %u.5
  %u.7 = fmul float %v, %u.6
  %u.8 = fmul float %v, %u.7
  br label %endif

endif:
  %r = phi float [ %v, %entry ], [ %u.8, %if ]
  store float %r, float addrspace(1)* %out
  ret void
}

; Short chain of cheap instructions to not convert
; GCN-LABEL: {{^}}test_vccnz_ifcvt_triangle_min_expensive:
; GCN: s_cbranch_vccnz [[ENDIF:BB[0-9]+_[0-9]+]]

; GCN: v_mul_f32
; GCN: v_mul_f32
; GCN: v_mul_f32
; GCN: v_mul_f32
; GCN: v_mul_f32
; GCN: v_mul_f32
; GCN: v_mul_f32
; GCN: v_mul_f32
; GCN: v_mul_f32
; GCN: v_mul_f32

; GCN: [[ENDIF]]:
; GCN: buffer_store_dword
define amdgpu_kernel void @test_vccnz_ifcvt_triangle_min_expensive(float addrspace(1)* %out, float addrspace(1)* %in) #0 {
entry:
  %v = load float, float addrspace(1)* %in
  %cc = fcmp oeq float %v, 1.000000e+00
  br i1 %cc, label %if, label %endif

if:
  %u.0 = fmul float %v, %v
  %u.1 = fmul float %v, %u.0
  %u.2 = fmul float %v, %u.1
  %u.3 = fmul float %v, %u.2
  %u.4 = fmul float %v, %u.3
  %u.5 = fmul float %v, %u.4
  %u.6 = fmul float %v, %u.5
  %u.7 = fmul float %v, %u.6
  %u.8 = fmul float %v, %u.7
  %u.9 = fmul float %v, %u.8
  br label %endif

endif:
  %r = phi float [ %v, %entry ], [ %u.9, %if ]
  store float %r, float addrspace(1)* %out
  ret void
}

; Should still branch over fdiv expansion
; GCN-LABEL: {{^}}test_vccnz_ifcvt_triangle_expensive:
; GCN: v_cmp_neq_f32_e32
; GCN: s_cbranch_vccnz [[ENDIF:BB[0-9]+_[0-9]+]]

; GCN: v_div_scale_f32

; GCN: [[ENDIF]]:
; GCN: buffer_store_dword
define amdgpu_kernel void @test_vccnz_ifcvt_triangle_expensive(float addrspace(1)* %out, float addrspace(1)* %in) #0 {
entry:
  %v = load float, float addrspace(1)* %in
  %cc = fcmp oeq float %v, 1.000000e+00
  br i1 %cc, label %if, label %endif

if:
  %u = fdiv float %v, %v
  br label %endif

endif:
  %r = phi float [ %v, %entry ], [ %u, %if ]
  store float %r, float addrspace(1)* %out
  ret void
}

; vcc branch with SGPR inputs
; GCN-LABEL: {{^}}test_vccnz_sgpr_ifcvt_triangle:
; GCN: v_cmp_neq_f32_e64
; GCN: s_cbranch_vccnz [[ENDIF:BB[0-9]+_[0-9]+]]

; GCN: s_add_i32

; GCN: [[ENDIF]]:
; GCN: buffer_store_dword
define amdgpu_kernel void @test_vccnz_sgpr_ifcvt_triangle(i32 addrspace(1)* %out, i32 addrspace(2)* %in, float %cnd) #0 {
entry:
  %v = load i32, i32 addrspace(2)* %in
  %cc = fcmp oeq float %cnd, 1.000000e+00
  br i1 %cc, label %if, label %endif

if:
  %u = add i32 %v, %v
  br label %endif

endif:
  %r = phi i32 [ %v, %entry ], [ %u, %if ]
  store i32 %r, i32 addrspace(1)* %out
  ret void

}

; GCN-LABEL: {{^}}test_vccnz_ifcvt_triangle_constant_load:
; GCN: v_cndmask_b32
define amdgpu_kernel void @test_vccnz_ifcvt_triangle_constant_load(float addrspace(1)* %out, float addrspace(2)* %in) #0 {
entry:
  %v = load float, float addrspace(2)* %in
  %cc = fcmp oeq float %v, 1.000000e+00
  br i1 %cc, label %if, label %endif

if:
  %u = fadd float %v, %v
  br label %endif

endif:
  %r = phi float [ %v, %entry ], [ %u, %if ]
  store float %r, float addrspace(1)* %out
  ret void
}

; Due to broken cost heuristic, this is not if converted like
; test_vccnz_ifcvt_triangle_constant_load even though it should be.

; GCN-LABEL: {{^}}test_vccnz_ifcvt_triangle_argload:
; GCN: v_cndmask_b32
define amdgpu_kernel void @test_vccnz_ifcvt_triangle_argload(float addrspace(1)* %out, float %v) #0 {
entry:
  %cc = fcmp oeq float %v, 1.000000e+00
  br i1 %cc, label %if, label %endif

if:
  %u = fadd float %v, %v
  br label %endif

endif:
  %r = phi float [ %v, %entry ], [ %u, %if ]
  store float %r, float addrspace(1)* %out
  ret void
}

; Scalar branch and scalar inputs
; GCN-LABEL: {{^}}test_scc1_sgpr_ifcvt_triangle:
; GCN: s_load_dword [[VAL:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x0
; GCN: s_add_i32 [[ADD:s[0-9]+]], [[VAL]], [[VAL]]
; GCN: s_cmp_lg_u32 s{{[0-9]+}}, 1
; GCN-NEXT: s_cselect_b32 [[SELECT:s[0-9]+]], [[ADD]], [[VAL]]
define amdgpu_kernel void @test_scc1_sgpr_ifcvt_triangle(i32 addrspace(2)* %in, i32 %cond) #0 {
entry:
  %v = load i32, i32 addrspace(2)* %in
  %cc = icmp eq i32 %cond, 1
  br i1 %cc, label %if, label %endif

if:
  %u = add i32 %v, %v
  br label %endif

endif:
  %r = phi i32 [ %v, %entry ], [ %u, %if ]
  call void asm sideeffect "; reg use $0", "s"(i32 %r) #0
  ret void
}

; FIXME: Should be able to use VALU compare and select
; Scalar branch but VGPR select operands
; GCN-LABEL: {{^}}test_scc1_vgpr_ifcvt_triangle:
; GCN: s_cmp_lg_u32
; GCN: s_cbranch_scc1 [[ENDIF:BB[0-9]+_[0-9]+]]

; GCN: v_add_f32_e32

; GCN: [[ENDIF]]:
; GCN: buffer_store_dword
define amdgpu_kernel void @test_scc1_vgpr_ifcvt_triangle(float addrspace(1)* %out, float addrspace(1)* %in, i32 %cond) #0 {
entry:
  %v = load float, float addrspace(1)* %in
  %cc = icmp eq i32 %cond, 1
  br i1 %cc, label %if, label %endif

if:
  %u = fadd float %v, %v
  br label %endif

endif:
  %r = phi float [ %v, %entry ], [ %u, %if ]
  store float %r, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_scc1_sgpr_ifcvt_triangle64:
; GCN: s_add_u32
; GCN: s_addc_u32
; GCN: s_cmp_lg_u32 s{{[0-9]+}}, 1
; GCN-NEXT: s_cselect_b64 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @test_scc1_sgpr_ifcvt_triangle64(i64 addrspace(2)* %in, i32 %cond) #0 {
entry:
  %v = load i64, i64 addrspace(2)* %in
  %cc = icmp eq i32 %cond, 1
  br i1 %cc, label %if, label %endif

if:
  %u = add i64 %v, %v
  br label %endif

endif:
  %r = phi i64 [ %v, %entry ], [ %u, %if ]
  call void asm sideeffect "; reg use $0", "s"(i64 %r) #0
  ret void
}

; TODO: Can do s_cselect_b64; s_cselect_b32
; GCN-LABEL: {{^}}test_scc1_sgpr_ifcvt_triangle96:
; GCN: s_add_i32
; GCN: s_add_i32
; GCN: s_add_i32
; GCN: s_add_i32
; GCN: s_cmp_lg_u32 s{{[0-9]+}}, 1
; GCN-NEXT: s_cselect_b64 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}
; GCN-NEXT: s_cselect_b64 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @test_scc1_sgpr_ifcvt_triangle96(<3 x i32> addrspace(2)* %in, i32 %cond) #0 {
entry:
  %v = load <3 x i32>, <3 x i32> addrspace(2)* %in
  %cc = icmp eq i32 %cond, 1
  br i1 %cc, label %if, label %endif

if:
  %u = add <3 x i32> %v, %v
  br label %endif

endif:
  %r = phi <3 x i32> [ %v, %entry ], [ %u, %if ]
  %r.ext = shufflevector <3 x i32> %r, <3 x i32> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  call void asm sideeffect "; reg use $0", "s"(<4 x i32> %r.ext) #0
  ret void
}

; GCN-LABEL: {{^}}test_scc1_sgpr_ifcvt_triangle128:
; GCN: s_add_i32
; GCN: s_add_i32
; GCN: s_add_i32
; GCN: s_add_i32
; GCN: s_cmp_lg_u32 s{{[0-9]+}}, 1
; GCN-NEXT: s_cselect_b64 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}
; GCN-NEXT: s_cselect_b64 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @test_scc1_sgpr_ifcvt_triangle128(<4 x i32> addrspace(2)* %in, i32 %cond) #0 {
entry:
  %v = load <4 x i32>, <4 x i32> addrspace(2)* %in
  %cc = icmp eq i32 %cond, 1
  br i1 %cc, label %if, label %endif

if:
  %u = add <4 x i32> %v, %v
  br label %endif

endif:
  %r = phi <4 x i32> [ %v, %entry ], [ %u, %if ]
  call void asm sideeffect "; reg use $0", "s"(<4 x i32> %r) #0
  ret void
}

; GCN-LABEL: {{^}}uniform_if_swap_br_targets_scc_constant_select:
; GCN: s_cmp_lg_u32 s{{[0-9]+}}, 0
; GCN: s_cselect_b32 s{{[0-9]+}}, 1, 0{{$}}
define amdgpu_kernel void @uniform_if_swap_br_targets_scc_constant_select(i32 %cond, i32 addrspace(1)* %out) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %else, label %if

if:
  br label %done

else:
  br label %done

done:
  %value = phi i32 [0, %if], [1, %else]
  store i32 %value, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}ifcvt_undef_scc:
; GCN: {{^}}; BB#0:
; GCN-NEXT: s_load_dwordx2
; GCN-NEXT: s_cselect_b32 s{{[0-9]+}}, 1, 0
define amdgpu_kernel void @ifcvt_undef_scc(i32 %cond, i32 addrspace(1)* %out) {
entry:
  br i1 undef, label %else, label %if

if:
  br label %done

else:
  br label %done

done:
  %value = phi i32 [0, %if], [1, %else]
  store i32 %value, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_vccnz_ifcvt_triangle256:
; GCN: v_cmp_neq_f32
; GCN: s_cbranch_vccnz [[ENDIF:BB[0-9]+_[0-9]+]]

; GCN: v_add_i32
; GCN: v_add_i32

; GCN: [[ENDIF]]:
; GCN: buffer_store_dword
define amdgpu_kernel void @test_vccnz_ifcvt_triangle256(<8 x i32> addrspace(1)* %out, <8 x i32> addrspace(1)* %in, float %cnd) #0 {
entry:
  %v = load <8 x i32>, <8 x i32> addrspace(1)* %in
  %cc = fcmp oeq float %cnd, 1.000000e+00
  br i1 %cc, label %if, label %endif

if:
  %u = add <8 x i32> %v, %v
  br label %endif

endif:
  %r = phi <8 x i32> [ %v, %entry ], [ %u, %if ]
  store <8 x i32> %r, <8 x i32> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_vccnz_ifcvt_triangle512:
; GCN: v_cmp_neq_f32
; GCN: s_cbranch_vccnz [[ENDIF:BB[0-9]+_[0-9]+]]

; GCN: v_add_i32
; GCN: v_add_i32

; GCN: [[ENDIF]]:
; GCN: buffer_store_dword
define amdgpu_kernel void @test_vccnz_ifcvt_triangle512(<16 x i32> addrspace(1)* %out, <16 x i32> addrspace(1)* %in, float %cnd) #0 {
entry:
  %v = load <16 x i32>, <16 x i32> addrspace(1)* %in
  %cc = fcmp oeq float %cnd, 1.000000e+00
  br i1 %cc, label %if, label %endif

if:
  %u = add <16 x i32> %v, %v
  br label %endif

endif:
  %r = phi <16 x i32> [ %v, %entry ], [ %u, %if ]
  store <16 x i32> %r, <16 x i32> addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind }
