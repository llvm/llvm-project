; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=VI -check-prefix=GCN %s

; GCN-LABEL: {{^}}mac_vvv:
; GCN: buffer_load_dword [[A:v[0-9]+]], off, s[{{[0-9]+:[0-9]+}}], 0{{$}}
; GCN: buffer_load_dword [[B:v[0-9]+]], off, s[{{[0-9]+:[0-9]+}}], 0 offset:4
; GCN: buffer_load_dword [[C:v[0-9]+]], off, s[{{[0-9]+:[0-9]+}}], 0 offset:8
; GCN: v_mac_f32_e32 [[C]], [[B]], [[A]]
; GCN: buffer_store_dword [[C]]
define void @mac_vvv(float addrspace(1)* %out, float addrspace(1)* %in) {
entry:
  %b_ptr = getelementptr float, float addrspace(1)* %in, i32 1
  %c_ptr = getelementptr float, float addrspace(1)* %in, i32 2

  %a = load volatile float, float addrspace(1)* %in
  %b = load volatile float, float addrspace(1)* %b_ptr
  %c = load volatile float, float addrspace(1)* %c_ptr

  %tmp0 = fmul float %a, %b
  %tmp1 = fadd float %tmp0, %c
  store float %tmp1, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}mad_inline_sgpr_inline:
; GCN-NOT: v_mac_f32
; GCN: v_mad_f32 v{{[0-9]}}, s{{[0-9]+}}, 0.5, 0.5
define void @mad_inline_sgpr_inline(float addrspace(1)* %out, float %in) {
entry:
  %tmp0 = fmul float 0.5, %in
  %tmp1 = fadd float %tmp0, 0.5
  store float %tmp1, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}mad_vvs:
; GCN-NOT: v_mac_f32
; GCN: v_mad_f32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, s{{[0-9]+}}
define void @mad_vvs(float addrspace(1)* %out, float addrspace(1)* %in, float %c) {
entry:
  %b_ptr = getelementptr float, float addrspace(1)* %in, i32 1

  %a = load float, float addrspace(1)* %in
  %b = load float, float addrspace(1)* %b_ptr

  %tmp0 = fmul float %a, %b
  %tmp1 = fadd float %tmp0, %c
  store float %tmp1, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}mac_ssv:
; GCN: v_mac_f32_e64 v{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}
define void @mac_ssv(float addrspace(1)* %out, float addrspace(1)* %in, float %a) {
entry:
  %c = load float, float addrspace(1)* %in

  %tmp0 = fmul float %a, %a
  %tmp1 = fadd float %tmp0, %c
  store float %tmp1, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}mac_mad_same_add:
; GCN: v_mad_f32 v{{[0-9]}}, v{{[0-9]+}}, v{{[0-9]+}}, [[ADD:v[0-9]+]]
; GCN: v_mac_f32_e32 [[ADD]], v{{[0-9]+}}, v{{[0-9]+}}
define void @mac_mad_same_add(float addrspace(1)* %out, float addrspace(1)* %in) {
entry:
  %b_ptr = getelementptr float, float addrspace(1)* %in, i32 1
  %c_ptr = getelementptr float, float addrspace(1)* %in, i32 2
  %d_ptr = getelementptr float, float addrspace(1)* %in, i32 3
  %e_ptr = getelementptr float, float addrspace(1)* %in, i32 4

  %a = load volatile float, float addrspace(1)* %in
  %b = load volatile float, float addrspace(1)* %b_ptr
  %c = load volatile float, float addrspace(1)* %c_ptr
  %d = load volatile float, float addrspace(1)* %d_ptr
  %e = load volatile float, float addrspace(1)* %e_ptr

  %tmp0 = fmul float %a, %b
  %tmp1 = fadd float %tmp0, %c

  %tmp2 = fmul float %d, %e
  %tmp3 = fadd float %tmp2, %c

  %out1 = getelementptr float, float addrspace(1)* %out, i32 1
  store float %tmp1, float addrspace(1)* %out
  store float %tmp3, float addrspace(1)* %out1
  ret void
}

; There is no advantage to using v_mac when one of the operands is negated
; and v_mad accepts more operand types.

; GCN-LABEL: {{^}}mad_neg_src0:
; GCN-NOT: v_mac_f32
; GCN: v_mad_f32 v{{[0-9]+}}, -v{{[0-9]+}}, v{{[0-9]+}}, v{{[-0-9]}}
define void @mad_neg_src0(float addrspace(1)* %out, float addrspace(1)* %in) #0 {
entry:
  %b_ptr = getelementptr float, float addrspace(1)* %in, i32 1
  %c_ptr = getelementptr float, float addrspace(1)* %in, i32 2

  %a = load float, float addrspace(1)* %in
  %b = load float, float addrspace(1)* %b_ptr
  %c = load float, float addrspace(1)* %c_ptr

  %neg_a = fsub float 0.0, %a
  %tmp0 = fmul float %neg_a, %b
  %tmp1 = fadd float %tmp0, %c

  store float %tmp1, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}mad_neg_src1:
; GCN-NOT: v_mac_f32
; GCN: v_mad_f32 v{{[0-9]+}}, -v{{[0-9]+}}, v{{[0-9]+}}, v{{[-0-9]}}
define void @mad_neg_src1(float addrspace(1)* %out, float addrspace(1)* %in) #0 {
entry:
  %b_ptr = getelementptr float, float addrspace(1)* %in, i32 1
  %c_ptr = getelementptr float, float addrspace(1)* %in, i32 2

  %a = load float, float addrspace(1)* %in
  %b = load float, float addrspace(1)* %b_ptr
  %c = load float, float addrspace(1)* %c_ptr

  %neg_b = fsub float 0.0, %b
  %tmp0 = fmul float %a, %neg_b
  %tmp1 = fadd float %tmp0, %c

  store float %tmp1, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}mad_neg_src2:
; GCN-NOT: v_mac
; GCN: v_mad_f32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, -v{{[-0-9]}}
define void @mad_neg_src2(float addrspace(1)* %out, float addrspace(1)* %in) #0 {
entry:
  %b_ptr = getelementptr float, float addrspace(1)* %in, i32 1
  %c_ptr = getelementptr float, float addrspace(1)* %in, i32 2

  %a = load float, float addrspace(1)* %in
  %b = load float, float addrspace(1)* %b_ptr
  %c = load float, float addrspace(1)* %c_ptr

  %neg_c = fsub float 0.0, %c
  %tmp0 = fmul float %a, %b
  %tmp1 = fadd float %tmp0, %neg_c

  store float %tmp1, float addrspace(1)* %out
  ret void
}

attributes #0 = { "true" "unsafe-fp-math"="true" }
