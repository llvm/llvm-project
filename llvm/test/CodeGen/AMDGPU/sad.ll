; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}v_sad_u32_pat1:
; GCN: v_sad_u32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @v_sad_u32_pat1(i32 addrspace(1)* %out, i32 %a, i32 %b, i32 %c) {
  %icmp0 = icmp ugt i32 %a, %b
  %t0 = select i1 %icmp0, i32 %a, i32 %b

  %icmp1 = icmp ule i32 %a, %b
  %t1 = select i1 %icmp1, i32 %a, i32 %b

  %ret0 = sub i32 %t0, %t1
  %ret = add i32 %ret0, %c

  store i32 %ret, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_sad_u32_constant_pat1:
; GCN: v_sad_u32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, 20
define amdgpu_kernel void @v_sad_u32_constant_pat1(i32 addrspace(1)* %out, i32 %a) {
  %icmp0 = icmp ugt i32 %a, 90
  %t0 = select i1 %icmp0, i32 %a, i32 90

  %icmp1 = icmp ule i32 %a, 90
  %t1 = select i1 %icmp1, i32 %a, i32 90

  %ret0 = sub i32 %t0, %t1
  %ret = add i32 %ret0, 20

  store i32 %ret, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_sad_u32_pat2:
; GCN: v_sad_u32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @v_sad_u32_pat2(i32 addrspace(1)* %out, i32 %a, i32 %b, i32 %c) {
  %icmp0 = icmp ugt i32 %a, %b
  %sub0 = sub i32 %a, %b
  %sub1 = sub i32 %b, %a
  %ret0 = select i1 %icmp0, i32 %sub0, i32 %sub1

  %ret = add i32 %ret0, %c

  store i32 %ret, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_sad_u32_multi_use_sub_pat1:
; GCN: s_max_u32 s{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}
; GCN: s_min_u32 s{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}
; GCN: s_sub_i32 s{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}
; GCN: s_add_i32 s{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}
define amdgpu_kernel void @v_sad_u32_multi_use_sub_pat1(i32 addrspace(1)* %out, i32 %a, i32 %b, i32 %c) {
  %icmp0 = icmp ugt i32 %a, %b
  %t0 = select i1 %icmp0, i32 %a, i32 %b

  %icmp1 = icmp ule i32 %a, %b
  %t1 = select i1 %icmp1, i32 %a, i32 %b

  %ret0 = sub i32 %t0, %t1
  store volatile i32 %ret0, i32 *undef
  %ret = add i32 %ret0, %c

  store i32 %ret, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_sad_u32_multi_use_add_pat1:
; GCN: v_sad_u32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @v_sad_u32_multi_use_add_pat1(i32 addrspace(1)* %out, i32 %a, i32 %b, i32 %c) {
  %icmp0 = icmp ugt i32 %a, %b
  %t0 = select i1 %icmp0, i32 %a, i32 %b

  %icmp1 = icmp ule i32 %a, %b
  %t1 = select i1 %icmp1, i32 %a, i32 %b

  %ret0 = sub i32 %t0, %t1
  %ret = add i32 %ret0, %c
  store volatile i32 %ret, i32 *undef
  store i32 %ret, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_sad_u32_multi_use_max_pat1:
; GCN: v_sad_u32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @v_sad_u32_multi_use_max_pat1(i32 addrspace(1)* %out, i32 %a, i32 %b, i32 %c) {
  %icmp0 = icmp ugt i32 %a, %b
  %t0 = select i1 %icmp0, i32 %a, i32 %b
  store volatile i32 %t0, i32 *undef

  %icmp1 = icmp ule i32 %a, %b
  %t1 = select i1 %icmp1, i32 %a, i32 %b

  %ret0 = sub i32 %t0, %t1
  %ret = add i32 %ret0, %c

  store i32 %ret, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_sad_u32_multi_use_min_pat1:
; GCN: v_sad_u32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @v_sad_u32_multi_use_min_pat1(i32 addrspace(1)* %out, i32 %a, i32 %b, i32 %c) {
  %icmp0 = icmp ugt i32 %a, %b
  %t0 = select i1 %icmp0, i32 %a, i32 %b

  %icmp1 = icmp ule i32 %a, %b
  %t1 = select i1 %icmp1, i32 %a, i32 %b

  store volatile i32 %t1, i32 *undef

  %ret0 = sub i32 %t0, %t1
  %ret = add i32 %ret0, %c

  store i32 %ret, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_sad_u32_multi_use_sub_pat2:
; GCN: v_sad_u32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @v_sad_u32_multi_use_sub_pat2(i32 addrspace(1)* %out, i32 %a, i32 %b, i32 %c) {
  %icmp0 = icmp ugt i32 %a, %b
  %sub0 = sub i32 %a, %b
  store volatile i32 %sub0, i32 *undef
  %sub1 = sub i32 %b, %a
  %ret0 = select i1 %icmp0, i32 %sub0, i32 %sub1

  %ret = add i32 %ret0, %c

  store i32 %ret, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_sad_u32_multi_use_select_pat2:
; GCN: s_sub_i32 s{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}
; GCN: v_cmp_gt_u32_e32 vcc, s{{[0-9]+}}, v{{[0-9]+}}
; GCN: s_sub_i32 s{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}
define amdgpu_kernel void @v_sad_u32_multi_use_select_pat2(i32 addrspace(1)* %out, i32 %a, i32 %b, i32 %c) {
  %icmp0 = icmp ugt i32 %a, %b
  %sub0 = sub i32 %a, %b
  %sub1 = sub i32 %b, %a
  %ret0 = select i1 %icmp0, i32 %sub0, i32 %sub1
  store volatile i32 %ret0, i32 *undef

  %ret = add i32 %ret0, %c

  store i32 %ret, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_sad_u32_vector_pat1:
; GCN: v_sad_u32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_sad_u32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_sad_u32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_sad_u32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @v_sad_u32_vector_pat1(<4 x i32> addrspace(1)* %out, <4 x i32> %a, <4 x i32> %b, <4 x i32> %c) {
  %icmp0 = icmp ugt <4 x i32> %a, %b
  %t0 = select <4 x i1> %icmp0, <4 x i32> %a, <4 x i32> %b

  %icmp1 = icmp ule <4 x i32> %a, %b
  %t1 = select <4 x i1> %icmp1, <4 x i32> %a, <4 x i32> %b

  %ret0 = sub <4 x i32> %t0, %t1
  %ret = add <4 x i32> %ret0, %c

  store <4 x i32> %ret, <4 x i32> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_sad_u32_vector_pat2:
; GCN: v_sad_u32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_sad_u32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_sad_u32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_sad_u32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @v_sad_u32_vector_pat2(<4 x i32> addrspace(1)* %out, <4 x i32> %a, <4 x i32> %b, <4 x i32> %c) {
  %icmp0 = icmp ugt <4 x i32> %a, %b
  %sub0 = sub <4 x i32> %a, %b
  %sub1 = sub <4 x i32> %b, %a
  %ret0 = select <4 x i1> %icmp0, <4 x i32> %sub0, <4 x i32> %sub1

  %ret = add <4 x i32> %ret0, %c

  store <4 x i32> %ret, <4 x i32> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_sad_u32_i16_pat1:
; GCN: v_sad_u32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @v_sad_u32_i16_pat1(i16 addrspace(1)* %out, i16 %a, i16 %b, i16 %c) {

  %icmp0 = icmp ugt i16 %a, %b
  %t0 = select i1 %icmp0, i16 %a, i16 %b

  %icmp1 = icmp ule i16 %a, %b
  %t1 = select i1 %icmp1, i16 %a, i16 %b

  %ret0 = sub i16 %t0, %t1
  %ret = add i16 %ret0, %c

  store i16 %ret, i16 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_sad_u32_i16_pat2:
; GCN: v_sad_u32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @v_sad_u32_i16_pat2(i16 addrspace(1)* %out, i16 zeroext %a, i16 zeroext %b, i16 zeroext %c) {
  %icmp0 = icmp ugt i16 %a, %b
  %sub0 = sub i16 %a, %b
  %sub1 = sub i16 %b, %a
  %ret0 = select i1 %icmp0, i16 %sub0, i16 %sub1

  %ret = add i16 %ret0, %c

  store i16 %ret, i16 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_sad_u32_i8_pat1:
; GCN: v_sad_u32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @v_sad_u32_i8_pat1(i8 addrspace(1)* %out, i8 %a, i8 %b, i8 %c) {
  %icmp0 = icmp ugt i8 %a, %b
  %t0 = select i1 %icmp0, i8 %a, i8 %b

  %icmp1 = icmp ule i8 %a, %b
  %t1 = select i1 %icmp1, i8 %a, i8 %b

  %ret0 = sub i8 %t0, %t1
  %ret = add i8 %ret0, %c

  store i8 %ret, i8 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_sad_u32_i8_pat2:
; GCN: v_sad_u32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @v_sad_u32_i8_pat2(i8 addrspace(1)* %out, i8 zeroext %a, i8 zeroext %b, i8 zeroext %c) {
  %icmp0 = icmp ugt i8 %a, %b
  %sub0 = sub i8 %a, %b
  %sub1 = sub i8 %b, %a
  %ret0 = select i1 %icmp0, i8 %sub0, i8 %sub1

  %ret = add i8 %ret0, %c

  store i8 %ret, i8 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_sad_u32_mismatched_operands_pat1:
; GCN: v_cmp_le_u32_e32 vcc, s{{[0-9]+}}, v{{[0-9]+}}
; GCN: s_max_u32 s{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}
; GCN: v_sub_i32_e32 v{{[0-9]+}}, vcc, s{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_add_i32_e32 v{{[0-9]+}}, vcc, s{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @v_sad_u32_mismatched_operands_pat1(i32 addrspace(1)* %out, i32 %a, i32 %b, i32 %c, i32 %d) {
  %icmp0 = icmp ugt i32 %a, %b
  %t0 = select i1 %icmp0, i32 %a, i32 %b

  %icmp1 = icmp ule i32 %a, %b
  %t1 = select i1 %icmp1, i32 %a, i32 %d

  %ret0 = sub i32 %t0, %t1
  %ret = add i32 %ret0, %c

  store i32 %ret, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_sad_u32_mismatched_operands_pat2:
; GCN: s_sub_i32 s{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}
; GCN: s_sub_i32 s{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}
; GCN: v_add_i32_e32 v{{[0-9]+}}, vcc, s{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @v_sad_u32_mismatched_operands_pat2(i32 addrspace(1)* %out, i32 %a, i32 %b, i32 %c, i32 %d) {
  %icmp0 = icmp ugt i32 %a, %b
  %sub0 = sub i32 %a, %d
  %sub1 = sub i32 %b, %a
  %ret0 = select i1 %icmp0, i32 %sub0, i32 %sub1

  %ret = add i32 %ret0, %c

  store i32 %ret, i32 addrspace(1)* %out
  ret void
}

