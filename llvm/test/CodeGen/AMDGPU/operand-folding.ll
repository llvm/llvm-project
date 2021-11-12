; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs -early-live-intervals < %s | FileCheck %s

; CHECK-LABEL: {{^}}fold_sgpr:
; CHECK: v_add_i32_e32 v{{[0-9]+}}, vcc, s
define amdgpu_kernel void @fold_sgpr(i32 addrspace(1)* %out, i32 %fold) #1 {
entry:
  %tmp0 = icmp ne i32 %fold, 0
  br i1 %tmp0, label %if, label %endif

if:
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %offset = add i32 %fold, %id
  %tmp1 = getelementptr i32, i32 addrspace(1)* %out, i32 %offset
  store i32 0, i32 addrspace(1)* %tmp1
  br label %endif

endif:
  ret void
}

; CHECK-LABEL: {{^}}fold_imm:
; CHECK: v_or_b32_e32 v{{[0-9]+}}, 5
define amdgpu_kernel void @fold_imm(i32 addrspace(1)* %out, i32 %cmp) #1 {
entry:
  %fold = add i32 3, 2
  %tmp0 = icmp ne i32 %cmp, 0
  br i1 %tmp0, label %if, label %endif

if:
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %val = or i32 %id, %fold
  store i32 %val, i32 addrspace(1)* %out
  br label %endif

endif:
  ret void
}

; CHECK-LABEL: {{^}}fold_64bit_constant_add:
; CHECK-NOT: s_mov_b64
; FIXME: It would be better if we could use v_add here and drop the extra
; v_mov_b32 instructions.
; CHECK-DAG: s_add_u32 [[LO:s[0-9]+]], s{{[0-9]+}}, 1
; CHECK-DAG: s_addc_u32 [[HI:s[0-9]+]], s{{[0-9]+}}, 0
; CHECK-DAG: v_mov_b32_e32 v[[VLO:[0-9]+]], [[LO]]
; CHECK-DAG: v_mov_b32_e32 v[[VHI:[0-9]+]], [[HI]]
; CHECK: buffer_store_dwordx2 v{{\[}}[[VLO]]:[[VHI]]{{\]}},

define amdgpu_kernel void @fold_64bit_constant_add(i64 addrspace(1)* %out, i32 %cmp, i64 %val) #1 {
entry:
  %tmp0 = add i64 %val, 1
  store i64 %tmp0, i64 addrspace(1)* %out
  ret void
}

; Inline constants should always be folded.

; CHECK-LABEL: {{^}}vector_inline:
; CHECK: v_xor_b32_e32 v{{[0-9]+}}, 5, v{{[0-9]+}}
; CHECK: v_xor_b32_e32 v{{[0-9]+}}, 5, v{{[0-9]+}}
; CHECK: v_xor_b32_e32 v{{[0-9]+}}, 5, v{{[0-9]+}}
; CHECK: v_xor_b32_e32 v{{[0-9]+}}, 5, v{{[0-9]+}}

define amdgpu_kernel void @vector_inline(<4 x i32> addrspace(1)* %out) #1 {
entry:
  %tmp0 = call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = add i32 %tmp0, 1
  %tmp2 = add i32 %tmp0, 2
  %tmp3 = add i32 %tmp0, 3
  %vec0 = insertelement <4 x i32> undef, i32 %tmp0, i32 0
  %vec1 = insertelement <4 x i32> %vec0, i32 %tmp1, i32 1
  %vec2 = insertelement <4 x i32> %vec1, i32 %tmp2, i32 2
  %vec3 = insertelement <4 x i32> %vec2, i32 %tmp3, i32 3
  %tmp4 = xor <4 x i32> <i32 5, i32 5, i32 5, i32 5>, %vec3
  store <4 x i32> %tmp4, <4 x i32> addrspace(1)* %out
  ret void
}

; Immediates with one use should be folded
; CHECK-LABEL: {{^}}imm_one_use:
; CHECK: v_xor_b32_e32 v{{[0-9]+}}, 0x64, v{{[0-9]+}}

define amdgpu_kernel void @imm_one_use(i32 addrspace(1)* %out) #1 {
entry:
  %tmp0 = call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = xor i32 %tmp0, 100
  store i32 %tmp1, i32 addrspace(1)* %out
  ret void
}
; CHECK-LABEL: {{^}}vector_imm:
; CHECK: s_movk_i32 [[IMM:s[0-9]+]], 0x64
; CHECK: v_xor_b32_e32 v{{[0-9]}}, [[IMM]], v{{[0-9]}}
; CHECK: v_xor_b32_e32 v{{[0-9]}}, [[IMM]], v{{[0-9]}}
; CHECK: v_xor_b32_e32 v{{[0-9]}}, [[IMM]], v{{[0-9]}}
; CHECK: v_xor_b32_e32 v{{[0-9]}}, [[IMM]], v{{[0-9]}}

define amdgpu_kernel void @vector_imm(<4 x i32> addrspace(1)* %out) #1 {
entry:
  %tmp0 = call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = add i32 %tmp0, 1
  %tmp2 = add i32 %tmp0, 2
  %tmp3 = add i32 %tmp0, 3
  %vec0 = insertelement <4 x i32> undef, i32 %tmp0, i32 0
  %vec1 = insertelement <4 x i32> %vec0, i32 %tmp1, i32 1
  %vec2 = insertelement <4 x i32> %vec1, i32 %tmp2, i32 2
  %vec3 = insertelement <4 x i32> %vec2, i32 %tmp3, i32 3
  %tmp4 = xor <4 x i32> <i32 100, i32 100, i32 100, i32 100>, %vec3
  store <4 x i32> %tmp4, <4 x i32> addrspace(1)* %out
  ret void
}

; A subregister use operand should not be tied.
; CHECK-LABEL: {{^}}no_fold_tied_subregister:
; CHECK: buffer_load_dwordx2 v{{\[}}[[LO:[0-9]+]]:[[HI:[0-9]+]]{{\]}}
; CHECK: v_mac_f32_e32 v[[LO]], 0x41200000, v[[HI]]
; CHECK: buffer_store_dword v[[LO]]
define amdgpu_kernel void @no_fold_tied_subregister() #1 {
  %tmp1 = load volatile <2 x float>, <2 x float> addrspace(1)* undef
  %tmp2 = extractelement <2 x float> %tmp1, i32 0
  %tmp3 = extractelement <2 x float> %tmp1, i32 1
  %tmp4 = fmul float %tmp3, 10.0
  %tmp5 = fadd float %tmp4, %tmp2
  store volatile float %tmp5, float addrspace(1)* undef
  ret void
}

; There should be exact one folding on the same operand.
; CHECK-LABEL: {{^}}no_extra_fold_on_same_opnd
; CHECK: v_xor_b32_e32 v{{[0-9]+}}, 0, v{{[0-9]+}}
; CHECK: v_xor_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define void @no_extra_fold_on_same_opnd() #1 {
entry:
  %s0 = load i32, i32 addrspace(5)* undef, align 4
  %s0.i64= zext i32 %s0 to i64
  br label %for.body.i.i

for.body.i.i:
  %s1 = load i32, i32 addrspace(1)* undef, align 8
  %s1.i64 = sext i32 %s1 to i64
  %xor = xor i64 %s1.i64, %s0.i64
  %flag = icmp ult i64 %xor, 8
  br i1 %flag, label %if.then, label %if.else

if.then:
  unreachable

if.else:
  unreachable
}

declare i32 @llvm.amdgcn.workitem.id.x() #0

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind "denormal-fp-math-f32"="preserve-sign,preserve-sign" }
