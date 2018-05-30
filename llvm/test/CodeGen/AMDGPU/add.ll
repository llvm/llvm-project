; RUN: llc -march=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,SIVI,FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,SIVI,FUNC %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9,FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -enable-var-scope -check-prefix=EG -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}s_add_i32:
; EG: ADD_INT {{[* ]*}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

; GCN: s_add_i32 s[[REG:[0-9]+]], {{s[0-9]+, s[0-9]+}}
; GCN: v_mov_b32_e32 v[[V_REG:[0-9]+]], s[[REG]]
; GCN: buffer_store_dword v[[V_REG]],
define amdgpu_kernel void @s_add_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %b_ptr = getelementptr i32, i32 addrspace(1)* %in, i32 1
  %a = load i32, i32 addrspace(1)* %in
  %b = load i32, i32 addrspace(1)* %b_ptr
  %result = add i32 %a, %b
  store i32 %result, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}s_add_v2i32:
; EG: ADD_INT {{[* ]*}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; EG: ADD_INT {{[* ]*}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

; GCN: s_add_i32 s{{[0-9]+, s[0-9]+, s[0-9]+}}
; GCN: s_add_i32 s{{[0-9]+, s[0-9]+, s[0-9]+}}
define amdgpu_kernel void @s_add_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> addrspace(1)* %in) {
  %b_ptr = getelementptr <2 x i32>, <2 x i32> addrspace(1)* %in, i32 1
  %a = load <2 x i32>, <2 x i32> addrspace(1)* %in
  %b = load <2 x i32>, <2 x i32> addrspace(1)* %b_ptr
  %result = add <2 x i32> %a, %b
  store <2 x i32> %result, <2 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}s_add_v4i32:
; EG: ADD_INT {{[* ]*}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; EG: ADD_INT {{[* ]*}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; EG: ADD_INT {{[* ]*}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; EG: ADD_INT {{[* ]*}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

; GCN: s_add_i32 s{{[0-9]+, s[0-9]+, s[0-9]+}}
; GCN: s_add_i32 s{{[0-9]+, s[0-9]+, s[0-9]+}}
; GCN: s_add_i32 s{{[0-9]+, s[0-9]+, s[0-9]+}}
; GCN: s_add_i32 s{{[0-9]+, s[0-9]+, s[0-9]+}}
define amdgpu_kernel void @s_add_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %in) {
  %b_ptr = getelementptr <4 x i32>, <4 x i32> addrspace(1)* %in, i32 1
  %a = load <4 x i32>, <4 x i32> addrspace(1)* %in
  %b = load <4 x i32>, <4 x i32> addrspace(1)* %b_ptr
  %result = add <4 x i32> %a, %b
  store <4 x i32> %result, <4 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}s_add_v8i32:
; EG: ADD_INT
; EG: ADD_INT
; EG: ADD_INT
; EG: ADD_INT
; EG: ADD_INT
; EG: ADD_INT
; EG: ADD_INT
; EG: ADD_INT

; GCN: s_add_i32
; GCN: s_add_i32
; GCN: s_add_i32
; GCN: s_add_i32
; GCN: s_add_i32
; GCN: s_add_i32
; GCN: s_add_i32
; GCN: s_add_i32
define amdgpu_kernel void @s_add_v8i32(<8 x i32> addrspace(1)* %out, <8 x i32> %a, <8 x i32> %b) {
entry:
  %0 = add <8 x i32> %a, %b
  store <8 x i32> %0, <8 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}s_add_v16i32:
; EG: ADD_INT
; EG: ADD_INT
; EG: ADD_INT
; EG: ADD_INT
; EG: ADD_INT
; EG: ADD_INT
; EG: ADD_INT
; EG: ADD_INT
; EG: ADD_INT
; EG: ADD_INT
; EG: ADD_INT
; EG: ADD_INT
; EG: ADD_INT
; EG: ADD_INT
; EG: ADD_INT
; EG: ADD_INT

; GCN: s_add_i32
; GCN: s_add_i32
; GCN: s_add_i32
; GCN: s_add_i32
; GCN: s_add_i32
; GCN: s_add_i32
; GCN: s_add_i32
; GCN: s_add_i32
; GCN: s_add_i32
; GCN: s_add_i32
; GCN: s_add_i32
; GCN: s_add_i32
; GCN: s_add_i32
; GCN: s_add_i32
; GCN: s_add_i32
; GCN: s_add_i32
define amdgpu_kernel void @s_add_v16i32(<16 x i32> addrspace(1)* %out, <16 x i32> %a, <16 x i32> %b) {
entry:
  %0 = add <16 x i32> %a, %b
  store <16 x i32> %0, <16 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v_add_i32:
; GCN: {{buffer|flat|global}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat|global}}_load_dword [[B:v[0-9]+]]
; SIVI: v_add_{{i|u}}32_e32 v{{[0-9]+}}, vcc, [[B]], [[A]]
; GFX9: v_add_u32_e32 v{{[0-9]+}}, [[A]], [[B]]
define amdgpu_kernel void @v_add_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.r600.read.tidig.x()
  %gep = getelementptr inbounds i32, i32 addrspace(1)* %in, i32 %tid
  %b_ptr = getelementptr i32, i32 addrspace(1)* %gep, i32 1
  %a = load volatile i32, i32 addrspace(1)* %gep
  %b = load volatile i32, i32 addrspace(1)* %b_ptr
  %result = add i32 %a, %b
  store i32 %result, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v_add_imm_i32:
; GCN: {{buffer|flat|global}}_load_dword [[A:v[0-9]+]]
; SIVI: v_add_{{i|u}}32_e32 v{{[0-9]+}}, vcc, 0x7b, [[A]]
; GFX9: v_add_u32_e32 v{{[0-9]+}}, 0x7b, [[A]]
define amdgpu_kernel void @v_add_imm_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.r600.read.tidig.x()
  %gep = getelementptr inbounds i32, i32 addrspace(1)* %in, i32 %tid
  %b_ptr = getelementptr i32, i32 addrspace(1)* %gep, i32 1
  %a = load volatile i32, i32 addrspace(1)* %gep
  %result = add i32 %a, 123
  store i32 %result, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}add64:
; GCN: s_add_u32
; GCN: s_addc_u32

; EG: MEM_RAT_CACHELESS STORE_RAW [[LO:T[0-9]+\.XY]]
; EG-DAG: ADD_INT {{[* ]*}}
; EG-DAG: ADDC_UINT
; EG-DAG: ADD_INT
; EG-DAG: ADD_INT {{[* ]*}}
; EG-NOT: SUB
define amdgpu_kernel void @add64(i64 addrspace(1)* %out, i64 %a, i64 %b) {
entry:
  %add = add i64 %a, %b
  store i64 %add, i64 addrspace(1)* %out
  ret void
}

; The v_addc_u32 and v_add_i32 instruction can't read SGPRs, because they
; use VCC.  The test is designed so that %a will be stored in an SGPR and
; %0 will be stored in a VGPR, so the comiler will be forced to copy %a
; to a VGPR before doing the add.

; FUNC-LABEL: {{^}}add64_sgpr_vgpr:
; GCN-NOT: v_addc_u32_e32 s

; EG: MEM_RAT_CACHELESS STORE_RAW [[LO:T[0-9]+\.XY]]
; EG-DAG: ADD_INT {{[* ]*}}
; EG-DAG: ADDC_UINT
; EG-DAG: ADD_INT
; EG-DAG: ADD_INT {{[* ]*}}
; EG-NOT: SUB
define amdgpu_kernel void @add64_sgpr_vgpr(i64 addrspace(1)* %out, i64 %a, i64 addrspace(1)* %in) {
entry:
  %0 = load i64, i64 addrspace(1)* %in
  %1 = add i64 %a, %0
  store i64 %1, i64 addrspace(1)* %out
  ret void
}

; Test i64 add inside a branch.
; FUNC-LABEL: {{^}}add64_in_branch:
; GCN: s_add_u32
; GCN: s_addc_u32

; EG: MEM_RAT_CACHELESS STORE_RAW [[LO:T[0-9]+\.XY]]
; EG-DAG: ADD_INT {{[* ]*}}
; EG-DAG: ADDC_UINT
; EG-DAG: ADD_INT
; EG-DAG: ADD_INT {{[* ]*}}
; EG-NOT: SUB
define amdgpu_kernel void @add64_in_branch(i64 addrspace(1)* %out, i64 addrspace(1)* %in, i64 %a, i64 %b, i64 %c) {
entry:
  %0 = icmp eq i64 %a, 0
  br i1 %0, label %if, label %else

if:
  %1 = load i64, i64 addrspace(1)* %in
  br label %endif

else:
  %2 = add i64 %a, %b
  br label %endif

endif:
  %3 = phi i64 [%1, %if], [%2, %else]
  store i64 %3, i64 addrspace(1)* %out
  ret void
}

declare i32 @llvm.r600.read.tidig.x() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone speculatable }
