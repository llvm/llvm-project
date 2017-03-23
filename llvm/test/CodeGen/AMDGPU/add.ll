; RUN: llc -march=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

;FUNC-LABEL: {{^}}test1:
;EG: ADD_INT {{[* ]*}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

;SI: v_add_i32_e32 [[REG:v[0-9]+]], vcc, {{v[0-9]+, v[0-9]+}}
;SI-NOT: [[REG]]
;SI: buffer_store_dword [[REG]],
define amdgpu_kernel void @test1(i32 addrspace(1)* %out, i32 addrspace(1)* %in) {
  %b_ptr = getelementptr i32, i32 addrspace(1)* %in, i32 1
  %a = load i32, i32 addrspace(1)* %in
  %b = load i32, i32 addrspace(1)* %b_ptr
  %result = add i32 %a, %b
  store i32 %result, i32 addrspace(1)* %out
  ret void
}

;FUNC-LABEL: {{^}}test2:
;EG: ADD_INT {{[* ]*}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG: ADD_INT {{[* ]*}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

;SI: v_add_i32_e32 v{{[0-9]+, vcc, v[0-9]+, v[0-9]+}}
;SI: v_add_i32_e32 v{{[0-9]+, vcc, v[0-9]+, v[0-9]+}}

define amdgpu_kernel void @test2(<2 x i32> addrspace(1)* %out, <2 x i32> addrspace(1)* %in) {
  %b_ptr = getelementptr <2 x i32>, <2 x i32> addrspace(1)* %in, i32 1
  %a = load <2 x i32>, <2 x i32> addrspace(1)* %in
  %b = load <2 x i32>, <2 x i32> addrspace(1)* %b_ptr
  %result = add <2 x i32> %a, %b
  store <2 x i32> %result, <2 x i32> addrspace(1)* %out
  ret void
}

;FUNC-LABEL: {{^}}test4:
;EG: ADD_INT {{[* ]*}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG: ADD_INT {{[* ]*}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG: ADD_INT {{[* ]*}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG: ADD_INT {{[* ]*}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

;SI: v_add_i32_e32 v{{[0-9]+, vcc, v[0-9]+, v[0-9]+}}
;SI: v_add_i32_e32 v{{[0-9]+, vcc, v[0-9]+, v[0-9]+}}
;SI: v_add_i32_e32 v{{[0-9]+, vcc, v[0-9]+, v[0-9]+}}
;SI: v_add_i32_e32 v{{[0-9]+, vcc, v[0-9]+, v[0-9]+}}

define amdgpu_kernel void @test4(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %in) {
  %b_ptr = getelementptr <4 x i32>, <4 x i32> addrspace(1)* %in, i32 1
  %a = load <4 x i32>, <4 x i32> addrspace(1)* %in
  %b = load <4 x i32>, <4 x i32> addrspace(1)* %b_ptr
  %result = add <4 x i32> %a, %b
  store <4 x i32> %result, <4 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test8:
; EG: ADD_INT
; EG: ADD_INT
; EG: ADD_INT
; EG: ADD_INT
; EG: ADD_INT
; EG: ADD_INT
; EG: ADD_INT
; EG: ADD_INT

; SI: s_add_i32
; SI: s_add_i32
; SI: s_add_i32
; SI: s_add_i32
; SI: s_add_i32
; SI: s_add_i32
; SI: s_add_i32
; SI: s_add_i32
define amdgpu_kernel void @test8(<8 x i32> addrspace(1)* %out, <8 x i32> %a, <8 x i32> %b) {
entry:
  %0 = add <8 x i32> %a, %b
  store <8 x i32> %0, <8 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test16:
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

; SI: s_add_i32
; SI: s_add_i32
; SI: s_add_i32
; SI: s_add_i32
; SI: s_add_i32
; SI: s_add_i32
; SI: s_add_i32
; SI: s_add_i32
; SI: s_add_i32
; SI: s_add_i32
; SI: s_add_i32
; SI: s_add_i32
; SI: s_add_i32
; SI: s_add_i32
; SI: s_add_i32
; SI: s_add_i32
define amdgpu_kernel void @test16(<16 x i32> addrspace(1)* %out, <16 x i32> %a, <16 x i32> %b) {
entry:
  %0 = add <16 x i32> %a, %b
  store <16 x i32> %0, <16 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}add64:
; SI: s_add_u32
; SI: s_addc_u32

; EG: MEM_RAT_CACHELESS STORE_RAW [[LO:T[0-9]+\.XY]]
; EG-DAG: ADD_INT {{[* ]*}}
; EG-DAG: ADDC_UINT
; EG-DAG: ADD_INT
; EG-DAG: ADD_INT {{[* ]*}}
; EG-NOT: SUB
define amdgpu_kernel void @add64(i64 addrspace(1)* %out, i64 %a, i64 %b) {
entry:
  %0 = add i64 %a, %b
  store i64 %0, i64 addrspace(1)* %out
  ret void
}

; The v_addc_u32 and v_add_i32 instruction can't read SGPRs, because they
; use VCC.  The test is designed so that %a will be stored in an SGPR and
; %0 will be stored in a VGPR, so the comiler will be forced to copy %a
; to a VGPR before doing the add.

; FUNC-LABEL: {{^}}add64_sgpr_vgpr:
; SI-NOT: v_addc_u32_e32 s

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
; SI: s_add_u32
; SI: s_addc_u32

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
