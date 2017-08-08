; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck --check-prefix=SI --check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck --check-prefix=SI --check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck --check-prefix=EG --check-prefix=FUNC %s

; FUNC-LABEL: {{^}}test_udivrem:
; EG: RECIP_UINT
; EG-DAG: MULHI
; EG-DAG: MULLO_INT
; EG-DAG: SUB_INT
; EG: CNDE_INT
; EG: MULHI
; EG-DAG: ADD_INT
; EG-DAG: SUB_INT
; EG: CNDE_INT
; EG: MULHI
; EG: MULLO_INT
; EG: SUB_INT
; EG-DAG: SETGE_UINT
; EG-DAG: SETGE_UINT
; EG: AND_INT
; EG-DAG: ADD_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: CNDE_INT
; EG-DAG: ADD_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: CNDE_INT

; SI: v_rcp_iflag_f32_e32 [[RCP:v[0-9]+]]
; SI-DAG: v_mul_hi_u32 [[RCP_HI:v[0-9]+]], [[RCP]]
; SI-DAG: v_mul_lo_i32 [[RCP_LO:v[0-9]+]], [[RCP]]
; SI-DAG: v_sub_i32_e32 [[NEG_RCP_LO:v[0-9]+]], vcc, 0, [[RCP_LO]]
; SI: v_cndmask_b32_e64
; SI: v_mul_hi_u32 [[E:v[0-9]+]], {{v[0-9]+}}, [[RCP]]
; SI-DAG: v_add_i32_e32 [[RCP_A_E:v[0-9]+]], vcc, [[E]], [[RCP]]
; SI-DAG: v_subrev_i32_e32 [[RCP_S_E:v[0-9]+]], vcc, [[E]], [[RCP]]
; SI: v_cndmask_b32_e64
; SI: v_mul_hi_u32 [[Quotient:v[0-9]+]]
; SI: v_mul_lo_i32 [[Num_S_Remainder:v[0-9]+]]
; SI-DAG: v_add_i32_e32 [[Quotient_A_One:v[0-9]+]], vcc, 1, [[Quotient]]
; SI-DAG: v_sub_i32_e32 [[Remainder:v[0-9]+]], vcc, {{[vs][0-9]+}}, [[Num_S_Remainder]]
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_subrev_i32_e32 [[Quotient_S_One:v[0-9]+]],
; SI-DAG: v_subrev_i32_e32 [[Remainder_S_Den:v[0-9]+]],
; SI: v_and_b32_e32 [[Tmp1:v[0-9]+]]
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_add_i32_e32 [[Remainder_A_Den:v[0-9]+]],
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_cndmask_b32_e64
; SI: s_endpgm
define amdgpu_kernel void @test_udivrem(i32 addrspace(1)* %out0, i32 addrspace(1)* %out1, i32 %x, i32 %y) {
  %result0 = udiv i32 %x, %y
  store i32 %result0, i32 addrspace(1)* %out0
  %result1 = urem i32 %x, %y
  store i32 %result1, i32 addrspace(1)* %out1
  ret void
}

; FUNC-LABEL: {{^}}test_udivrem_v2:
; EG-DAG: RECIP_UINT
; EG-DAG: MULHI
; EG-DAG: MULLO_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: MULHI
; EG-DAG: ADD_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: MULHI
; EG-DAG: MULLO_INT
; EG-DAG: SUB_INT
; EG-DAG: SETGE_UINT
; EG-DAG: SETGE_UINT
; EG-DAG: AND_INT
; EG-DAG: ADD_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: CNDE_INT
; EG-DAG: ADD_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: CNDE_INT
; EG-DAG: RECIP_UINT
; EG-DAG: MULHI
; EG-DAG: MULLO_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: MULHI
; EG-DAG: ADD_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: MULHI
; EG-DAG: MULLO_INT
; EG-DAG: SUB_INT
; EG-DAG: SETGE_UINT
; EG-DAG: SETGE_UINT
; EG-DAG: AND_INT
; EG-DAG: ADD_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: CNDE_INT
; EG-DAG: ADD_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: CNDE_INT

; For SI, we used to have checks for the input and output registers
; of the instructions, but these are way too fragile.  The division for
; the two vector elements can be intermixed which makes it impossible to
; accurately check all the operands.
; SI-DAG: v_rcp_iflag_f32_e32
; SI-DAG: v_mul_hi_u32
; SI-DAG: v_mul_lo_i32
; SI-DAG: v_sub_i32_e32
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_mul_hi_u32
; SI-DAG: v_add_i32_e32
; SI-DAG: v_subrev_i32_e32
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_mul_hi_u32
; SI-DAG: v_mul_lo_i32
; SI-DAG: v_subrev_i32_e32
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_and_b32_e32
; SI-DAG: v_add_i32_e32
; SI-DAG: v_subrev_i32_e32
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_add_i32_e32
; SI-DAG: v_subrev_i32_e32
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_rcp_iflag_f32_e32
; SI-DAG: v_mul_hi_u32
; SI-DAG: v_mul_lo_i32
; SI-DAG: v_sub_i32_e32
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_mul_hi_u32
; SI-DAG: v_add_i32_e32
; SI-DAG: v_subrev_i32_e32
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_mul_hi_u32
; SI-DAG: v_mul_lo_i32
; SI-DAG: v_subrev_i32_e32
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_and_b32_e32
; SI-DAG: v_add_i32_e32
; SI-DAG: v_subrev_i32_e32
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_add_i32_e32
; SI-DAG: v_subrev_i32_e32
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_cndmask_b32_e64
; SI: s_endpgm
define amdgpu_kernel void @test_udivrem_v2(<2 x i32> addrspace(1)* %out, <2 x i32> %x, <2 x i32> %y) {
  %result0 = udiv <2 x i32> %x, %y
  store <2 x i32> %result0, <2 x i32> addrspace(1)* %out
  %result1 = urem <2 x i32> %x, %y
  store <2 x i32> %result1, <2 x i32> addrspace(1)* %out
  ret void
}


; FUNC-LABEL: {{^}}test_udivrem_v4:
; EG-DAG: RECIP_UINT
; EG-DAG: MULHI
; EG-DAG: MULLO_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: MULHI
; EG-DAG: ADD_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: MULHI
; EG-DAG: MULLO_INT
; EG-DAG: SUB_INT
; EG-DAG: SETGE_UINT
; EG-DAG: SETGE_UINT
; EG-DAG: AND_INT
; EG-DAG: ADD_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: CNDE_INT
; EG-DAG: ADD_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: CNDE_INT
; EG-DAG: RECIP_UINT
; EG-DAG: MULHI
; EG-DAG: MULLO_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: MULHI
; EG-DAG: ADD_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: MULHI
; EG-DAG: MULLO_INT
; EG-DAG: SUB_INT
; EG-DAG: SETGE_UINT
; EG-DAG: SETGE_UINT
; EG-DAG: AND_INT
; EG-DAG: ADD_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: CNDE_INT
; EG-DAG: ADD_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: CNDE_INT
; EG-DAG: RECIP_UINT
; EG-DAG: MULHI
; EG-DAG: MULLO_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: MULHI
; EG-DAG: ADD_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: MULHI
; EG-DAG: MULLO_INT
; EG-DAG: SUB_INT
; EG-DAG: SETGE_UINT
; EG-DAG: SETGE_UINT
; EG-DAG: AND_INT
; EG-DAG: ADD_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: CNDE_INT
; EG-DAG: ADD_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: CNDE_INT
; EG-DAG: RECIP_UINT
; EG-DAG: MULHI
; EG-DAG: MULLO_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: MULHI
; EG-DAG: ADD_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: MULHI
; EG-DAG: MULLO_INT
; EG-DAG: SUB_INT
; EG-DAG: SETGE_UINT
; EG-DAG: SETGE_UINT
; EG-DAG: AND_INT
; EG-DAG: ADD_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: CNDE_INT
; EG-DAG: ADD_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: CNDE_INT

; SI-DAG: v_rcp_iflag_f32_e32
; SI-DAG: v_mul_hi_u32
; SI-DAG: v_mul_lo_i32
; SI-DAG: v_sub_i32_e32
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_mul_hi_u32
; SI-DAG: v_add_i32_e32
; SI-DAG: v_subrev_i32_e32
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_mul_hi_u32
; SI-DAG: v_mul_lo_i32
; SI-DAG: v_subrev_i32_e32
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_and_b32_e32
; SI-DAG: v_add_i32_e32
; SI-DAG: v_subrev_i32_e32
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_add_i32_e32
; SI-DAG: v_subrev_i32_e32
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_rcp_iflag_f32_e32
; SI-DAG: v_mul_hi_u32
; SI-DAG: v_mul_lo_i32
; SI-DAG: v_sub_i32_e32
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_mul_hi_u32
; SI-DAG: v_add_i32_e32
; SI-DAG: v_subrev_i32_e32
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_mul_hi_u32
; SI-DAG: v_mul_lo_i32
; SI-DAG: v_subrev_i32_e32
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_and_b32_e32
; SI-DAG: v_add_i32_e32
; SI-DAG: v_subrev_i32_e32
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_add_i32_e32
; SI-DAG: v_subrev_i32_e32
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_rcp_iflag_f32_e32
; SI-DAG: v_mul_hi_u32
; SI-DAG: v_mul_lo_i32
; SI-DAG: v_sub_i32_e32
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_mul_hi_u32
; SI-DAG: v_add_i32_e32
; SI-DAG: v_subrev_i32_e32
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_mul_hi_u32
; SI-DAG: v_mul_lo_i32
; SI-DAG: v_subrev_i32_e32
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_and_b32_e32
; SI-DAG: v_add_i32_e32
; SI-DAG: v_subrev_i32_e32
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_add_i32_e32
; SI-DAG: v_subrev_i32_e32
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_rcp_iflag_f32_e32
; SI-DAG: v_mul_hi_u32
; SI-DAG: v_mul_lo_i32
; SI-DAG: v_sub_i32_e32
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_mul_hi_u32
; SI-DAG: v_add_i32_e32
; SI-DAG: v_subrev_i32_e32
; SI-DAG: v_cndmask_b32_e64
; SI: s_endpgm
define amdgpu_kernel void @test_udivrem_v4(<4 x i32> addrspace(1)* %out, <4 x i32> %x, <4 x i32> %y) {
  %result0 = udiv <4 x i32> %x, %y
  store <4 x i32> %result0, <4 x i32> addrspace(1)* %out
  %result1 = urem <4 x i32> %x, %y
  store <4 x i32> %result1, <4 x i32> addrspace(1)* %out
  ret void
}
