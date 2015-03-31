; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck --check-prefix=SI --check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck --check-prefix=SI --check-prefix=FUNC %s
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
; SI-DAG: v_sub_i32_e32 [[NEG_RCP_LO:v[0-9]+]], 0, [[RCP_LO]]
; SI: v_cndmask_b32_e64
; SI: v_mul_hi_u32 [[E:v[0-9]+]], {{v[0-9]+}}, [[RCP]]
; SI-DAG: v_add_i32_e32 [[RCP_A_E:v[0-9]+]], [[E]], [[RCP]]
; SI-DAG: v_subrev_i32_e32 [[RCP_S_E:v[0-9]+]], [[E]], [[RCP]]
; SI: v_cndmask_b32_e64
; SI: v_mul_hi_u32 [[Quotient:v[0-9]+]]
; SI: v_mul_lo_i32 [[Num_S_Remainder:v[0-9]+]]
; SI-DAG: v_sub_i32_e32 [[Remainder:v[0-9]+]], {{[vs][0-9]+}}, [[Num_S_Remainder]]
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_cndmask_b32_e64
; SI: v_and_b32_e32 [[Tmp1:v[0-9]+]]
; SI-DAG: v_add_i32_e32 [[Quotient_A_One:v[0-9]+]], 1, [[Quotient]]
; SI-DAG: v_subrev_i32_e32 [[Quotient_S_One:v[0-9]+]],
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_add_i32_e32 [[Remainder_A_Den:v[0-9]+]],
; SI-DAG: v_subrev_i32_e32 [[Remainder_S_Den:v[0-9]+]],
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_cndmask_b32_e64
; SI: s_endpgm
define void @test_udivrem(i32 addrspace(1)* %out, i32 %x, i32 %y) {
  %result0 = udiv i32 %x, %y
  store i32 %result0, i32 addrspace(1)* %out
  %result1 = urem i32 %x, %y
  store i32 %result1, i32 addrspace(1)* %out
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

; SI-DAG: v_rcp_iflag_f32_e32 [[FIRST_RCP:v[0-9]+]]
; SI-DAG: v_mul_hi_u32 [[FIRST_RCP_HI:v[0-9]+]], [[FIRST_RCP]]
; SI-DAG: v_mul_lo_i32 [[FIRST_RCP_LO:v[0-9]+]], [[FIRST_RCP]]
; SI-DAG: v_sub_i32_e32 [[FIRST_NEG_RCP_LO:v[0-9]+]], 0, [[FIRST_RCP_LO]]
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_mul_hi_u32 [[FIRST_E:v[0-9]+]], {{v[0-9]+}}, [[FIRST_RCP]]
; SI-DAG: v_add_i32_e32 [[FIRST_RCP_A_E:v[0-9]+]], [[FIRST_E]], [[FIRST_RCP]]
; SI-DAG: v_subrev_i32_e32 [[FIRST_RCP_S_E:v[0-9]+]], [[FIRST_E]], [[FIRST_RCP]]
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_mul_hi_u32 [[FIRST_Quotient:v[0-9]+]]
; SI-DAG: v_mul_lo_i32 [[FIRST_Num_S_Remainder:v[0-9]+]]
; SI-DAG: v_sub_i32_e32 [[FIRST_Remainder:v[0-9]+]], {{[vs][0-9]+}}, [[FIRST_Num_S_Remainder]]
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_and_b32_e32 [[FIRST_Tmp1:v[0-9]+]]
; SI-DAG: v_add_i32_e32 [[FIRST_Quotient_A_One:v[0-9]+]], {{.*}}, [[FIRST_Quotient]]
; SI-DAG: v_subrev_i32_e32 [[FIRST_Quotient_S_One:v[0-9]+]],
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_add_i32_e32 [[FIRST_Remainder_A_Den:v[0-9]+]],
; SI-DAG: v_subrev_i32_e32 [[FIRST_Remainder_S_Den:v[0-9]+]],
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_rcp_iflag_f32_e32 [[SECOND_RCP:v[0-9]+]]
; SI-DAG: v_mul_hi_u32 [[SECOND_RCP_HI:v[0-9]+]], [[SECOND_RCP]]
; SI-DAG: v_mul_lo_i32 [[SECOND_RCP_LO:v[0-9]+]], [[SECOND_RCP]]
; SI-DAG: v_sub_i32_e32 [[SECOND_NEG_RCP_LO:v[0-9]+]], 0, [[SECOND_RCP_LO]]
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_mul_hi_u32 [[SECOND_E:v[0-9]+]], {{v[0-9]+}}, [[SECOND_RCP]]
; SI-DAG: v_add_i32_e32 [[SECOND_RCP_A_E:v[0-9]+]], [[SECOND_E]], [[SECOND_RCP]]
; SI-DAG: v_subrev_i32_e32 [[SECOND_RCP_S_E:v[0-9]+]], [[SECOND_E]], [[SECOND_RCP]]
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_mul_hi_u32 [[SECOND_Quotient:v[0-9]+]]
; SI-DAG: v_mul_lo_i32 [[SECOND_Num_S_Remainder:v[0-9]+]]
; SI-DAG: v_sub_i32_e32 [[SECOND_Remainder:v[0-9]+]], {{[vs][0-9]+}}, [[SECOND_Num_S_Remainder]]
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_and_b32_e32 [[SECOND_Tmp1:v[0-9]+]]
; SI-DAG: v_add_i32_e32 [[SECOND_Quotient_A_One:v[0-9]+]], {{.*}}, [[SECOND_Quotient]]
; SI-DAG: v_subrev_i32_e32 [[SECOND_Quotient_S_One:v[0-9]+]],
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_add_i32_e32 [[SECOND_Remainder_A_Den:v[0-9]+]],
; SI-DAG: v_subrev_i32_e32 [[SECOND_Remainder_S_Den:v[0-9]+]],
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_cndmask_b32_e64
; SI: s_endpgm
define void @test_udivrem_v2(<2 x i32> addrspace(1)* %out, <2 x i32> %x, <2 x i32> %y) {
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

; SI-DAG: v_rcp_iflag_f32_e32 [[FIRST_RCP:v[0-9]+]]
; SI-DAG: v_mul_hi_u32 [[FIRST_RCP_HI:v[0-9]+]], [[FIRST_RCP]]
; SI-DAG: v_mul_lo_i32 [[FIRST_RCP_LO:v[0-9]+]], [[FIRST_RCP]]
; SI-DAG: v_sub_i32_e32 [[FIRST_NEG_RCP_LO:v[0-9]+]], 0, [[FIRST_RCP_LO]]
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_mul_hi_u32 [[FIRST_E:v[0-9]+]], {{v[0-9]+}}, [[FIRST_RCP]]
; SI-DAG: v_add_i32_e32 [[FIRST_RCP_A_E:v[0-9]+]], [[FIRST_E]], [[FIRST_RCP]]
; SI-DAG: v_subrev_i32_e32 [[FIRST_RCP_S_E:v[0-9]+]], [[FIRST_E]], [[FIRST_RCP]]
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_mul_hi_u32 [[FIRST_Quotient:v[0-9]+]]
; SI-DAG: v_mul_lo_i32 [[FIRST_Num_S_Remainder:v[0-9]+]]
; SI-DAG: v_sub_i32_e32 [[FIRST_Remainder:v[0-9]+]], {{[vs][0-9]+}}, [[FIRST_Num_S_Remainder]]
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_and_b32_e32 [[FIRST_Tmp1:v[0-9]+]]
; SI-DAG: v_add_i32_e32 [[FIRST_Quotient_A_One:v[0-9]+]], {{.*}}, [[FIRST_Quotient]]
; SI-DAG: v_subrev_i32_e32 [[FIRST_Quotient_S_One:v[0-9]+]],
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_add_i32_e32 [[FIRST_Remainder_A_Den:v[0-9]+]],
; SI-DAG: v_subrev_i32_e32 [[FIRST_Remainder_S_Den:v[0-9]+]],
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_rcp_iflag_f32_e32 [[SECOND_RCP:v[0-9]+]]
; SI-DAG: v_mul_hi_u32 [[SECOND_RCP_HI:v[0-9]+]], [[SECOND_RCP]]
; SI-DAG: v_mul_lo_i32 [[SECOND_RCP_LO:v[0-9]+]], [[SECOND_RCP]]
; SI-DAG: v_sub_i32_e32 [[SECOND_NEG_RCP_LO:v[0-9]+]], 0, [[SECOND_RCP_LO]]
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_mul_hi_u32 [[SECOND_E:v[0-9]+]], {{v[0-9]+}}, [[SECOND_RCP]]
; SI-DAG: v_add_i32_e32 [[SECOND_RCP_A_E:v[0-9]+]], [[SECOND_E]], [[SECOND_RCP]]
; SI-DAG: v_subrev_i32_e32 [[SECOND_RCP_S_E:v[0-9]+]], [[SECOND_E]], [[SECOND_RCP]]
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_mul_hi_u32 [[SECOND_Quotient:v[0-9]+]]
; SI-DAG: v_mul_lo_i32 [[SECOND_Num_S_Remainder:v[0-9]+]]
; SI-DAG: v_sub_i32_e32 [[SECOND_Remainder:v[0-9]+]], {{[vs][0-9]+}}, [[SECOND_Num_S_Remainder]]
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_and_b32_e32 [[SECOND_Tmp1:v[0-9]+]]
; SI-DAG: v_add_i32_e32 [[SECOND_Quotient_A_One:v[0-9]+]], {{.*}}, [[SECOND_Quotient]]
; SI-DAG: v_subrev_i32_e32 [[SECOND_Quotient_S_One:v[0-9]+]],
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_add_i32_e32 [[SECOND_Remainder_A_Den:v[0-9]+]],
; SI-DAG: v_subrev_i32_e32 [[SECOND_Remainder_S_Den:v[0-9]+]],
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_rcp_iflag_f32_e32 [[THIRD_RCP:v[0-9]+]]
; SI-DAG: v_mul_hi_u32 [[THIRD_RCP_HI:v[0-9]+]], [[THIRD_RCP]]
; SI-DAG: v_mul_lo_i32 [[THIRD_RCP_LO:v[0-9]+]], [[THIRD_RCP]]
; SI-DAG: v_sub_i32_e32 [[THIRD_NEG_RCP_LO:v[0-9]+]], 0, [[THIRD_RCP_LO]]
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_mul_hi_u32 [[THIRD_E:v[0-9]+]], {{v[0-9]+}}, [[THIRD_RCP]]
; SI-DAG: v_add_i32_e32 [[THIRD_RCP_A_E:v[0-9]+]], [[THIRD_E]], [[THIRD_RCP]]
; SI-DAG: v_subrev_i32_e32 [[THIRD_RCP_S_E:v[0-9]+]], [[THIRD_E]], [[THIRD_RCP]]
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_mul_hi_u32 [[THIRD_Quotient:v[0-9]+]]
; SI-DAG: v_mul_lo_i32 [[THIRD_Num_S_Remainder:v[0-9]+]]
; SI-DAG: v_sub_i32_e32 [[THIRD_Remainder:v[0-9]+]], {{[vs][0-9]+}}, [[THIRD_Num_S_Remainder]]
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_and_b32_e32 [[THIRD_Tmp1:v[0-9]+]]
; SI-DAG: v_add_i32_e32 [[THIRD_Quotient_A_One:v[0-9]+]], {{.*}}, [[THIRD_Quotient]]
; SI-DAG: v_subrev_i32_e32 [[THIRD_Quotient_S_One:v[0-9]+]],
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_add_i32_e32 [[THIRD_Remainder_A_Den:v[0-9]+]],
; SI-DAG: v_subrev_i32_e32 [[THIRD_Remainder_S_Den:v[0-9]+]],
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_rcp_iflag_f32_e32 [[FOURTH_RCP:v[0-9]+]]
; SI-DAG: v_mul_hi_u32 [[FOURTH_RCP_HI:v[0-9]+]], [[FOURTH_RCP]]
; SI-DAG: v_mul_lo_i32 [[FOURTH_RCP_LO:v[0-9]+]], [[FOURTH_RCP]]
; SI-DAG: v_sub_i32_e32 [[FOURTH_NEG_RCP_LO:v[0-9]+]], 0, [[FOURTH_RCP_LO]]
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_mul_hi_u32 [[FOURTH_E:v[0-9]+]], {{v[0-9]+}}, [[FOURTH_RCP]]
; SI-DAG: v_add_i32_e32 [[FOURTH_RCP_A_E:v[0-9]+]], [[FOURTH_E]], [[FOURTH_RCP]]
; SI-DAG: v_subrev_i32_e32 [[FOURTH_RCP_S_E:v[0-9]+]], [[FOURTH_E]], [[FOURTH_RCP]]
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_mul_hi_u32 [[FOURTH_Quotient:v[0-9]+]]
; SI-DAG: v_mul_lo_i32 [[FOURTH_Num_S_Remainder:v[0-9]+]]
; SI-DAG: v_sub_i32_e32 [[FOURTH_Remainder:v[0-9]+]], {{[vs][0-9]+}}, [[FOURTH_Num_S_Remainder]]
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_and_b32_e32 [[FOURTH_Tmp1:v[0-9]+]]
; SI-DAG: v_add_i32_e32 [[FOURTH_Quotient_A_One:v[0-9]+]], {{.*}}, [[FOURTH_Quotient]]
; SI-DAG: v_subrev_i32_e32 [[FOURTH_Quotient_S_One:v[0-9]+]],
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_add_i32_e32 [[FOURTH_Remainder_A_Den:v[0-9]+]],
; SI-DAG: v_subrev_i32_e32 [[FOURTH_Remainder_S_Den:v[0-9]+]],
; SI-DAG: v_cndmask_b32_e64
; SI-DAG: v_cndmask_b32_e64
; SI: s_endpgm
define void @test_udivrem_v4(<4 x i32> addrspace(1)* %out, <4 x i32> %x, <4 x i32> %y) {
  %result0 = udiv <4 x i32> %x, %y
  store <4 x i32> %result0, <4 x i32> addrspace(1)* %out
  %result1 = urem <4 x i32> %x, %y
  store <4 x i32> %result1, <4 x i32> addrspace(1)* %out
  ret void
}
