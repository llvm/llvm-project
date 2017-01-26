;RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck --check-prefix=SI --check-prefix=GCN --check-prefix=FUNC %s
;RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck --check-prefix=VI --check-prefix=GCN --check-prefix=FUNC %s
;RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck --check-prefix=EG --check-prefix=FUNC %s

;FUNC-LABEL: {{^}}s_test_sdiv:
;EG: RECIP_UINT
;EG: LSHL {{.*}}, 1,
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT

; GCN: s_bfe_u32
; GCN: s_bfe_u32
; GCN: s_bfe_u32
; GCN: s_bfe_u32
; GCN: s_bfe_u32
; GCN: s_bfe_u32
; GCN: s_bfe_u32
; GCN: s_bfe_u32
; GCN: s_bfe_u32
; GCN: s_bfe_u32
; GCN: s_bfe_u32
; GCN: s_bfe_u32
; GCN: s_bfe_u32
; GCN: s_bfe_u32
; GCN: s_bfe_u32
; GCN: s_bfe_u32
; GCN: s_bfe_u32
; GCN: s_bfe_u32
; GCN: s_bfe_u32
; GCN: s_bfe_u32
; GCN: s_bfe_u32
; GCN: s_bfe_u32
; GCN: s_bfe_u32
; GCN: s_bfe_u32
; GCN: s_bfe_u32
; GCN: s_bfe_u32
; GCN: s_bfe_u32
; GCN: s_bfe_u32
; GCN: s_bfe_u32
; GCN: s_bfe_u32
; GCN-NOT: v_mad_f32
; SI-NOT: v_lshr_b64
; VI-NOT: v_lshrrev_b64
; GCN: s_endpgm
define void @s_test_sdiv(i64 addrspace(1)* %out, i64 %x, i64 %y) {
  %result = sdiv i64 %x, %y
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

;FUNC-LABEL: {{^}}s_test_srem:
;EG: RECIP_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: AND_INT {{.*}}, 1,

;GCN: s_bfe_u32
;GCN: s_bfe_u32
;GCN: s_bfe_u32
;GCN: s_bfe_u32
;GCN: s_bfe_u32
;GCN: s_bfe_u32
;GCN: s_bfe_u32
;GCN: s_bfe_u32
;GCN: s_bfe_u32
;GCN: s_bfe_u32
;GCN: s_bfe_u32
;GCN: s_bfe_u32
;GCN: s_bfe_u32
;GCN: s_bfe_u32
;GCN: s_bfe_u32
;GCN: s_bfe_u32
;GCN: s_bfe_u32
;GCN: s_bfe_u32
;GCN: s_bfe_u32
;GCN: s_bfe_u32
;GCN: s_bfe_u32
;GCN: s_bfe_u32
;GCN: s_bfe_u32
;GCN: s_bfe_u32
;GCN: s_bfe_u32
;GCN: s_bfe_u32
;GCN: s_bfe_u32
;GCN: s_bfe_u32
;GCN: s_bfe_u32
;GCN: s_bfe_u32
;GCN-NOT: v_mad_f32
;SI-NOT: v_lshr_b64
;VI-NOT: v_lshrrev_b64
;GCN: s_endpgm
define void @s_test_srem(i64 addrspace(1)* %out, i64 %x, i64 %y) {
  %result = urem i64 %x, %y
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

;FUNC-LABEL: {{^}}test_sdiv3264:
;EG: RECIP_UINT
;EG-NOT: BFE_UINT

;GCN-NOT: s_bfe_u32
;GCN-NOT: v_mad_f32
;SI-NOT: v_lshr_b64
;VI-NOT: v_lshrrev_b64
;GCN: s_endpgm
define void @test_sdiv3264(i64 addrspace(1)* %out, i64 %x, i64 %y) {
  %1 = ashr i64 %x, 33
  %2 = ashr i64 %y, 33
  %result = sdiv i64 %1, %2
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

;FUNC-LABEL: {{^}}test_srem3264:
;EG: RECIP_UINT
;EG-NOT: BFE_UINT

;GCN-NOT: s_bfe_u32
;GCN-NOT: v_mad_f32
;SI-NOT: v_lshr_b64
;VI-NOT: v_lshrrev_b64
;GCN: s_endpgm
define void @test_srem3264(i64 addrspace(1)* %out, i64 %x, i64 %y) {
  %1 = ashr i64 %x, 33
  %2 = ashr i64 %y, 33
  %result = srem i64 %1, %2
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

;FUNC-LABEL: {{^}}test_sdiv2464:
;EG: INT_TO_FLT
;EG: INT_TO_FLT
;EG: FLT_TO_INT
;EG-NOT: RECIP_UINT
;EG-NOT: BFE_UINT

;GCN-NOT: s_bfe_u32
;GCN: v_mad_f32
;SI-NOT: v_lshr_b64
;VI-NOT: v_lshrrev_b64
;GCN: s_endpgm
define void @test_sdiv2464(i64 addrspace(1)* %out, i64 %x, i64 %y) {
  %1 = ashr i64 %x, 40
  %2 = ashr i64 %y, 40
  %result = sdiv i64 %1, %2
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

;FUNC-LABEL: {{^}}test_srem2464:
;EG: INT_TO_FLT
;EG: INT_TO_FLT
;EG: FLT_TO_INT
;EG-NOT: RECIP_UINT
;EG-NOT: BFE_UINT

;GCN-NOT: s_bfe_u32
;GCN: v_mad_f32
;SI-NOT: v_lshr_b64
;VI-NOT: v_lshrrev_b64
;GCN: s_endpgm
define void @test_srem2464(i64 addrspace(1)* %out, i64 %x, i64 %y) {
  %1 = ashr i64 %x, 40
  %2 = ashr i64 %y, 40
  %result = srem i64 %1, %2
  store i64 %result, i64 addrspace(1)* %out
  ret void
}
