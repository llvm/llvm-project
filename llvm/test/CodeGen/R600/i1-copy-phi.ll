; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s

; SI-LABEL: {{^}}br_i1_phi:
; SI: v_mov_b32_e32 [[REG:v[0-9]+]], 0{{$}}
; SI: s_and_saveexec_b64
; SI: s_xor_b64
; SI: v_mov_b32_e32 [[REG]], -1{{$}}
; SI: v_cmp_ne_i32_e64 {{s\[[0-9]+:[0-9]+\]}}, [[REG]], 0
; SI: s_and_saveexec_b64
; SI: s_xor_b64
; SI: s_endpgm
define void @br_i1_phi(i32 %arg, i1 %arg1) #0 {
bb:
  br i1 %arg1, label %bb2, label %bb3

bb2:                                              ; preds = %bb
  br label %bb3

bb3:                                              ; preds = %bb2, %bb
  %tmp = phi i1 [ true, %bb2 ], [ false, %bb ]
  br i1 %tmp, label %bb4, label %bb6

bb4:                                              ; preds = %bb3
  %tmp5 = mul i32 undef, %arg
  br label %bb6

bb6:                                              ; preds = %bb4, %bb3
  ret void
}
