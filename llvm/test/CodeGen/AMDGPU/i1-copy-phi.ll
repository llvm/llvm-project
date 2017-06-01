; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s

; SI-LABEL: {{^}}br_i1_phi:
; SI: v_mov_b32_e32 [[REG:v[0-9]+]], 0{{$}}
; SI: s_and_saveexec_b64
; SI: s_xor_b64
; SI: v_mov_b32_e32 [[REG]], -1{{$}}
; SI: v_cmp_ne_u32_e32 vcc, 0, [[REG]]
; SI: s_and_saveexec_b64
; SI: s_xor_b64
; SI: s_endpgm
define void @br_i1_phi(i32 %arg) {
bb:
  %tidig = call i32 @llvm.r600.read.tidig.x() #0
  %cmp = trunc i32 %tidig to i1
  br i1 %cmp, label %bb2, label %bb3

bb2:                                              ; preds = %bb
  br label %bb3

bb3:                                              ; preds = %bb2, %bb
  %tmp = phi i1 [ true, %bb2 ], [ false, %bb ]
  br i1 %tmp, label %bb4, label %bb6

bb4:                                              ; preds = %bb3
  %val = load volatile i32, i32 addrspace(1)* undef
  %tmp5 = mul i32 %val, %arg
  br label %bb6

bb6:                                              ; preds = %bb4, %bb3
  ret void
}

declare i32 @llvm.r600.read.tidig.x() #0

attributes #0 = { readnone }
