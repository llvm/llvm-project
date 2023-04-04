; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s

; SI-LABEL: {{^}}br_i1_phi:

; SI: ; %bb
; SI:    s_mov_b64           [[TMP:s\[[0-9]+:[0-9]+\]]], 0

; SI: ; %bb2
; SI:    s_mov_b64           [[TMP]], exec

; SI: ; %bb3
; SI:    s_and_saveexec_b64  {{s\[[0-9]+:[0-9]+\]}}, [[TMP]]

define amdgpu_kernel void @br_i1_phi(i32 %arg) {
bb:
  %tidig = call i32 @llvm.amdgcn.workitem.id.x()
  %cmp = trunc i32 %tidig to i1
  br i1 %cmp, label %bb2, label %bb3

bb2:                                              ; preds = %bb
  br label %bb3

bb3:                                              ; preds = %bb2, %bb
  %tmp = phi i1 [ true, %bb2 ], [ false, %bb ]
  br i1 %tmp, label %bb4, label %bb6

bb4:                                              ; preds = %bb3
  %val = load volatile i32, ptr addrspace(1) undef
  %tmp5 = mul i32 %val, %arg
  br label %bb6

bb6:                                              ; preds = %bb4, %bb3
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #0

attributes #0 = { nounwind readnone }

; Make sure this won't crash.
; SI-LABEL: {{^}}vcopy_i1_undef
; SI: v_cndmask_b32_e64
; SI: v_cndmask_b32_e64
define <2 x float> @vcopy_i1_undef(ptr addrspace(1) %p) {
entry:
  br i1 undef, label %exit, label %false

false:
  %x = load <2 x float>, ptr addrspace(1) %p
  %cmp = fcmp one <2 x float> %x, zeroinitializer
  br label %exit

exit:
  %c = phi <2 x i1> [ undef, %entry ], [ %cmp, %false ]
  %ret = select <2 x i1> %c, <2 x float> <float 2.0, float 2.0>, <2 x float> <float 4.0, float 4.0>
  ret <2 x float> %ret
}
