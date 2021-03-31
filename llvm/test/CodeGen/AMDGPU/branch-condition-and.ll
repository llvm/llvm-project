; RUN: llc -march=amdgcn -verify-machineinstrs -simplifycfg-require-and-preserve-domtree=1 < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs -simplifycfg-require-and-preserve-domtree=1 < %s | FileCheck -check-prefix=GCN %s

; This used to crash because during intermediate control flow lowering, there
; was a sequence
;       s_mov_b64 s[0:1], exec
;       s_and_b64 s[2:3], s[0:1], s[2:3] ; def & use of the same register pair
;       ...
;       s_mov_b64_term exec, s[2:3]
; that was not treated correctly.
;
; GCN-LABEL: {{^}}ham:
; GCN-DAG: v_cmp_lt_f32_e64 [[OTHERCC:s\[[0-9]+:[0-9]+\]]],
; GCN-DAG: v_cmp_lt_f32_e32 vcc,
; GCN: s_and_b64 [[AND:s\[[0-9]+:[0-9]+\]]], vcc, [[OTHERCC]]
; GCN: s_and_saveexec_b64 [[SAVED:s\[[0-9]+:[0-9]+\]]], [[AND]]
; GCN-NEXT: s_cbranch_execz BB0_{{[0-9]+}}

; GCN-NEXT: ; %bb.{{[0-9]+}}: ; %bb4
; GCN: ds_write_b32

; GCN: BB0_{{[0-9]+}}: ; %UnifiedReturnBlock
; GCN-NEXT: s_endpgm
; GCN-NEXT: .Lfunc_end
define amdgpu_ps void @ham(float %arg, float %arg1) #0 {
bb:
  %tmp = fcmp ogt float %arg, 0.000000e+00
  %tmp2 = fcmp ogt float %arg1, 0.000000e+00
  %tmp3 = and i1 %tmp, %tmp2
  br i1 %tmp3, label %bb4, label %bb5

bb4:                                              ; preds = %bb
  store volatile i32 4, i32 addrspace(3)* undef
  unreachable

bb5:                                              ; preds = %bb
  ret void
}

attributes #0 = { nounwind readonly "InitialPSInputAddr"="36983" }
attributes #1 = { nounwind readnone }
