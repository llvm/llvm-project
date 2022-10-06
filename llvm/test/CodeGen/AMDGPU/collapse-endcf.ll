; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs -amdgpu-remove-redundant-endcf < %s | FileCheck -enable-var-scope -check-prefix=GCN %s

; Disabled endcf collapse at -O0.
; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs -O0 -amdgpu-remove-redundant-endcf < %s | FileCheck -enable-var-scope -check-prefix=GCN-O0 %s

; GCN-LABEL: {{^}}simple_nested_if:
; GCN:      s_and_saveexec_b64 [[SAVEEXEC:s\[[0-9:]+\]]]
; GCN-NEXT: s_cbranch_execz [[ENDIF:.LBB[0-9_]+]]
; GCN:      s_and_b64 exec, exec, vcc
; GCN-NEXT: s_cbranch_execz [[ENDIF]]
; GCN-NEXT: ; %bb.{{[0-9]+}}:
; GCN:      store_dword
; GCN-NEXT: {{^}}[[ENDIF]]:
; GCN-NEXT: s_or_b64 exec, exec, [[SAVEEXEC]]
; GCN: ds_write_b32
; GCN: s_endpgm
;
; GCN-O0-LABEL: {{^}}simple_nested_if:
; GCN-O0:      s_mov_b64 s[{{[0-9:]+}}], exec
; GCN-O0-DAG:  v_writelane_b32 [[VGPR:v[0-9]+]], s{{[0-9]+}}, [[OUTER_SPILL_LANE_0:[0-9]+]]
; GCN-O0-DAG:  v_writelane_b32 [[VGPR]], s{{[0-9]+}}, [[OUTER_SPILL_LANE_1:[0-9]+]]
; GCN-O0-NEXT: s_and_b64 s[{{[0-9:]+}}], s[{{[0-9:]+}}], s[{{[0-9:]+}}]
; GCN-O0-NEXT: s_mov_b64 exec, s[{{[0-9:]+}}]
; GCN-O0-NEXT: s_cbranch_execz [[ENDIF_OUTER:.LBB[0-9_]+]]
; GCN-O0-NEXT: ; %bb.{{[0-9]+}}:
; GCN-O0:      s_mov_b64 s[{{[0-9:]+}}], exec
; GCN-O0-DAG:  v_writelane_b32 [[VGPR]], s{{[0-9]+}}, [[INNER_SPILL_LANE_0:[0-9]+]]
; GCN-O0-DAG:  v_writelane_b32 [[VGPR]], s{{[0-9]+}}, [[INNER_SPILL_LANE_1:[0-9]+]]
; GCN-O0-NEXT: s_and_b64 s[{{[0-9:]+}}], s[{{[0-9:]+}}], s[{{[0-9:]+}}]
; GCN-O0-NEXT: s_mov_b64 exec, s[{{[0-9:]+}}]
; GCN-O0-NEXT: s_cbranch_execz [[ENDIF_INNER:.LBB[0-9_]+]]
; GCN-O0-NEXT: ; %bb.{{[0-9]+}}:
; GCN-O0:      store_dword
; GCN-O0-NEXT: {{^}}[[ENDIF_INNER]]:
; GCN-O0-DAG:  v_readlane_b32 s{{[0-9]+}}, [[VGPR]], [[INNER_SPILL_LANE_0]]
; GCN-O0-DAG:  v_readlane_b32 s{{[0-9]+}}, [[VGPR]], [[INNER_SPILL_LANE_1]]
; GCN-O0-NEXT: s_or_b64 exec, exec, s[{{[0-9:]+}}]
; GCN-O0-NEXT: {{^}}[[ENDIF_OUTER]]:
; GCN-O0-DAG:  v_readlane_b32 s{{[0-9]+}}, [[VGPR]], [[OUTER_SPILL_LANE_0]]
; GCN-O0-DAG:  v_readlane_b32 s{{[0-9]+}}, [[VGPR]], [[OUTER_SPILL_LANE_1]]
; GCN-O0-NEXT: s_or_b64 exec, exec, s[{{[0-9:]+}}]
; GCN-O0:      ds_write_b32
; GCN-O0:      s_endpgm
;
define amdgpu_kernel void @simple_nested_if(i32 addrspace(1)* nocapture %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = icmp ugt i32 %tmp, 1
  br i1 %tmp1, label %bb.outer.then, label %bb.outer.end

bb.outer.then:                                    ; preds = %bb
  %tmp4 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i32 %tmp
  store i32 0, i32 addrspace(1)* %tmp4, align 4
  %tmp5 = icmp eq i32 %tmp, 2
  br i1 %tmp5, label %bb.outer.end, label %bb.inner.then

bb.inner.then:                                    ; preds = %bb.outer.then
  %tmp7 = add i32 %tmp, 1
  %tmp9 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i32 %tmp7
  store i32 1, i32 addrspace(1)* %tmp9, align 4
  br label %bb.outer.end

bb.outer.end:                                     ; preds = %bb.outer.then, %bb.inner.then, %bb
  store i32 3, i32 addrspace(3)* null
  ret void
}

; GCN-LABEL: {{^}}uncollapsable_nested_if:
; GCN:      s_and_saveexec_b64 [[SAVEEXEC_OUTER:s\[[0-9:]+\]]]
; GCN-NEXT: s_cbranch_execz [[ENDIF_OUTER:.LBB[0-9_]+]]
; GCN:      s_and_saveexec_b64 [[SAVEEXEC_INNER:s\[[0-9:]+\]]]
; GCN-NEXT: s_cbranch_execz [[ENDIF_INNER:.LBB[0-9_]+]]
; GCN-NEXT: ; %bb.{{[0-9]+}}:
; GCN:      store_dword
; GCN-NEXT: {{^}}[[ENDIF_INNER]]:
; GCN-NEXT: s_or_b64 exec, exec, [[SAVEEXEC_INNER]]
; GCN:      store_dword
; GCN-NEXT: {{^}}[[ENDIF_OUTER]]:
; GCN-NEXT: s_or_b64 exec, exec, [[SAVEEXEC_OUTER]]
; GCN: ds_write_b32
; GCN: s_endpgm
;
; GCN-O0-LABEL: {{^}}uncollapsable_nested_if:
; GCN-O0:      s_mov_b64 s[{{[0-9:]+}}], exec
; GCN-O0-DAG:  v_writelane_b32 [[VGPR:v[0-9]+]], s{{[0-9]+}}, [[OUTER_SPILL_LANE_0:[0-9]+]]
; GCN-O0-DAG:  v_writelane_b32 [[VGPR]], s{{[0-9]+}}, [[OUTER_SPILL_LANE_1:[0-9]+]]
; GCN-O0-NEXT: s_and_b64 s[{{[0-9:]+}}]
; GCN-O0-NEXT: s_mov_b64 exec, s[{{[0-9:]+}}]
; GCN-O0-NEXT: s_cbranch_execz [[ENDIF_OUTER:.LBB[0-9_]+]]
; GCN-O0-NEXT: ; %bb.{{[0-9]+}}:
; GCN-O0:      s_mov_b64 s[{{[0-9:]+}}], exec
; GCN-O0-DAG:  v_writelane_b32 [[VGPR]], s{{[0-9]+}}, [[INNER_SPILL_LANE_0:[0-9]+]]
; GCN-O0-DAG:  v_writelane_b32 [[VGPR]], s{{[0-9]+}}, [[INNER_SPILL_LANE_1:[0-9]+]]
; GCN-O0-NEXT: s_and_b64 s[{{[0-9:]+}}]
; GCN-O0-NEXT: s_mov_b64 exec, s[{{[0-9:]+}}]
; GCN-O0-NEXT: s_cbranch_execz [[ENDIF_INNER:.LBB[0-9_]+]]
; GCN-O0-NEXT: ; %bb.{{[0-9]+}}:
; GCN-O0:      store_dword
; GCN-O0-NEXT: s_branch [[ENDIF_INNER]]
; GCN-O0-NEXT: {{^}}[[ENDIF_OUTER]]:
; GCN-O0-DAG:  v_readlane_b32 s{{[0-9]+}}, [[VGPR]], [[OUTER_SPILL_LANE_0]]
; GCN-O0-DAG:  v_readlane_b32 s{{[0-9]+}}, [[VGPR]], [[OUTER_SPILL_LANE_1]]
; GCN-O0-NEXT: s_or_b64 exec, exec, s[{{[0-9:]+}}]
; GCN-O0-NEXT: s_branch [[LAST_BB:.LBB[0-9_]+]]
; GCN-O0-NEXT: {{^}}[[ENDIF_INNER]]:
; GCN-O0-DAG:  v_readlane_b32 s{{[0-9]+}}, [[VGPR]], [[INNER_SPILL_LANE_0]]
; GCN-O0-DAG:  v_readlane_b32 s{{[0-9]+}}, [[VGPR]], [[INNER_SPILL_LANE_1]]
; GCN-O0-NEXT: s_or_b64 exec, exec, s[{{[0-9:]+}}]
; GCN-O0:      s_branch [[ENDIF_OUTER]]
; GCN-O0-NEXT: {{^}}[[LAST_BB]]:
; GCN-O0:      ds_write_b32
; GCN-O0:      s_endpgm
;
define amdgpu_kernel void @uncollapsable_nested_if(i32 addrspace(1)* nocapture %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = icmp ugt i32 %tmp, 1
  br i1 %tmp1, label %bb.outer.then, label %bb.outer.end

bb.outer.then:                                    ; preds = %bb
  %tmp4 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i32 %tmp
  store i32 0, i32 addrspace(1)* %tmp4, align 4
  %tmp5 = icmp eq i32 %tmp, 2
  br i1 %tmp5, label %bb.inner.end, label %bb.inner.then

bb.inner.then:                                    ; preds = %bb.outer.then
  %tmp7 = add i32 %tmp, 1
  %tmp8 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i32 %tmp7
  store i32 1, i32 addrspace(1)* %tmp8, align 4
  br label %bb.inner.end

bb.inner.end:                                     ; preds = %bb.inner.then, %bb.outer.then
  %tmp9 = add i32 %tmp, 2
  %tmp10 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i32 %tmp9
  store i32 2, i32 addrspace(1)* %tmp10, align 4
  br label %bb.outer.end

bb.outer.end:                                     ; preds = %bb.inner.then, %bb
  store i32 3, i32 addrspace(3)* null
  ret void
}

; GCN-LABEL: {{^}}nested_if_if_else:
; GCN:      s_and_saveexec_b64 [[SAVEEXEC_OUTER:s\[[0-9:]+\]]]
; GCN-NEXT: s_cbranch_execz [[ENDIF_OUTER:.LBB[0-9_]+]]
; GCN:      s_and_saveexec_b64 [[SAVEEXEC_INNER:s\[[0-9:]+\]]]
; GCN-NEXT: s_xor_b64 [[SAVEEXEC_INNER2:s\[[0-9:]+\]]], exec, [[SAVEEXEC_INNER]]
; GCN-NEXT: s_cbranch_execz [[THEN_INNER:.LBB[0-9_]+]]
; GCN-NEXT: ; %bb.{{[0-9]+}}:
; GCN:      store_dword
; GCN:      {{^}}[[THEN_INNER]]:
; GCN-NEXT: s_andn2_saveexec_b64 [[SAVEEXEC_INNER2]], [[SAVEEXEC_INNER2]]
; GCN-NEXT: s_cbranch_execz [[ENDIF_OUTER]]
; GCN:      store_dword
; GCN-NEXT: {{^}}[[ENDIF_OUTER]]:
; GCN-NEXT: s_or_b64 exec, exec, [[SAVEEXEC_OUTER]]
; GCN: ds_write_b32
; GCN: s_endpgm
;
; GCN-O0-LABEL: {{^}}nested_if_if_else:
; GCN-O0:      s_mov_b64 s[{{[0-9:]+}}], exec
; GCN-O0-DAG:  v_writelane_b32 [[VGPR:v[0-9]+]], s{{[0-9]+}}, [[OUTER_SPILL_LANE_0:[0-9]+]]
; GCN-O0-DAG:  v_writelane_b32 [[VGPR]], s{{[0-9]+}}, [[OUTER_SPILL_LANE_1:[0-9]+]]
; GCN-O0-NEXT: s_and_b64 s[{{[0-9:]+}}]
; GCN-O0-NEXT: s_mov_b64 exec, s[{{[0-9:]+}}]
; GCN-O0-NEXT: s_cbranch_execz [[ENDIF_OUTER:.LBB[0-9_]+]]
; GCN-O0-NEXT: ; %bb.{{[0-9]+}}:
; GCN-O0:      s_mov_b64 s[{{[0-9:]+}}], exec
; GCN-O0-NEXT: s_and_b64 s[{{[0-9:]+}}], s[{{[0-9:]+}}], s[{{[0-9:]+}}]
; GCN-O0-NEXT: s_xor_b64 s[{{[0-9:]+}}], s[{{[0-9:]+}}], s[{{[0-9:]+}}]
; GCN-O0-DAG:  v_writelane_b32 [[VGPR]], s{{[0-9]+}}, [[THEN_SPILL_LANE_0:[0-9]+]]
; GCN-O0-DAG:  v_writelane_b32 [[VGPR]], s{{[0-9]+}}, [[THEN_SPILL_LANE_1:[0-9]+]]
; GCN-O0-NEXT: s_mov_b64 exec, s[{{[0-9:]+}}]
; GCN-O0-NEXT: s_cbranch_execz [[THEN_INNER:.LBB[0-9_]+]]
; GCN-O0-NEXT: s_branch [[TEMP_BB:.LBB[0-9_]+]]
; GCN-O0-NEXT: {{^}}[[THEN_INNER]]:
; GCN-O0-DAG:  v_readlane_b32 s{{[0-9]+}}, [[VGPR]], [[THEN_SPILL_LANE_0]]
; GCN-O0-DAG:  v_readlane_b32 s{{[0-9]+}}, [[VGPR]], [[THEN_SPILL_LANE_1]]
; GCN-O0-NEXT: s_or_saveexec_b64 s[{{[0-9:]+}}], s[{{[0-9:]+}}]
; GCN-O0-NEXT: s_and_b64 s[{{[0-9:]+}}], exec, s[{{[0-9:]+}}]
; GCN-O0-DAG:  v_writelane_b32 [[VGPR]], s{{[0-9]+}}, [[INNER_SPILL_LANE_0:[0-9]+]]
; GCN-O0-DAG:  v_writelane_b32 [[VGPR]], s{{[0-9]+}}, [[INNER_SPILL_LANE_1:[0-9]+]]
; GCN-O0-NEXT: s_xor_b64 exec, exec, s[{{[0-9:]+}}]
; GCN-O0-NEXT: s_cbranch_execz [[ENDIF_INNER:.LBB[0-9_]+]]
; GCN-O0-NEXT: ; %bb.{{[0-9]+}}:
; GCN-O0:      store_dword
; GCN-O0-NEXT: s_branch [[ENDIF_INNER]]
; GCN-O0-NEXT: {{^}}[[TEMP_BB]]:
; GCN-O0:      s_branch [[THEN_INNER]]
; GCN-O0-NEXT: {{^}}[[ENDIF_INNER]]:
; GCN-O0-DAG:  v_readlane_b32 s{{[0-9]+}}, [[VGPR]], [[INNER_SPILL_LANE_0]]
; GCN-O0-DAG:  v_readlane_b32 s{{[0-9]+}}, [[VGPR]], [[INNER_SPILL_LANE_1]]
; GCN-O0-NEXT: s_or_b64 exec, exec, s[{{[0-9:]+}}]
; GCN-O0-NEXT: {{^}}[[ENDIF_OUTER]]:
; GCN-O0-DAG:  v_readlane_b32 s{{[0-9]+}}, [[VGPR]], [[OUTER_SPILL_LANE_0]]
; GCN-O0-DAG:  v_readlane_b32 s{{[0-9]+}}, [[VGPR]], [[OUTER_SPILL_LANE_1]]
; GCN-O0-NEXT: s_or_b64 exec, exec, s[{{[0-9:]+}}]
; GCN-O0:      ds_write_b32
; GCN-O0:      s_endpgm
;
define amdgpu_kernel void @nested_if_if_else(i32 addrspace(1)* nocapture %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i32 %tmp
  store i32 0, i32 addrspace(1)* %tmp1, align 4
  %tmp2 = icmp ugt i32 %tmp, 1
  br i1 %tmp2, label %bb.outer.then, label %bb.outer.end

bb.outer.then:                                       ; preds = %bb
  %tmp5 = icmp eq i32 %tmp, 2
  br i1 %tmp5, label %bb.then, label %bb.else

bb.then:                                             ; preds = %bb.outer.then
  %tmp3 = add i32 %tmp, 1
  %tmp4 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i32 %tmp3
  store i32 1, i32 addrspace(1)* %tmp4, align 4
  br label %bb.outer.end

bb.else:                                             ; preds = %bb.outer.then
  %tmp7 = add i32 %tmp, 2
  %tmp9 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i32 %tmp7
  store i32 2, i32 addrspace(1)* %tmp9, align 4
  br label %bb.outer.end

bb.outer.end:                                        ; preds = %bb, %bb.then, %bb.else
  store i32 3, i32 addrspace(3)* null
  ret void
}

; GCN-LABEL: {{^}}nested_if_else_if:
; GCN:      s_and_saveexec_b64 [[SAVEEXEC_OUTER:s\[[0-9:]+\]]]
; GCN-NEXT: s_xor_b64 [[SAVEEXEC_OUTER2:s\[[0-9:]+\]]], exec, [[SAVEEXEC_OUTER]]
; GCN-NEXT: s_cbranch_execz [[THEN_OUTER:.LBB[0-9_]+]]
; GCN-NEXT: ; %bb.{{[0-9]+}}:
; GCN:      store_dword
; GCN-NEXT: s_and_saveexec_b64 [[SAVEEXEC_INNER_IF_OUTER_ELSE:s\[[0-9:]+\]]]
; GCN-NEXT: s_cbranch_execz [[THEN_OUTER_FLOW:.LBB[0-9_]+]]
; GCN-NEXT: ; %bb.{{[0-9]+}}:
; GCN:      store_dword
; GCN-NEXT: {{^}}[[THEN_OUTER_FLOW]]:
; GCN-NEXT: s_or_b64 exec, exec, [[SAVEEXEC_INNER_IF_OUTER_ELSE]]
; GCN:      {{^}}[[THEN_OUTER]]:
; GCN-NEXT: s_andn2_saveexec_b64 [[SAVEEXEC_OUTER2]], [[SAVEEXEC_OUTER2]]
; GCN-NEXT: s_cbranch_execz [[ENDIF_OUTER:.LBB[0-9_]+]]
; GCN-NEXT: ; %bb.{{[0-9]+}}:
; GCN:      store_dword
; GCN-NEXT: s_and_saveexec_b64 [[SAVEEXEC_ELSE:s\[[0-9:]+\]]],
; GCN-NEXT: s_cbranch_execz [[FLOW1:.LBB[0-9_]+]]
; GCN-NEXT: ; %bb.{{[0-9]+}}:
; GCN:      store_dword
; GCN-NEXT: [[FLOW1]]:
; GCN-NEXT: s_or_b64 exec, exec, [[SAVEEXEC_ELSE]]
; GCN:      s_or_b64 exec, exec, [[SAVEEXEC_OUTER2]]
; GCN:      ds_write_b32
; GCN:      s_endpgm
;
; GCN-O0-LABEL: {{^}}nested_if_else_if:
; GCN-O0:      s_mov_b64 s[{{[0-9:]+}}], exec
; GCN-O0-NEXT: s_and_b64 s[{{[0-9:]+}}], s[{{[0-9:]+}}], s[{{[0-9:]+}}]
; GCN-O0-NEXT: s_xor_b64 s[{{[0-9:]+}}], s[{{[0-9:]+}}], s[{{[0-9:]+}}]
; GCN-O0-DAG:  v_writelane_b32 [[VGPR:v[0-9]+]], s{{[0-9]+}}, [[OUTER_SPILL_LANE_0:[0-9]+]]
; GCN-O0-DAG:  v_writelane_b32 [[VGPR]], s{{[0-9]+}}, [[OUTER_SPILL_LANE_1:[0-9]+]]
; GCN-O0-NEXT: s_mov_b64 exec, s[{{[0-9:]+}}]
; GCN-O0-NEXT: s_cbranch_execz [[THEN_OUTER:.LBB[0-9_]+]]
; GCN-O0-NEXT: s_branch [[INNER_IF_OUTER_ELSE:.LBB[0-9_]+]]
; GCN-O0-NEXT: {{^}}[[THEN_OUTER]]:
; GCN-O0-DAG:  v_readlane_b32 s{{[0-9]+}}, [[VGPR]], [[OUTER_SPILL_LANE_0]]
; GCN-O0-DAG:  v_readlane_b32 s{{[0-9]+}}, [[VGPR]], [[OUTER_SPILL_LANE_1]]
; GCN-O0-NEXT: s_or_saveexec_b64 s[{{[0-9:]+}}], s[{{[0-9:]+}}]
; GCN-O0-NEXT: s_and_b64 s[{{[0-9:]+}}], exec, s[{{[0-9:]+}}]
; GCN-O0-DAG:  v_writelane_b32 [[VGPR]], s{{[0-9]+}}, [[OUTER_2_SPILL_LANE_0:[0-9]+]]
; GCN-O0-DAG:  v_writelane_b32 [[VGPR]], s{{[0-9]+}}, [[OUTER_2_SPILL_LANE_1:[0-9]+]]
; GCN-O0-NEXT: s_xor_b64 exec, exec, s[{{[0-9:]+}}]
; GCN-O0-NEXT: s_cbranch_execz [[ENDIF_OUTER:.LBB[0-9_]+]]
; GCN-O0-NEXT: ; %bb.{{[0-9]+}}:
; GCN-O0:      store_dword
; GCN-O0:      s_mov_b64 s[{{[0-9:]+}}], exec
; GCN-O0-DAG:  v_writelane_b32 [[VGPR]], s{{[0-9]+}}, [[ELSE_SPILL_LANE_0:[0-9]+]]
; GCN-O0-DAG:  v_writelane_b32 [[VGPR]], s{{[0-9]+}}, [[ELSE_SPILL_LANE_1:[0-9]+]]
; GCN-O0-NEXT: s_and_b64 s[{{[0-9:]+}}], s[{{[0-9:]+}}], s[{{[0-9:]+}}]
; GCN-O0-NEXT: s_mov_b64 exec, s[{{[0-9:]+}}]
; GCN-O0-NEXT: s_cbranch_execz [[FLOW1:.LBB[0-9_]+]]
; GCN-O0-NEXT: ; %bb.{{[0-9]+}}:
; GCN-O0:      store_dword
; GCN-O0-NEXT: s_branch [[FLOW1]]
; GCN-O0-NEXT: {{^}}[[INNER_IF_OUTER_ELSE]]
; GCN-O0:      s_mov_b64 s[{{[0-9:]+}}], exec
; GCN-O0-DAG:  v_writelane_b32 [[VGPR]], s{{[0-9]+}}, [[INNER_IF_OUTER_ELSE_SPILL_LANE_0:[0-9]+]]
; GCN-O0-DAG:  v_writelane_b32 [[VGPR]], s{{[0-9]+}}, [[INNER_IF_OUTER_ELSE_SPILL_LANE_1:[0-9]+]]
; GCN-O0-NEXT: s_and_b64 s[{{[0-9:]+}}], s[{{[0-9:]+}}], s[{{[0-9:]+}}]
; GCN-O0-NEXT: s_mov_b64 exec, s[{{[0-9:]+}}]
; GCN-O0-NEXT: s_cbranch_execz [[THEN_OUTER_FLOW:.LBB[0-9_]+]]
; GCN-O0-NEXT: ; %bb.{{[0-9]+}}:
; GCN-O0:      store_dword
; GCN-O0-NEXT: {{^}}[[THEN_OUTER_FLOW]]
; GCN-O0-DAG:  v_readlane_b32 s{{[0-9]+}}, [[VGPR]], [[INNER_IF_OUTER_ELSE_SPILL_LANE_0]]
; GCN-O0-DAG:  v_readlane_b32 s{{[0-9]+}}, [[VGPR]], [[INNER_IF_OUTER_ELSE_SPILL_LANE_1]]
; GCN-O0-NEXT: s_or_b64 exec, exec, s[{{[0-9:]+}}]
; GCN-O0-NEXT: s_branch [[THEN_OUTER]]
; GCN-O0-NEXT: {{^}}[[FLOW1]]
; GCN-O0-DAG:  v_readlane_b32 s{{[0-9]+}}, [[VGPR]], [[ELSE_SPILL_LANE_0]]
; GCN-O0-DAG:  v_readlane_b32 s{{[0-9]+}}, [[VGPR]], [[ELSE_SPILL_LANE_1]]
; GCN-O0-NEXT: s_or_b64 exec, exec, s[{{[0-9:]+}}]
; GCN-O0-NEXT: {{^}}[[ENDIF_OUTER]]
; GCN-O0-DAG:  v_readlane_b32 s{{[0-9]+}}, [[VGPR]], [[OUTER_2_SPILL_LANE_0]]
; GCN-O0-DAG:  v_readlane_b32 s{{[0-9]+}}, [[VGPR]], [[OUTER_2_SPILL_LANE_1]]
; GCN-O0-NEXT: s_or_b64 exec, exec, s[{{[0-9:]+}}]
; GCN-O0:      ds_write_b32
; GCN-O0:      s_endpgm
;
define amdgpu_kernel void @nested_if_else_if(i32 addrspace(1)* nocapture %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i32 %tmp
  store i32 0, i32 addrspace(1)* %tmp1, align 4
  %cc1 = icmp ugt i32 %tmp, 1
  br i1 %cc1, label %bb.outer.then, label %bb.outer.else

bb.outer.then:
  %tmp2 = getelementptr inbounds i32, i32 addrspace(1)* %tmp1, i32 1
  store i32 1, i32 addrspace(1)* %tmp2, align 4
  %cc2 = icmp eq i32 %tmp, 2
  br i1 %cc2, label %bb.inner.then, label %bb.outer.end

bb.inner.then:
  %tmp3 = getelementptr inbounds i32, i32 addrspace(1)* %tmp1, i32 2
  store i32 2, i32 addrspace(1)* %tmp3, align 4
  br label %bb.outer.end

bb.outer.else:
  %tmp4 = getelementptr inbounds i32, i32 addrspace(1)* %tmp1, i32 3
  store i32 3, i32 addrspace(1)* %tmp4, align 4
  %cc3 = icmp eq i32 %tmp, 2
  br i1 %cc3, label %bb.inner.then2, label %bb.outer.end

bb.inner.then2:
  %tmp5 = getelementptr inbounds i32, i32 addrspace(1)* %tmp1, i32 4
  store i32 4, i32 addrspace(1)* %tmp5, align 4
  br label %bb.outer.end

bb.outer.end:
  store i32 3, i32 addrspace(3)* null
  ret void
}

; GCN-LABEL: {{^}}s_endpgm_unsafe_barrier:
; GCN:      s_and_saveexec_b64 [[SAVEEXEC:s\[[0-9:]+\]]]
; GCN-NEXT: s_cbranch_execz [[ENDIF:.LBB[0-9_]+]]
; GCN-NEXT: ; %bb.{{[0-9]+}}:
; GCN:      store_dword
; GCN-NEXT: {{^}}[[ENDIF]]:
; GCN-NEXT: s_or_b64 exec, exec, [[SAVEEXEC]]
; GCN:      s_barrier
; GCN-NEXT: s_endpgm
;
; GCN-O0-LABEL: {{^}}s_endpgm_unsafe_barrier:
; GCN-O0:      s_mov_b64 s[{{[0-9:]+}}], exec
; GCN-O0-DAG:  v_writelane_b32 [[VGPR:v[0-9]+]], s{{[0-9]+}}, [[SPILL_LANE_0:[0-9]+]]
; GCN-O0-DAG:  v_writelane_b32 [[VGPR]], s{{[0-9]+}}, [[SPILL_LANE_1:[0-9]+]]
; GCN-O0-NEXT: s_and_b64 s[{{[0-9:]+}}]
; GCN-O0-NEXT: s_mov_b64 exec, s[{{[0-9:]+}}]
; GCN-O0-NEXT: s_cbranch_execz [[ENDIF:.LBB[0-9_]+]]
; GCN-O0-NEXT: ; %bb.{{[0-9]+}}:
; GCN-O0:      store_dword
; GCN-O0-NEXT: {{^}}[[ENDIF]]:
; GCN-O0-DAG:  v_readlane_b32 s{{[0-9]+}}, [[VGPR]], [[SPILL_LANE_0]]
; GCN-O0-DAG:  v_readlane_b32 s{{[0-9]+}}, [[VGPR]], [[SPILL_LANE_1]]
; GCN-O0-NEXT: s_or_b64 exec, exec, s[{{[0-9:]+}}]
; GCN-O0:      s_barrier
; GCN-O0:      s_endpgm
;
define amdgpu_kernel void @s_endpgm_unsafe_barrier(i32 addrspace(1)* nocapture %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = icmp ugt i32 %tmp, 1
  br i1 %tmp1, label %bb.then, label %bb.end

bb.then:                                          ; preds = %bb
  %tmp4 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i32 %tmp
  store i32 0, i32 addrspace(1)* %tmp4, align 4
  br label %bb.end

bb.end:                                           ; preds = %bb.then, %bb
  call void @llvm.amdgcn.s.barrier()
  ret void
}

; GCN-LABEL: {{^}}scc_liveness:

; GCN: [[BB1_OUTER_LOOP:.LBB[0-9]+_[0-9]+]]:
; GCN: s_or_b64 exec, exec, [[SAVEEXEC_OUTER:s\[[0-9:]+\]]]
;
; GCN: [[BB1_INNER_LOOP:.LBB[0-9]+_[0-9]+]]:
; GCN: s_or_b64 exec, exec, s{{\[[0-9]+:[0-9]+\]}}
; GCN: s_andn2_b64
; GCN-NEXT: s_cbranch_execz

; GCN: [[BB1_LOOP:.LBB[0-9]+_[0-9]+]]:
; GCN: s_andn2_b64 exec, exec,
; GCN-NEXT: s_cbranch_execnz [[BB1_LOOP]]

; GCN: buffer_load_dword v{{[0-9]+}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}, 0 offen

; GCN: s_and_saveexec_b64 [[SAVEEXEC_OUTER]], {{vcc|s\[[0-9:]+\]}}
; GCN-NEXT: s_cbranch_execz [[BB1_OUTER_LOOP]]

; GCN-NOT: s_or_b64 exec, exec

; GCN: s_or_b64 exec, exec, s{{\[[0-9]+:[0-9]+\]}}
; GCN: buffer_store_dword
; GCN: buffer_store_dword
; GCN: buffer_store_dword
; GCN: buffer_store_dword
; GCN: s_setpc_b64
;
; GCN-O0-LABEL: {{^}}scc_liveness:
; GCN-O0-COUNT-2: buffer_store_dword
; GCN-O0-DAG:  v_writelane_b32 [[VGPR:v[0-9]+]], s{{[0-9]+}}, [[INNER_LOOP_IN_EXEC_SPILL_LANE_0:[0-9]+]]
; GCN-O0-DAG:  v_writelane_b32 [[VGPR]], s{{[0-9]+}}, [[INNER_LOOP_IN_EXEC_SPILL_LANE_1:[0-9]+]]
; GCN-O0-DAG:  v_writelane_b32 [[VGPR]], s{{[0-9]+}}, [[INNER_LOOP_BACK_EDGE_EXEC_SPILL_LANE_0:[0-9]+]]
; GCN-O0-DAG:  v_writelane_b32 [[VGPR]], s{{[0-9]+}}, [[INNER_LOOP_BACK_EDGE_EXEC_SPILL_LANE_1:[0-9]+]]
; GCN-O0: [[INNER_LOOP:.LBB[0-9]+_[0-9]+]]:
; GCN-O0: buffer_load_dword
; GCN-O0-DAG:  v_readlane_b32 s{{[0-9]+}}, [[VGPR]], [[INNER_LOOP_BACK_EDGE_EXEC_SPILL_LANE_0]]
; GCN-O0-DAG:  v_readlane_b32 s{{[0-9]+}}, [[VGPR]], [[INNER_LOOP_BACK_EDGE_EXEC_SPILL_LANE_1]]
; GCN-O0-DAG:  v_readlane_b32 s{{[0-9]+}}, [[VGPR]], [[INNER_LOOP_IN_EXEC_SPILL_LANE_0]]
; GCN-O0-DAG:  v_readlane_b32 s{{[0-9]+}}, [[VGPR]], [[INNER_LOOP_IN_EXEC_SPILL_LANE_1]]
; GCN-O0-DAG:  v_writelane_b32 [[VGPR]], s{{[0-9]+}}, [[OUTER_LOOP_EXEC_SPILL_LANE_0:[0-9]+]]
; GCN-O0-DAG:  v_writelane_b32 [[VGPR]], s{{[0-9]+}}, [[OUTER_LOOP_EXEC_SPILL_LANE_1:[0-9]+]]
; GCN-O0:      s_or_b64 s[{{[0-9:]+}}], s[{{[0-9:]+}}], s[{{[0-9:]+}}]
; GCN-O0-DAG:  v_writelane_b32 [[VGPR]], s{{[0-9]+}}, [[INNER_LOOP_OUT_EXEC_SPILL_LANE_0:[0-9]+]]
; GCN-O0-DAG:  v_writelane_b32 [[VGPR]], s{{[0-9]+}}, [[INNER_LOOP_OUT_EXEC_SPILL_LANE_1:[0-9]+]]
; GCN-O0-DAG:  v_writelane_b32 [[VGPR]], s{{[0-9]+}}, [[INNER_LOOP_IN_EXEC_SPILL_LANE_0]]
; GCN-O0-DAG:  v_writelane_b32 [[VGPR]], s{{[0-9]+}}, [[INNER_LOOP_IN_EXEC_SPILL_LANE_1]]
; GCN-O0-NEXT: s_mov_b64 s[{{[0-9:]+}}], s[{{[0-9:]+}}]
; GCN-O0-DAG:  v_writelane_b32 [[VGPR]], s{{[0-9]+}}, [[INNER_LOOP_BACK_EDGE_EXEC_SPILL_LANE_0]]
; GCN-O0-DAG:  v_writelane_b32 [[VGPR]], s{{[0-9]+}}, [[INNER_LOOP_BACK_EDGE_EXEC_SPILL_LANE_1]]
; GCN-O0-NEXT: s_andn2_b64 exec, exec, s[{{[0-9:]+}}]
; GCN-O0-NEXT: s_cbranch_execnz [[INNER_LOOP]]
; GCN-O0-NEXT: ; %bb.{{[0-9]+}}:
; GCN-O0-DAG:  v_readlane_b32 s{{[0-9]+}}, [[VGPR]], [[INNER_LOOP_OUT_EXEC_SPILL_LANE_0]]
; GCN-O0-DAG:  v_readlane_b32 s{{[0-9]+}}, [[VGPR]], [[INNER_LOOP_OUT_EXEC_SPILL_LANE_1]]
; GCN-O0-NEXT: s_or_b64 exec, exec, s[{{[0-9:]+}}]
; GCN-O0:      s_mov_b64 s[{{[0-9:]+}}], exec
; GCN-O0-DAG:  v_writelane_b32 [[VGPR]], s{{[0-9]+}}, [[FLOW2_IN_EXEC_SPILL_LANE_0:[0-9]+]]
; GCN-O0-DAG:  v_writelane_b32 [[VGPR]], s{{[0-9]+}}, [[FLOW2_IN_EXEC_SPILL_LANE_1:[0-9]+]]
; GCN-O0-NEXT: s_and_b64 s[{{[0-9:]+}}], s[{{[0-9:]+}}], s[{{[0-9:]+}}]
; GCN-O0-NEXT: s_mov_b64 exec, s[{{[0-9:]+}}]
; GCN-O0-NEXT: s_cbranch_execz [[FLOW2:.LBB[0-9_]+]]
; GCN-O0: {{^}}[[FLOW2]]:
; GCN-O0-DAG:  v_readlane_b32 s{{[0-9]+}}, [[VGPR]], [[FLOW2_IN_EXEC_SPILL_LANE_0]]
; GCN-O0-DAG:  v_readlane_b32 s{{[0-9]+}}, [[VGPR]], [[FLOW2_IN_EXEC_SPILL_LANE_1]]
; GCN-O0:      s_branch [[FLOW:.LBB[0-9_]+]]
; GCN-O0: {{^}}[[FLOW]]:
; GCN-O0:      s_mov_b64 s[{{[0-9:]+}}], exec
; GCN-O0-DAG:  v_writelane_b32 [[VGPR]], s{{[0-9]+}}, [[FLOW3_IN_EXEC_SPILL_LANE_0:[0-9]+]]
; GCN-O0-DAG:  v_writelane_b32 [[VGPR]], s{{[0-9]+}}, [[FLOW3_IN_EXEC_SPILL_LANE_1:[0-9]+]]
; GCN-O0-NEXT: s_and_b64 s[{{[0-9:]+}}], s[{{[0-9:]+}}], s[{{[0-9:]+}}]
; GCN-O0-NEXT: s_mov_b64 exec, s[{{[0-9:]+}}]
; GCN-O0-NEXT: s_cbranch_execz [[FLOW3:.LBB[0-9_]+]]
; GCN-O0:      ; %bb.{{[0-9]+}}:
; GCN-O0-DAG:  v_writelane_b32 [[VGPR]], s{{[0-9]+}}, [[FLOW1_OUT_EXEC_SPILL_LANE_0:[0-9]+]]
; GCN-O0-DAG:  v_writelane_b32 [[VGPR]], s{{[0-9]+}}, [[FLOW1_OUT_EXEC_SPILL_LANE_1:[0-9]+]]
; GCN-O0: {{^}}[[FLOW3]]:
; GCN-O0-COUNT-4: buffer_load_dword
; GCN-O0-DAG:  v_readlane_b32 s{{[0-9]+}}, [[VGPR]], [[OUTER_LOOP_EXEC_SPILL_LANE_0]]
; GCN-O0-DAG:  v_readlane_b32 s{{[0-9]+}}, [[VGPR]], [[OUTER_LOOP_EXEC_SPILL_LANE_1]]
; GCN-O0-DAG:  v_readlane_b32 s{{[0-9]+}}, [[VGPR]], [[FLOW1_OUT_EXEC_SPILL_LANE_0]]
; GCN-O0-DAG:  v_readlane_b32 s{{[0-9]+}}, [[VGPR]], [[FLOW1_OUT_EXEC_SPILL_LANE_1]]
; GCN-O0:      s_and_b64 s[{{[0-9:]+}}], exec, s[{{[0-9:]+}}]
; GCN-O0-NEXT: s_or_b64 s[{{[0-9:]+}}], s[{{[0-9:]+}}], s[{{[0-9:]+}}]
; GCN-O0-COUNT-2: s_mov_b64
; GCN-O0-DAG:  v_writelane_b32 [[VGPR]], s{{[0-9]+}}, [[INNER_LOOP_IN_EXEC_SPILL_LANE_0]]
; GCN-O0-DAG:  v_writelane_b32 [[VGPR]], s{{[0-9]+}}, [[INNER_LOOP_IN_EXEC_SPILL_LANE_1]]
; GCN-O0-DAG:  v_writelane_b32 [[VGPR]], s{{[0-9]+}}, [[INNER_LOOP_BACK_EDGE_EXEC_SPILL_LANE_0]]
; GCN-O0-DAG:  v_writelane_b32 [[VGPR]], s{{[0-9]+}}, [[INNER_LOOP_BACK_EDGE_EXEC_SPILL_LANE_1]]
; GCN-O0-COUNT-4: buffer_store_dword
; GCN-O0:      s_andn2_b64 exec, exec, s[{{[0-9:]+}}]
; GCN-O0-NEXT: s_cbranch_execnz [[INNER_LOOP]]
; GCN-O0:      ; %bb.{{[0-9]+}}:
; GCN-O0-COUNT-4: buffer_store_dword
; GCN-O0:     s_setpc_b64
;
define void @scc_liveness(i32 %arg) local_unnamed_addr #0 {
bb:
  br label %bb1

bb1:                                              ; preds = %Flow1, %bb1, %bb
  %tmp = icmp slt i32 %arg, 519
  br i1 %tmp, label %bb2, label %bb1

bb2:                                              ; preds = %bb1
  %tmp3 = icmp eq i32 %arg, 0
  br i1 %tmp3, label %bb4, label %bb10

bb4:                                              ; preds = %bb2
  %tmp6 = load float, float addrspace(5)* undef
  %tmp7 = fcmp olt float %tmp6, 0.0
  br i1 %tmp7, label %bb8, label %Flow

bb8:                                              ; preds = %bb4
  %tmp9 = insertelement <4 x float> undef, float 0.0, i32 1
  br label %Flow

Flow:                                             ; preds = %bb8, %bb4
  %tmp8 = phi <4 x float> [ %tmp9, %bb8 ], [ zeroinitializer, %bb4 ]
  br label %bb10

bb10:                                             ; preds = %Flow, %bb2
  %tmp11 = phi <4 x float> [ zeroinitializer, %bb2 ], [ %tmp8, %Flow ]
  br i1 %tmp3, label %bb12, label %Flow1

Flow1:                                            ; preds = %bb10
  br label %bb1

bb12:                                             ; preds = %bb10
  store volatile <4 x float> %tmp11, <4 x float> addrspace(5)* undef, align 16
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #0
declare void @llvm.amdgcn.s.barrier() #1

attributes #0 = { nounwind readnone speculatable }
attributes #1 = { nounwind convergent }
attributes #2 = { nounwind }
