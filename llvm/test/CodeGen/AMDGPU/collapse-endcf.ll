; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}simple_nested_if:
; GCN:      s_and_saveexec_b64 [[SAVEEXEC:s\[[0-9:]+\]]]
; GCN-NEXT: ; mask branch [[ENDIF:BB[0-9_]+]]
; GCN-NEXT: s_cbranch_execz [[ENDIF]]
; GCN:      s_and_b64 exec, exec, vcc
; GCN-NEXT: ; mask branch [[ENDIF]]
; GCN-NEXT: {{^BB[0-9_]+}}:
; GCN:      store_dword
; GCN-NEXT: {{^}}[[ENDIF]]:
; GCN-NEXT: s_endpgm
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
  ret void
}

; GCN-LABEL: {{^}}uncollapsable_nested_if:
; GCN:      s_and_saveexec_b64 [[SAVEEXEC_OUTER:s\[[0-9:]+\]]]
; GCN-NEXT: ; mask branch [[ENDIF_OUTER:BB[0-9_]+]]
; GCN-NEXT: s_cbranch_execz [[ENDIF_OUTER]]
; GCN:      s_and_saveexec_b64 [[SAVEEXEC_INNER:s\[[0-9:]+\]]]
; GCN-NEXT: ; mask branch [[ENDIF_INNER:BB[0-9_]+]]
; GCN-NEXT: {{^BB[0-9_]+}}:
; GCN:      store_dword
; GCN-NEXT: {{^}}[[ENDIF_INNER]]:
; GCN-NEXT: s_or_b64 exec, exec, [[SAVEEXEC_INNER]]
; GCN:      store_dword
; GCN-NEXT: {{^}}[[ENDIF_OUTER]]:
; GCN-NEXT: s_endpgm
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
  ret void
}

; GCN-LABEL: {{^}}nested_if_if_else:
; GCN:      s_and_saveexec_b64 [[SAVEEXEC_OUTER:s\[[0-9:]+\]]]
; GCN-NEXT: ; mask branch [[ENDIF_OUTER:BB[0-9_]+]]
; GCN-NEXT: s_cbranch_execz [[ENDIF_OUTER]]
; GCN:      s_and_saveexec_b64 [[SAVEEXEC_INNER:s\[[0-9:]+\]]]
; GCN-NEXT: s_xor_b64 [[SAVEEXEC_INNER2:s\[[0-9:]+\]]], exec, [[SAVEEXEC_INNER]]
; GCN-NEXT: ; mask branch [[THEN_INNER:BB[0-9_]+]]
; GCN-NEXT: {{^BB[0-9_]+}}:
; GCN:      store_dword
; GCN-NEXT: {{^}}[[THEN_INNER]]:
; GCN-NEXT: s_or_saveexec_b64 [[SAVEEXEC_INNER3:s\[[0-9:]+\]]], [[SAVEEXEC_INNER2]]
; GCN-NEXT: s_xor_b64 exec, exec, [[SAVEEXEC_INNER3]]
; GCN-NEXT: ; mask branch [[ENDIF_OUTER]]
; GCN:      store_dword
; GCN-NEXT: {{^}}[[ENDIF_OUTER]]:
; GCN-NEXT: s_endpgm
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
  ret void
}

; GCN-LABEL: {{^}}nested_if_else_if:
; GCN:      s_and_saveexec_b64 [[SAVEEXEC_OUTER:s\[[0-9:]+\]]]
; GCN-NEXT: s_xor_b64 [[SAVEEXEC_OUTER2:s\[[0-9:]+\]]], exec, [[SAVEEXEC_OUTER]]
; GCN-NEXT: ; mask branch [[THEN_OUTER:BB[0-9_]+]]
; GCN-NEXT: s_cbranch_execz [[THEN_OUTER]]
; GCN-NEXT: {{^BB[0-9_]+}}:
; GCN:      store_dword
; GCN-NEXT: s_and_saveexec_b64 [[SAVEEXEC_INNER_IF_OUTER_ELSE:s\[[0-9:]+\]]]
; GCN-NEXT: ; mask branch [[THEN_OUTER_FLOW:BB[0-9_]+]]
; GCN-NEXT: {{^BB[0-9_]+}}:
; GCN:      store_dword
; GCN-NEXT: {{^}}[[THEN_OUTER_FLOW]]:
; GCN-NEXT: s_or_b64 exec, exec, [[SAVEEXEC_INNER_IF_OUTER_ELSE]]
; GCN-NEXT: {{^}}[[THEN_OUTER]]:
; GCN-NEXT: s_or_saveexec_b64 [[SAVEEXEC_OUTER3:s\[[0-9:]+\]]], [[SAVEEXEC_OUTER2]]
; GCN-NEXT: s_xor_b64 exec, exec, [[SAVEEXEC_OUTER3]]
; GCN-NEXT: ; mask branch [[ENDIF_OUTER:BB[0-9_]+]]
; GCN-NEXT: s_cbranch_execz [[ENDIF_OUTER]]
; GCN-NEXT: {{^BB[0-9_]+}}:
; GCN:      store_dword
; GCN-NEXT: s_and_saveexec_b64 [[SAVEEXEC_INNER_IF_OUTER_THEN:s\[[0-9:]+\]]]
; GCN-NEXT: ; mask branch [[ENDIF_OUTER]]
; GCN-NEXT: {{^BB[0-9_]+}}:
; GCN:      store_dword
; GCN-NEXT: {{^}}[[ENDIF_OUTER]]:
; GCN-NEXT: s_endpgm
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
  ret void
}

; GCN-LABEL: {{^}}s_endpgm_unsafe_barrier:
; GCN:      s_and_saveexec_b64 [[SAVEEXEC:s\[[0-9:]+\]]]
; GCN-NEXT: ; mask branch [[ENDIF:BB[0-9_]+]]
; GCN-NEXT: {{^BB[0-9_]+}}:
; GCN:      store_dword
; GCN-NEXT: {{^}}[[ENDIF]]:
; GCN-NEXT: s_or_b64 exec, exec, [[SAVEEXEC]]
; GCN:      s_barrier
; GCN-NEXT: s_endpgm
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

declare i32 @llvm.amdgcn.workitem.id.x() #0
declare void @llvm.amdgcn.s.barrier() #1

attributes #0 = { nounwind readnone speculatable }
attributes #1 = { nounwind convergent }
