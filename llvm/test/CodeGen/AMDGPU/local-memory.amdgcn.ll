; RUN: llc -march=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=bonaire -verify-machineinstrs < %s | FileCheck -check-prefix=CI -check-prefix=GCN %s

@local_memory.local_mem = internal unnamed_addr addrspace(3) global [128 x i32] undef, align 4

; Check that the LDS size emitted correctly
; SI: .long 47180
; SI-NEXT: .long 65668
; CI: .long 47180
; CI-NEXT: .long 32900

; GCN-LABEL: {{^}}local_memory:

; GCN-NOT: s_wqm_b64
; GCN: ds_write_b32

; GCN: s_barrier

; GCN: ds_read_b32 {{v[0-9]+}},
define amdgpu_kernel void @local_memory(i32 addrspace(1)* %out) #0 {
entry:
  %y.i = call i32 @llvm.amdgcn.workitem.id.x() #1
  %arrayidx = getelementptr inbounds [128 x i32], [128 x i32] addrspace(3)* @local_memory.local_mem, i32 0, i32 %y.i
  store i32 %y.i, i32 addrspace(3)* %arrayidx, align 4
  %add = add nsw i32 %y.i, 1
  %cmp = icmp eq i32 %add, 16
  %.add = select i1 %cmp, i32 0, i32 %add
  call void @llvm.amdgcn.s.barrier()
  %arrayidx1 = getelementptr inbounds [128 x i32], [128 x i32] addrspace(3)* @local_memory.local_mem, i32 0, i32 %.add
  %tmp = load i32, i32 addrspace(3)* %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %out, i32 %y.i
  store i32 %tmp, i32 addrspace(1)* %arrayidx2, align 4
  ret void
}

@local_memory_two_objects.local_mem0 = internal unnamed_addr addrspace(3) global [4 x i32] undef, align 4
@local_memory_two_objects.local_mem1 = internal unnamed_addr addrspace(3) global [4 x i32] undef, align 4

; Check that the LDS size emitted correctly
; EG: .long 166120
; EG-NEXT: .long 8
; GCN: .long 47180
; GCN-NEXT: .long 32900

; GCN-LABEL: {{^}}local_memory_two_objects:
; GCN: v_lshlrev_b32_e32 [[ADDRW:v[0-9]+]], 2, v0
; CI-DAG: ds_write2_b32 [[ADDRW]], {{v[0-9]+}}, {{v[0-9]+}} offset1:4

; SI: v_add_i32_e32 [[ADDRW_OFF:v[0-9]+]], vcc, 16, [[ADDRW]]

; SI-DAG: ds_write_b32 [[ADDRW]],
; SI-DAG: ds_write_b32 [[ADDRW_OFF]],

; GCN: s_barrier

; SI-DAG: v_sub_i32_e32 [[SUB0:v[0-9]+]], vcc, 28, [[ADDRW]]
; SI-DAG: v_sub_i32_e32 [[SUB1:v[0-9]+]], vcc, 12, [[ADDRW]]

; SI-DAG: ds_read_b32 v{{[0-9]+}}, [[SUB0]]
; SI-DAG: ds_read_b32 v{{[0-9]+}}, [[SUB1]]

; CI: v_sub_i32_e32 [[SUB:v[0-9]+]], vcc, 0, [[ADDRW]]
; CI: ds_read2_b32 {{v\[[0-9]+:[0-9]+\]}}, [[SUB]] offset0:3 offset1:7
define amdgpu_kernel void @local_memory_two_objects(i32 addrspace(1)* %out) #0 {
entry:
  %x.i = call i32 @llvm.amdgcn.workitem.id.x()
  %arrayidx = getelementptr inbounds [4 x i32], [4 x i32] addrspace(3)* @local_memory_two_objects.local_mem0, i32 0, i32 %x.i
  store i32 %x.i, i32 addrspace(3)* %arrayidx, align 4
  %mul = shl nsw i32 %x.i, 1
  %arrayidx1 = getelementptr inbounds [4 x i32], [4 x i32] addrspace(3)* @local_memory_two_objects.local_mem1, i32 0, i32 %x.i
  store i32 %mul, i32 addrspace(3)* %arrayidx1, align 4
  %sub = sub nsw i32 3, %x.i
  call void @llvm.amdgcn.s.barrier()
  %arrayidx2 = getelementptr inbounds [4 x i32], [4 x i32] addrspace(3)* @local_memory_two_objects.local_mem0, i32 0, i32 %sub
  %tmp = load i32, i32 addrspace(3)* %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds i32, i32 addrspace(1)* %out, i32 %x.i
  store i32 %tmp, i32 addrspace(1)* %arrayidx3, align 4
  %arrayidx4 = getelementptr inbounds [4 x i32], [4 x i32] addrspace(3)* @local_memory_two_objects.local_mem1, i32 0, i32 %sub
  %tmp1 = load i32, i32 addrspace(3)* %arrayidx4, align 4
  %add = add nsw i32 %x.i, 4
  %arrayidx5 = getelementptr inbounds i32, i32 addrspace(1)* %out, i32 %add
  store i32 %tmp1, i32 addrspace(1)* %arrayidx5, align 4
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1
declare void @llvm.amdgcn.s.barrier() #2

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { convergent nounwind }
