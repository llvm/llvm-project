; RUN: llc -march=amdgcn -mcpu=bonaire -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=CI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

@local_memory_two_objects.local_mem0 = internal unnamed_addr addrspace(3) global [4 x i32] undef, align 4
@local_memory_two_objects.local_mem1 = internal unnamed_addr addrspace(3) global [4 x i32] undef, align 4


; Check that the LDS size emitted correctly
; EG: .long 166120
; EG-NEXT: .long 8
; GCN: .long 47180
; GCN-NEXT: .long 32900


; FUNC-LABEL: {{^}}local_memory_two_objects:

; We would like to check the lds writes are using different
; addresses, but due to variations in the scheduler, we can't do
; this consistently on evergreen GPUs.
; EG: LDS_WRITE
; EG: LDS_WRITE

; GROUP_BARRIER must be the last instruction in a clause
; EG: GROUP_BARRIER
; EG-NEXT: ALU clause

; Make sure the lds reads are using different addresses, at different
; constant offsets.
; EG: LDS_READ_RET {{[*]*}} OQAP, {{PV|T}}[[ADDRR:[0-9]*\.[XYZW]]]
; EG-NOT: LDS_READ_RET {{[*]*}} OQAP, T[[ADDRR]]


; GCN: v_lshlrev_b32_e32 [[ADDRW:v[0-9]+]], 2, v0
; CI-DAG: ds_write_b32 [[ADDRW]], {{v[0-9]*}} offset:16
; CI-DAG: ds_write_b32 [[ADDRW]], {{v[0-9]*$}}


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

define void @local_memory_two_objects(i32 addrspace(1)* %out) {
entry:
  %x.i = call i32 @llvm.r600.read.tidig.x() #0
  %arrayidx = getelementptr inbounds [4 x i32], [4 x i32] addrspace(3)* @local_memory_two_objects.local_mem0, i32 0, i32 %x.i
  store i32 %x.i, i32 addrspace(3)* %arrayidx, align 4
  %mul = shl nsw i32 %x.i, 1
  %arrayidx1 = getelementptr inbounds [4 x i32], [4 x i32] addrspace(3)* @local_memory_two_objects.local_mem1, i32 0, i32 %x.i
  store i32 %mul, i32 addrspace(3)* %arrayidx1, align 4
  %sub = sub nsw i32 3, %x.i
  call void @llvm.AMDGPU.barrier.local()
  %arrayidx2 = getelementptr inbounds [4 x i32], [4 x i32] addrspace(3)* @local_memory_two_objects.local_mem0, i32 0, i32 %sub
  %0 = load i32, i32 addrspace(3)* %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds i32, i32 addrspace(1)* %out, i32 %x.i
  store i32 %0, i32 addrspace(1)* %arrayidx3, align 4
  %arrayidx4 = getelementptr inbounds [4 x i32], [4 x i32] addrspace(3)* @local_memory_two_objects.local_mem1, i32 0, i32 %sub
  %1 = load i32, i32 addrspace(3)* %arrayidx4, align 4
  %add = add nsw i32 %x.i, 4
  %arrayidx5 = getelementptr inbounds i32, i32 addrspace(1)* %out, i32 %add
  store i32 %1, i32 addrspace(1)* %arrayidx5, align 4
  ret void
}

declare i32 @llvm.r600.read.tidig.x() #0
declare void @llvm.AMDGPU.barrier.local()

attributes #0 = { readnone }
