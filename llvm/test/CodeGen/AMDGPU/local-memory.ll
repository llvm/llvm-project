; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=bonaire -verify-machineinstrs < %s | FileCheck -check-prefix=CI -check-prefix=FUNC %s

@local_memory.local_mem = internal unnamed_addr addrspace(3) global [128 x i32] undef, align 4


; Check that the LDS size emitted correctly
; EG: .long 166120
; EG-NEXT: .long 128
; SI: .long 47180
; SI-NEXT: .long 71560
; CI: .long 47180
; CI-NEXT: .long 38792

; FUNC-LABEL: {{^}}local_memory:

; EG: LDS_WRITE
; SI-NOT: s_wqm_b64
; SI: ds_write_b32

; GROUP_BARRIER must be the last instruction in a clause
; EG: GROUP_BARRIER
; EG-NEXT: ALU clause
; SI: s_barrier

; EG: LDS_READ_RET
; SI: ds_read_b32 {{v[0-9]+}},

define void @local_memory(i32 addrspace(1)* %out) {
entry:
  %y.i = call i32 @llvm.r600.read.tidig.x() #0
  %arrayidx = getelementptr inbounds [128 x i32], [128 x i32] addrspace(3)* @local_memory.local_mem, i32 0, i32 %y.i
  store i32 %y.i, i32 addrspace(3)* %arrayidx, align 4
  %add = add nsw i32 %y.i, 1
  %cmp = icmp eq i32 %add, 16
  %.add = select i1 %cmp, i32 0, i32 %add
  call void @llvm.AMDGPU.barrier.local()
  %arrayidx1 = getelementptr inbounds [128 x i32], [128 x i32] addrspace(3)* @local_memory.local_mem, i32 0, i32 %.add
  %0 = load i32, i32 addrspace(3)* %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %out, i32 %y.i
  store i32 %0, i32 addrspace(1)* %arrayidx2, align 4
  ret void
}

declare i32 @llvm.r600.read.tidig.x() #0
declare void @llvm.AMDGPU.barrier.local()

attributes #0 = { readnone }
