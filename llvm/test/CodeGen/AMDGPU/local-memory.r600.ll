; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

@local_memory.local_mem = internal unnamed_addr addrspace(3) global [128 x i32] undef, align 4

; Check that the LDS size emitted correctly
; EG: .long 166120
; EG-NEXT: .long 128

; FUNC-LABEL: {{^}}local_memory:

; EG: LDS_WRITE

; GROUP_BARRIER must be the last instruction in a clause
; EG: GROUP_BARRIER
; EG-NEXT: ALU clause

; EG: LDS_READ_RET
define void @local_memory(i32 addrspace(1)* %out) #0 {
entry:
  %y.i = call i32 @llvm.r600.read.tidig.x() #1
  %arrayidx = getelementptr inbounds [128 x i32], [128 x i32] addrspace(3)* @local_memory.local_mem, i32 0, i32 %y.i
  store i32 %y.i, i32 addrspace(3)* %arrayidx, align 4
  %add = add nsw i32 %y.i, 1
  %cmp = icmp eq i32 %add, 16
  %.add = select i1 %cmp, i32 0, i32 %add
  call void @llvm.r600.group.barrier()
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

define void @local_memory_two_objects(i32 addrspace(1)* %out) #0 {
entry:
  %x.i = call i32 @llvm.r600.read.tidig.x() #1
  %arrayidx = getelementptr inbounds [4 x i32], [4 x i32] addrspace(3)* @local_memory_two_objects.local_mem0, i32 0, i32 %x.i
  store i32 %x.i, i32 addrspace(3)* %arrayidx, align 4
  %mul = shl nsw i32 %x.i, 1
  %arrayidx1 = getelementptr inbounds [4 x i32], [4 x i32] addrspace(3)* @local_memory_two_objects.local_mem1, i32 0, i32 %x.i
  store i32 %mul, i32 addrspace(3)* %arrayidx1, align 4
  %sub = sub nsw i32 3, %x.i
  call void @llvm.r600.group.barrier()
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

declare i32 @llvm.r600.read.tidig.x() #1
declare void @llvm.r600.group.barrier() #2

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { convergent nounwind }
