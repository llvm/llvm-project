; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}test_barrier:
; GCN: buffer_store_dword
; GCN: s_waitcnt
; GCN: s_barrier
define void @test_barrier(i32 addrspace(1)* %out) #0 {
entry:
  %tmp = call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = getelementptr i32, i32 addrspace(1)* %out, i32 %tmp
  store i32 %tmp, i32 addrspace(1)* %tmp1
  call void @llvm.amdgcn.s.barrier()
  %tmp2 = call i32 @llvm.r600.read.local.size.x()
  %tmp3 = sub i32 %tmp2, 1
  %tmp4 = sub i32 %tmp3, %tmp
  %tmp5 = getelementptr i32, i32 addrspace(1)* %out, i32 %tmp4
  %tmp6 = load i32, i32 addrspace(1)* %tmp5
  store i32 %tmp6, i32 addrspace(1)* %tmp1
  ret void
}

declare void @llvm.amdgcn.s.barrier() #1
declare i32 @llvm.amdgcn.workitem.id.x() #2
declare i32 @llvm.r600.read.local.size.x() #2

attributes #0 = { nounwind }
attributes #1 = { convergent nounwind }
attributes #2 = { nounwind readnone }
