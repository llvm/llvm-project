; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=EG  %s

; EG-LABEL: {{^}}test_group_barrier:
; EG: GROUP_BARRIER
define amdgpu_kernel void @test_group_barrier(ptr addrspace(1) %out) #0 {
entry:
  %tmp = call i32 @llvm.r600.read.tidig.x()
  %tmp1 = getelementptr i32, ptr addrspace(1) %out, i32 %tmp
  store i32 %tmp, ptr addrspace(1) %tmp1
  call void @llvm.r600.group.barrier()
  %tmp2 = call i32 @llvm.r600.read.local.size.x()
  %tmp3 = sub i32 %tmp2, 1
  %tmp4 = sub i32 %tmp3, %tmp
  %tmp5 = getelementptr i32, ptr addrspace(1) %out, i32 %tmp4
  %tmp6 = load i32, ptr addrspace(1) %tmp5
  store i32 %tmp6, ptr addrspace(1) %tmp1
  ret void
}

; Function Attrs: convergent nounwind
declare void @llvm.r600.group.barrier() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.r600.read.tidig.x() #2

; Function Attrs: nounwind readnone
declare i32 @llvm.r600.read.local.size.x() #2

attributes #0 = { nounwind }
attributes #1 = { convergent nounwind }
attributes #2 = { nounwind readnone }
