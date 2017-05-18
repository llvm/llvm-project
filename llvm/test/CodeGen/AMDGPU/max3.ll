; RUN: llc -march=amdgcn < %s | FileCheck -check-prefixes=GCN,SI %s
; RUN: llc -march=amdgcn -mcpu=tonga < %s | FileCheck -check-prefixes=GCN,VI %s
; RUN: llc -march=amdgcn -mcpu=gfx900 < %s | FileCheck -check-prefixes=GCN,GFX9 %s

; GCN-LABEL: {{^}}v_test_imax3_sgt_i32:
; GCN: v_max3_i32
define amdgpu_kernel void @v_test_imax3_sgt_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %aptr, i32 addrspace(1)* %bptr, i32 addrspace(1)* %cptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr i32, i32 addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr i32, i32 addrspace(1)* %bptr, i32 %tid
  %gep2 = getelementptr i32, i32 addrspace(1)* %cptr, i32 %tid
  %outgep = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %a = load i32, i32 addrspace(1)* %gep0
  %b = load i32, i32 addrspace(1)* %gep1
  %c = load i32, i32 addrspace(1)* %gep2
  %icmp0 = icmp sgt i32 %a, %b
  %i0 = select i1 %icmp0, i32 %a, i32 %b
  %icmp1 = icmp sgt i32 %i0, %c
  %i1 = select i1 %icmp1, i32 %i0, i32 %c
  store i32 %i1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_test_umax3_ugt_i32:
; GCN: v_max3_u32
define amdgpu_kernel void @v_test_umax3_ugt_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %aptr, i32 addrspace(1)* %bptr, i32 addrspace(1)* %cptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr i32, i32 addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr i32, i32 addrspace(1)* %bptr, i32 %tid
  %gep2 = getelementptr i32, i32 addrspace(1)* %cptr, i32 %tid
  %outgep = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %a = load i32, i32 addrspace(1)* %gep0
  %b = load i32, i32 addrspace(1)* %gep1
  %c = load i32, i32 addrspace(1)* %gep2
  %icmp0 = icmp ugt i32 %a, %b
  %i0 = select i1 %icmp0, i32 %a, i32 %b
  %icmp1 = icmp ugt i32 %i0, %c
  %i1 = select i1 %icmp1, i32 %i0, i32 %c
  store i32 %i1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_test_imax3_sgt_i16:
; SI: v_max3_i32

; VI: v_max_i16
; VI: v_max_i16

; GFX9: v_max3_i16
define amdgpu_kernel void @v_test_imax3_sgt_i16(i16 addrspace(1)* %out, i16 addrspace(1)* %aptr, i16 addrspace(1)* %bptr, i16 addrspace(1)* %cptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr i16, i16 addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr i16, i16 addrspace(1)* %bptr, i32 %tid
  %gep2 = getelementptr i16, i16 addrspace(1)* %cptr, i32 %tid
  %outgep = getelementptr i16, i16 addrspace(1)* %out, i32 %tid
  %a = load i16, i16 addrspace(1)* %gep0
  %b = load i16, i16 addrspace(1)* %gep1
  %c = load i16, i16 addrspace(1)* %gep2
  %icmp0 = icmp sgt i16 %a, %b
  %i0 = select i1 %icmp0, i16 %a, i16 %b
  %icmp1 = icmp sgt i16 %i0, %c
  %i1 = select i1 %icmp1, i16 %i0, i16 %c
  store i16 %i1, i16 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_test_umax3_ugt_i16:
; SI: v_max3_u32

; VI: v_max_u16
; VI: v_max_u16

; GFX9: v_max3_u16
define amdgpu_kernel void @v_test_umax3_ugt_i16(i16 addrspace(1)* %out, i16 addrspace(1)* %aptr, i16 addrspace(1)* %bptr, i16 addrspace(1)* %cptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr i16, i16 addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr i16, i16 addrspace(1)* %bptr, i32 %tid
  %gep2 = getelementptr i16, i16 addrspace(1)* %cptr, i32 %tid
  %outgep = getelementptr i16, i16 addrspace(1)* %out, i32 %tid
  %a = load i16, i16 addrspace(1)* %gep0
  %b = load i16, i16 addrspace(1)* %gep1
  %c = load i16, i16 addrspace(1)* %gep2
  %icmp0 = icmp ugt i16 %a, %b
  %i0 = select i1 %icmp0, i16 %a, i16 %b
  %icmp1 = icmp ugt i16 %i0, %c
  %i1 = select i1 %icmp1, i16 %i0, i16 %c
  store i16 %i1, i16 addrspace(1)* %out
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone speculatable }
