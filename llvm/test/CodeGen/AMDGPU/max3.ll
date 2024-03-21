; RUN: llc -mtriple=amdgcn < %s | FileCheck -check-prefixes=GCN,SI %s
; RUN: llc -mtriple=amdgcn -mcpu=tonga < %s | FileCheck -check-prefixes=GCN,VI %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx900 < %s | FileCheck -check-prefixes=GCN,GFX9 %s

; GCN-LABEL: {{^}}v_test_imax3_sgt_i32:
; GCN: v_max3_i32
define amdgpu_kernel void @v_test_imax3_sgt_i32(ptr addrspace(1) %out, ptr addrspace(1) %aptr, ptr addrspace(1) %bptr, ptr addrspace(1) %cptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr i32, ptr addrspace(1) %aptr, i32 %tid
  %gep1 = getelementptr i32, ptr addrspace(1) %bptr, i32 %tid
  %gep2 = getelementptr i32, ptr addrspace(1) %cptr, i32 %tid
  %outgep = getelementptr i32, ptr addrspace(1) %out, i32 %tid
  %a = load i32, ptr addrspace(1) %gep0
  %b = load i32, ptr addrspace(1) %gep1
  %c = load i32, ptr addrspace(1) %gep2
  %icmp0 = icmp sgt i32 %a, %b
  %i0 = select i1 %icmp0, i32 %a, i32 %b
  %icmp1 = icmp sgt i32 %i0, %c
  %i1 = select i1 %icmp1, i32 %i0, i32 %c
  store i32 %i1, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}v_test_umax3_ugt_i32:
; GCN: v_max3_u32
define amdgpu_kernel void @v_test_umax3_ugt_i32(ptr addrspace(1) %out, ptr addrspace(1) %aptr, ptr addrspace(1) %bptr, ptr addrspace(1) %cptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr i32, ptr addrspace(1) %aptr, i32 %tid
  %gep1 = getelementptr i32, ptr addrspace(1) %bptr, i32 %tid
  %gep2 = getelementptr i32, ptr addrspace(1) %cptr, i32 %tid
  %outgep = getelementptr i32, ptr addrspace(1) %out, i32 %tid
  %a = load i32, ptr addrspace(1) %gep0
  %b = load i32, ptr addrspace(1) %gep1
  %c = load i32, ptr addrspace(1) %gep2
  %icmp0 = icmp ugt i32 %a, %b
  %i0 = select i1 %icmp0, i32 %a, i32 %b
  %icmp1 = icmp ugt i32 %i0, %c
  %i1 = select i1 %icmp1, i32 %i0, i32 %c
  store i32 %i1, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}v_test_imax3_sgt_i16:
; SI: v_max3_i32

; VI: v_max_i16
; VI: v_max_i16

; GFX9: v_max3_i16
define amdgpu_kernel void @v_test_imax3_sgt_i16(ptr addrspace(1) %out, ptr addrspace(1) %aptr, ptr addrspace(1) %bptr, ptr addrspace(1) %cptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr i16, ptr addrspace(1) %aptr, i32 %tid
  %gep1 = getelementptr i16, ptr addrspace(1) %bptr, i32 %tid
  %gep2 = getelementptr i16, ptr addrspace(1) %cptr, i32 %tid
  %outgep = getelementptr i16, ptr addrspace(1) %out, i32 %tid
  %a = load i16, ptr addrspace(1) %gep0
  %b = load i16, ptr addrspace(1) %gep1
  %c = load i16, ptr addrspace(1) %gep2
  %icmp0 = icmp sgt i16 %a, %b
  %i0 = select i1 %icmp0, i16 %a, i16 %b
  %icmp1 = icmp sgt i16 %i0, %c
  %i1 = select i1 %icmp1, i16 %i0, i16 %c
  store i16 %i1, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}v_test_umax3_ugt_i16:
; SI: v_max3_u32

; VI: v_max_u16
; VI: v_max_u16

; GFX9: v_max3_u16
define amdgpu_kernel void @v_test_umax3_ugt_i16(ptr addrspace(1) %out, ptr addrspace(1) %aptr, ptr addrspace(1) %bptr, ptr addrspace(1) %cptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr i16, ptr addrspace(1) %aptr, i32 %tid
  %gep1 = getelementptr i16, ptr addrspace(1) %bptr, i32 %tid
  %gep2 = getelementptr i16, ptr addrspace(1) %cptr, i32 %tid
  %outgep = getelementptr i16, ptr addrspace(1) %out, i32 %tid
  %a = load i16, ptr addrspace(1) %gep0
  %b = load i16, ptr addrspace(1) %gep1
  %c = load i16, ptr addrspace(1) %gep2
  %icmp0 = icmp ugt i16 %a, %b
  %i0 = select i1 %icmp0, i16 %a, i16 %b
  %icmp1 = icmp ugt i16 %i0, %c
  %i1 = select i1 %icmp1, i16 %i0, i16 %c
  store i16 %i1, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}v_test_imax3_sgt_i8:
; SI: v_max3_i32

; VI: v_max_i16
; VI: v_max_i16

; GFX9: v_max3_i16
define amdgpu_kernel void @v_test_imax3_sgt_i8(ptr addrspace(1) %out, ptr addrspace(1) %aptr, ptr addrspace(1) %bptr, ptr addrspace(1) %cptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr i8, ptr addrspace(1) %aptr, i32 %tid
  %gep1 = getelementptr i8, ptr addrspace(1) %bptr, i32 %tid
  %gep2 = getelementptr i8, ptr addrspace(1) %cptr, i32 %tid
  %outgep = getelementptr i8, ptr addrspace(1) %out, i32 %tid
  %a = load i8, ptr addrspace(1) %gep0
  %b = load i8, ptr addrspace(1) %gep1
  %c = load i8, ptr addrspace(1) %gep2
  %icmp0 = icmp sgt i8 %a, %b
  %i0 = select i1 %icmp0, i8 %a, i8 %b
  %icmp1 = icmp sgt i8 %i0, %c
  %i1 = select i1 %icmp1, i8 %i0, i8 %c
  store i8 %i1, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}v_test_umax3_ugt_i8:
; SI: v_max3_u32

; VI: v_max_u16
; VI: v_max_u16

; GFX9: v_max3_u16
define amdgpu_kernel void @v_test_umax3_ugt_i8(ptr addrspace(1) %out, ptr addrspace(1) %aptr, ptr addrspace(1) %bptr, ptr addrspace(1) %cptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr i8, ptr addrspace(1) %aptr, i32 %tid
  %gep1 = getelementptr i8, ptr addrspace(1) %bptr, i32 %tid
  %gep2 = getelementptr i8, ptr addrspace(1) %cptr, i32 %tid
  %outgep = getelementptr i8, ptr addrspace(1) %out, i32 %tid
  %a = load i8, ptr addrspace(1) %gep0
  %b = load i8, ptr addrspace(1) %gep1
  %c = load i8, ptr addrspace(1) %gep2
  %icmp0 = icmp ugt i8 %a, %b
  %i0 = select i1 %icmp0, i8 %a, i8 %b
  %icmp1 = icmp ugt i8 %i0, %c
  %i1 = select i1 %icmp1, i8 %i0, i8 %c
  store i8 %i1, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}v_test_imax3_sgt_i7:
; SI: v_max3_i32

; VI: v_max_i16
; VI: v_max_i16

; GFX9: v_max3_i16
define amdgpu_kernel void @v_test_imax3_sgt_i7(ptr addrspace(1) %out, ptr addrspace(1) %aptr, ptr addrspace(1) %bptr, ptr addrspace(1) %cptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr i7, ptr addrspace(1) %aptr, i32 %tid
  %gep1 = getelementptr i7, ptr addrspace(1) %bptr, i32 %tid
  %gep2 = getelementptr i7, ptr addrspace(1) %cptr, i32 %tid
  %outgep = getelementptr i7, ptr addrspace(1) %out, i32 %tid
  %a = load i7, ptr addrspace(1) %gep0
  %b = load i7, ptr addrspace(1) %gep1
  %c = load i7, ptr addrspace(1) %gep2
  %icmp0 = icmp sgt i7 %a, %b
  %i0 = select i1 %icmp0, i7 %a, i7 %b
  %icmp1 = icmp sgt i7 %i0, %c
  %i1 = select i1 %icmp1, i7 %i0, i7 %c
  store i7 %i1, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}v_test_umax3_ugt_i7:
; SI: v_max3_u32

; VI: v_max_u16
; VI: v_max_u16

; GFX9: v_max3_u16
define amdgpu_kernel void @v_test_umax3_ugt_i7(ptr addrspace(1) %out, ptr addrspace(1) %aptr, ptr addrspace(1) %bptr, ptr addrspace(1) %cptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr i7, ptr addrspace(1) %aptr, i32 %tid
  %gep1 = getelementptr i7, ptr addrspace(1) %bptr, i32 %tid
  %gep2 = getelementptr i7, ptr addrspace(1) %cptr, i32 %tid
  %outgep = getelementptr i7, ptr addrspace(1) %out, i32 %tid
  %a = load i7, ptr addrspace(1) %gep0
  %b = load i7, ptr addrspace(1) %gep1
  %c = load i7, ptr addrspace(1) %gep2
  %icmp0 = icmp ugt i7 %a, %b
  %i0 = select i1 %icmp0, i7 %a, i7 %b
  %icmp1 = icmp ugt i7 %i0, %c
  %i1 = select i1 %icmp1, i7 %i0, i7 %c
  store i7 %i1, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}v_test_imax3_sgt_i33:
; GCN-NOT: v_max3
define amdgpu_kernel void @v_test_imax3_sgt_i33(ptr addrspace(1) %out, ptr addrspace(1) %aptr, ptr addrspace(1) %bptr, ptr addrspace(1) %cptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr i33, ptr addrspace(1) %aptr, i32 %tid
  %gep1 = getelementptr i33, ptr addrspace(1) %bptr, i32 %tid
  %gep2 = getelementptr i33, ptr addrspace(1) %cptr, i32 %tid
  %outgep = getelementptr i33, ptr addrspace(1) %out, i32 %tid
  %a = load i33, ptr addrspace(1) %gep0
  %b = load i33, ptr addrspace(1) %gep1
  %c = load i33, ptr addrspace(1) %gep2
  %icmp0 = icmp sgt i33 %a, %b
  %i0 = select i1 %icmp0, i33 %a, i33 %b
  %icmp1 = icmp sgt i33 %i0, %c
  %i1 = select i1 %icmp1, i33 %i0, i33 %c
  store i33 %i1, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}v_test_umax3_ugt_i33:
; GCN-NOT: v_max3
define amdgpu_kernel void @v_test_umax3_ugt_i33(ptr addrspace(1) %out, ptr addrspace(1) %aptr, ptr addrspace(1) %bptr, ptr addrspace(1) %cptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr i33, ptr addrspace(1) %aptr, i32 %tid
  %gep1 = getelementptr i33, ptr addrspace(1) %bptr, i32 %tid
  %gep2 = getelementptr i33, ptr addrspace(1) %cptr, i32 %tid
  %outgep = getelementptr i33, ptr addrspace(1) %out, i32 %tid
  %a = load i33, ptr addrspace(1) %gep0
  %b = load i33, ptr addrspace(1) %gep1
  %c = load i33, ptr addrspace(1) %gep2
  %icmp0 = icmp ugt i33 %a, %b
  %i0 = select i1 %icmp0, i33 %a, i33 %b
  %icmp1 = icmp ugt i33 %i0, %c
  %i1 = select i1 %icmp1, i33 %i0, i33 %c
  store i33 %i1, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}v_test_imax3_sgt_i64:
; GCN-NOT: v_max3
define amdgpu_kernel void @v_test_imax3_sgt_i64(ptr addrspace(1) %out, ptr addrspace(1) %aptr, ptr addrspace(1) %bptr, ptr addrspace(1) %cptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr i64, ptr addrspace(1) %aptr, i32 %tid
  %gep1 = getelementptr i64, ptr addrspace(1) %bptr, i32 %tid
  %gep2 = getelementptr i64, ptr addrspace(1) %cptr, i32 %tid
  %outgep = getelementptr i64, ptr addrspace(1) %out, i32 %tid
  %a = load i64, ptr addrspace(1) %gep0
  %b = load i64, ptr addrspace(1) %gep1
  %c = load i64, ptr addrspace(1) %gep2
  %icmp0 = icmp sgt i64 %a, %b
  %i0 = select i1 %icmp0, i64 %a, i64 %b
  %icmp1 = icmp sgt i64 %i0, %c
  %i1 = select i1 %icmp1, i64 %i0, i64 %c
  store i64 %i1, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}v_test_umax3_ugt_i64:
; GCN-NOT: v_max3
define amdgpu_kernel void @v_test_umax3_ugt_i64(ptr addrspace(1) %out, ptr addrspace(1) %aptr, ptr addrspace(1) %bptr, ptr addrspace(1) %cptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr i64, ptr addrspace(1) %aptr, i32 %tid
  %gep1 = getelementptr i64, ptr addrspace(1) %bptr, i32 %tid
  %gep2 = getelementptr i64, ptr addrspace(1) %cptr, i32 %tid
  %outgep = getelementptr i64, ptr addrspace(1) %out, i32 %tid
  %a = load i64, ptr addrspace(1) %gep0
  %b = load i64, ptr addrspace(1) %gep1
  %c = load i64, ptr addrspace(1) %gep2
  %icmp0 = icmp ugt i64 %a, %b
  %i0 = select i1 %icmp0, i64 %a, i64 %b
  %icmp1 = icmp ugt i64 %i0, %c
  %i1 = select i1 %icmp1, i64 %i0, i64 %c
  store i64 %i1, ptr addrspace(1) %out
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone speculatable }
