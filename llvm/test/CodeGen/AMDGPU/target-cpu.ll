; RUN: llc -march=amdgcn -disable-promote-alloca-to-vector -verify-machineinstrs < %s | FileCheck %s

declare i8 addrspace(4)* @llvm.amdgcn.kernarg.segment.ptr() #1

declare i32 @llvm.amdgcn.workitem.id.x() #1

; CI+ intrinsic
declare void @llvm.amdgcn.s.dcache.inv.vol() #0

; VI+ intrinsic
declare void @llvm.amdgcn.s.dcache.wb() #0

; CHECK-LABEL: {{^}}target_none:
; CHECK: s_movk_i32 [[OFFSETREG:s[0-9]+]], 0x400
; CHECK: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, [[OFFSETREG]]
; CHECK: buffer_store_dword v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64
define amdgpu_kernel void @target_none() #0 {
  %kernargs = call i8 addrspace(4)* @llvm.amdgcn.kernarg.segment.ptr()
  %kernargs.gep = getelementptr inbounds i8, i8 addrspace(4)* %kernargs, i64 1024
  %kernargs.gep.cast = bitcast i8 addrspace(4)* %kernargs.gep to i32 addrspace(1)* addrspace(4)*
  %ptr = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(4)* %kernargs.gep.cast
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %id.ext = sext i32 %id to i64
  %gep = getelementptr inbounds i32, i32 addrspace(1)* %ptr, i64 %id.ext
  store i32 0, i32 addrspace(1)* %gep
  ret void
}

; CHECK-LABEL: {{^}}target_tahiti:
; CHECK: s_movk_i32 [[OFFSETREG:s[0-9]+]], 0x400
; CHECK: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, [[OFFSETREG]]
; CHECK: buffer_store_dword v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64
define amdgpu_kernel void @target_tahiti() #1 {
  %kernargs = call i8 addrspace(4)* @llvm.amdgcn.kernarg.segment.ptr()
  %kernargs.gep = getelementptr inbounds i8, i8 addrspace(4)* %kernargs, i64 1024
  %kernargs.gep.cast = bitcast i8 addrspace(4)* %kernargs.gep to i32 addrspace(1)* addrspace(4)*
  %ptr = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(4)* %kernargs.gep.cast
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %id.ext = sext i32 %id to i64
  %gep = getelementptr inbounds i32, i32 addrspace(1)* %ptr, i64 %id.ext
  store i32 0, i32 addrspace(1)* %gep
  ret void
}

; CHECK-LABEL: {{^}}target_bonaire:
; CHECK: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0x100
; CHECK: buffer_store_dword v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64
; CHECK: s_dcache_inv_vol
define amdgpu_kernel void @target_bonaire() #3 {
  %kernargs = call i8 addrspace(4)* @llvm.amdgcn.kernarg.segment.ptr()
  %kernargs.gep = getelementptr inbounds i8, i8 addrspace(4)* %kernargs, i64 1024
  %kernargs.gep.cast = bitcast i8 addrspace(4)* %kernargs.gep to i32 addrspace(1)* addrspace(4)*
  %ptr = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(4)* %kernargs.gep.cast
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %id.ext = sext i32 %id to i64
  %gep = getelementptr inbounds i32, i32 addrspace(1)* %ptr, i64 %id.ext
  store i32 0, i32 addrspace(1)* %gep
  call void @llvm.amdgcn.s.dcache.inv.vol()
  ret void
}

; CHECK-LABEL: {{^}}target_fiji:
; CHECK: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0x400
; CHECK: flat_store_dword
; CHECK: s_dcache_wb{{$}}
define amdgpu_kernel void @target_fiji() #4 {
  %kernargs = call i8 addrspace(4)* @llvm.amdgcn.kernarg.segment.ptr()
  %kernargs.gep = getelementptr inbounds i8, i8 addrspace(4)* %kernargs, i64 1024
  %kernargs.gep.cast = bitcast i8 addrspace(4)* %kernargs.gep to i32 addrspace(1)* addrspace(4)*
  %ptr = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(4)* %kernargs.gep.cast
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %id.ext = sext i32 %id to i64
  %gep = getelementptr inbounds i32, i32 addrspace(1)* %ptr, i64 %id.ext
  store i32 0, i32 addrspace(1)* %gep
  call void @llvm.amdgcn.s.dcache.wb()
  ret void
}

; CHECK-LABEL: {{^}}promote_alloca_enabled:
; CHECK: ds_read_b32
define amdgpu_kernel void @promote_alloca_enabled(i32 addrspace(1)* nocapture %out, i32 addrspace(1)* nocapture %in) #5 {
entry:
  %stack = alloca [5 x i32], align 4, addrspace(5)
  %tmp = load i32, i32 addrspace(1)* %in, align 4
  %arrayidx1 = getelementptr inbounds [5 x i32], [5 x i32] addrspace(5)* %stack, i32 0, i32 %tmp
  %load = load i32, i32 addrspace(5)* %arrayidx1
  store i32 %load, i32 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}promote_alloca_disabled:
; CHECK: SCRATCH_RSRC_DWORD0
; CHECK: SCRATCH_RSRC_DWORD1
; CHECK: ScratchSize: 24
define amdgpu_kernel void @promote_alloca_disabled(i32 addrspace(1)* nocapture %out, i32 addrspace(1)* nocapture %in) #6 {
entry:
  %stack = alloca [5 x i32], align 4, addrspace(5)
  %tmp = load i32, i32 addrspace(1)* %in, align 4
  %arrayidx1 = getelementptr inbounds [5 x i32], [5 x i32] addrspace(5)* %stack, i32 0, i32 %tmp
  %load = load i32, i32 addrspace(5)* %arrayidx1
  store i32 %load, i32 addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind "target-cpu"="tahiti" }
attributes #3 = { nounwind "target-cpu"="bonaire" }
attributes #4 = { nounwind "target-cpu"="fiji" }
attributes #5 = { nounwind "target-features"="+promote-alloca" "amdgpu-waves-per-eu"="1,3" "amdgpu-flat-work-group-size"="1,256" }
attributes #6 = { nounwind "target-features"="-promote-alloca" "amdgpu-waves-per-eu"="1,3" "amdgpu-flat-work-group-size"="1,256" }
