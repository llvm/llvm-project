; RUN: llc -mtriple=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefixes=CHECK,NOHSA %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=kaveri -verify-machineinstrs < %s | FileCheck -check-prefixes=CHECK,HSA %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=fiji -verify-machineinstrs < %s | FileCheck -check-prefixes=CHECK,HSA %s

; RUN: llc -global-isel -mtriple=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefixes=CHECK,NOHSA %s
; RUN: llc -global-isel -mtriple=amdgcn--amdhsa -mcpu=kaveri -verify-machineinstrs < %s | FileCheck -check-prefixes=CHECK,HSA %s
; RUN: llc -global-isel -mtriple=amdgcn--amdhsa -mcpu=fiji -verify-machineinstrs < %s | FileCheck -check-prefixes=CHECK,HSA %s

@lds0 = addrspace(3) global [512 x float] undef, align 4
@lds1 = addrspace(3) global [256 x float] undef, align 4

@large = addrspace(3) global [4096 x i32] undef, align 4

; CHECK-LABEL: {{^}}groupstaticsize_test0:
; NOHSA: v_mov_b32_e32 v{{[0-9]+}}, llvm.amdgcn.groupstaticsize@abs32@lo
; HSA: v_mov_b32_e32 v{{[0-9]+}}, 0x800{{$}}
define amdgpu_kernel void @groupstaticsize_test0(ptr addrspace(1) %out, ptr addrspace(1) %lds_size) #0 {
  %tid.x = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %idx.0 = add nsw i32 %tid.x, 64
  %static_lds_size = call i32 @llvm.amdgcn.groupstaticsize() #1
  store i32 %static_lds_size, ptr addrspace(1) %lds_size, align 4
  %arrayidx0 = getelementptr inbounds [512 x float], ptr addrspace(3) @lds0, i32 0, i32 %idx.0
  %val0 = load float, ptr addrspace(3) %arrayidx0, align 4
  store float %val0, ptr addrspace(1) %out, align 4

  ret void
}

; CHECK-LABEL: {{^}}groupstaticsize_test1:
; NOHSA: v_mov_b32_e32 v{{[0-9]+}}, llvm.amdgcn.groupstaticsize@abs32@lo
; HSA: v_mov_b32_e32 v{{[0-9]+}}, 0xc00{{$}}
define amdgpu_kernel void @groupstaticsize_test1(ptr addrspace(1) %out, i32 %cond, ptr addrspace(1) %lds_size) {
entry:
  %static_lds_size = call i32 @llvm.amdgcn.groupstaticsize() #1
  store i32 %static_lds_size, ptr addrspace(1) %lds_size, align 4
  %tid.x = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %idx.0 = add nsw i32 %tid.x, 64
  %tmp = icmp eq i32 %cond, 0
  br i1 %tmp, label %if, label %else

if:                                               ; preds = %entry
  %arrayidx0 = getelementptr inbounds [512 x float], ptr addrspace(3) @lds0, i32 0, i32 %idx.0
  %val0 = load float, ptr addrspace(3) %arrayidx0, align 4
  store float %val0, ptr addrspace(1) %out, align 4
  br label %endif

else:                                             ; preds = %entry
  %arrayidx1 = getelementptr inbounds [256 x float], ptr addrspace(3) @lds1, i32 0, i32 %idx.0
  %val1 = load float, ptr addrspace(3) %arrayidx1, align 4
  store float %val1, ptr addrspace(1) %out, align 4
  br label %endif

endif:                                            ; preds = %else, %if
  ret void
}

; Exceeds 16-bit simm limit of s_movk_i32
; CHECK-LABEL: {{^}}large_groupstaticsize:
; NOHSA: v_mov_b32_e32 v{{[0-9]+}}, llvm.amdgcn.groupstaticsize@abs32@lo
; HSA: v_mov_b32_e32 [[REG:v[0-9]+]], 0x4000{{$}}
define amdgpu_kernel void @large_groupstaticsize(ptr addrspace(1) %size, i32 %idx) #0 {
  %gep = getelementptr inbounds [4096 x i32], ptr addrspace(3) @large, i32 0, i32 %idx
  store volatile i32 0, ptr addrspace(3) %gep
  %static_lds_size = call i32 @llvm.amdgcn.groupstaticsize()
  store i32 %static_lds_size, ptr addrspace(1) %size
  ret void
}

declare i32 @llvm.amdgcn.groupstaticsize() #1
declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
