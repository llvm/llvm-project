; RUN: llc -verify-machineinstrs -global-isel=1 -global-isel-abort=2 -mtriple=amdgcn--amdhsa -mcpu=gfx90a -stop-after=irtranslator -o - < %s | FileCheck -check-prefix=MIR %s
; RUN: llc -verify-machineinstrs -global-isel=1 -global-isel-abort=2 -mtriple=amdgcn--amdhsa -mcpu=gfx90a -o - < %s | FileCheck -check-prefix=ASM %s
; RUN: llc -verify-machineinstrs -O1 -global-isel=1 -global-isel-abort=2 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a -o - < %s | FileCheck -check-prefix=ISSUE %s

define amdgpu_kernel void @explicit_i32_inreg(i32 inreg %x,
                                              ptr addrspace(1) %out) #1 {
; MIR-LABEL: name: explicit_i32_inreg
; MIR: firstKernArgPreloadReg
; MIR: numKernargPreloadSGPRs: 1
; MIR: body:
; MIR: [[X:%[0-9]+]]:_(s32) = COPY %{{[0-9]+}}
; MIR-NOT: [[X]]:_(s32) = G_LOAD
; ASM-LABEL: explicit_i32_inreg:
; ASM: .amdhsa_user_sgpr_kernarg_preload_length 1
  store i32 %x, ptr addrspace(1) %out, align 4
  ret void
}

define amdgpu_kernel void @explicit_i64_inreg(i64 inreg %x,
                                              ptr addrspace(1) %out) #1 {
; MIR-LABEL: name: explicit_i64_inreg
; MIR: firstKernArgPreloadReg
; MIR: numKernargPreloadSGPRs: 2
; MIR: body:
; MIR: [[X:%[0-9]+]]:_(s64) = COPY %{{[0-9]+}}
; MIR-NOT: [[X]]:_(s64) = G_LOAD
; ASM-LABEL: explicit_i64_inreg:
; ASM: .amdhsa_user_sgpr_kernarg_preload_length 2
  store i64 %x, ptr addrspace(1) %out, align 8
  ret void
}

define amdgpu_kernel void @explicit_ptr_inreg(ptr addrspace(1) inreg %p,
                                              ptr addrspace(1) %out) #1 {
; MIR-LABEL: name: explicit_ptr_inreg
; MIR: firstKernArgPreloadReg
; MIR: numKernargPreloadSGPRs: 2
; MIR: body:
; MIR: [[PINT:%[0-9]+]]:_(s64) = COPY %{{[0-9]+}}
; MIR: [[P:%[0-9]+]]:_(p1) = G_INTTOPTR [[PINT]](s64)
; MIR-NOT: [[P]]:_(p1) = G_LOAD
; ASM-LABEL: explicit_ptr_inreg:
; ASM: .amdhsa_user_sgpr_kernarg_preload_length 2
  store ptr addrspace(1) %p, ptr addrspace(1) %out, align 8
  ret void
}

define amdgpu_kernel void @packed_i16_inreg(i16 inreg %a, i16 inreg %b,
                                            ptr addrspace(1) %out) #1 {
; MIR-LABEL: name: packed_i16_inreg
; MIR: firstKernArgPreloadReg
; MIR: numKernargPreloadSGPRs: 1
; MIR: body:
; MIR: [[PACKED:%[0-9]+]]:_(s32) = COPY %{{[0-9]+}}
; MIR: [[SHIFT:%[0-9]+]]:_(s32) = G_LSHR [[PACKED]], %{{[0-9]+}}(s32)
; MIR: [[B:%[0-9]+]]:_(s16) = G_TRUNC [[SHIFT]](s32)
; MIR-NOT: [[B]]:_(s16) = G_LOAD
; ASM-LABEL: packed_i16_inreg:
; ASM: .amdhsa_user_sgpr_kernarg_preload_length 1
  store i16 %b, ptr addrspace(1) %out, align 2
  ret void
}

define amdgpu_kernel void @preload_block_count_x(
    ptr addrspace(1) inreg %out,
    i32 inreg "amdgpu-hidden-argument" %_hidden_block_count_x) #0 {
; MIR-LABEL: name: preload_block_count_x
; MIR: firstKernArgPreloadReg
; MIR: numKernargPreloadSGPRs: 3
; MIR: body:
; MIR: {{%[0-9]+}}:_(p1) = G_INTTOPTR {{%[0-9]+}}(s64)
; MIR: {{%[0-9]+}}:_(s32) = COPY %{{[0-9]+}}
; ASM-LABEL: preload_block_count_x:
; ASM: .amdhsa_user_sgpr_kernarg_preload_length 3
  store i32 %_hidden_block_count_x, ptr addrspace(1) %out, align 4
  ret void
}

define amdgpu_kernel void @_Z15hipsmith_kernelv() local_unnamed_addr #0 {
entry:
; ISSUE-LABEL: _Z15hipsmith_kernelv:
; ISSUE: .amdhsa_user_sgpr_kernarg_preload_length 1
  %0 = tail call dereferenceable(256) ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %1 = load i32, ptr addrspace(4) %0, align 4
  %.not.i.not = icmp eq i32 %1, 0
  br i1 %.not.i.not, label %lor.end.i, label %common.ret1

common.ret1:
  ret void

lor.end.i:
  store ptr null, ptr inttoptr (i64 32 to ptr), align 32
  br label %common.ret1
}

declare noundef align 4 ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr() #2

attributes #0 = { "amdgpu-agpr-alloc"="0" "amdgpu-no-cluster-id-x" "amdgpu-no-cluster-id-y" "amdgpu-no-cluster-id-z" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-no-wwm" }
attributes #1 = { "target-cpu"="gfx90a" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
