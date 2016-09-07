; RUN: llc -verify-machineinstrs -march=amdgcn -mattr=-promote-alloca < %s | FileCheck -check-prefix=SI-ALLOCA -check-prefix=SI %s
; RUN: llc -verify-machineinstrs -march=amdgcn -mattr=+promote-alloca < %s | FileCheck -check-prefix=SI-PROMOTE -check-prefix=SI %s

declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #1
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #1
declare void @llvm.amdgcn.s.barrier() #2

; The required pointer calculations for the alloca'd actually requires
; an add and won't be folded into the addressing, which fails with a
; 64-bit pointer add. This should work since private pointers should
; be 32-bits.

; SI-LABEL: {{^}}test_private_array_ptr_calc:

; FIXME: We end up with zero argument for ADD, because
; SIRegisterInfo::eliminateFrameIndex() blindly replaces the frame index
; with the appropriate offset.  We should fold this into the store.

; SI-ALLOCA: v_add_i32_e32 [[PTRREG:v[0-9]+]], vcc, 0, v{{[0-9]+}}
; SI-ALLOCA: buffer_store_dword {{v[0-9]+}}, [[PTRREG]], s[{{[0-9]+:[0-9]+}}], s{{[0-9]+}} offen offset:64
; SI-ALLOCA: s_barrier
; SI-ALLOCA: buffer_load_dword {{v[0-9]+}}, [[PTRREG]], s[{{[0-9]+:[0-9]+}}], s{{[0-9]+}} offen offset:64
;
; FIXME: The AMDGPUPromoteAlloca pass should be able to convert this
; alloca to a vector.  It currently fails because it does not know how
; to interpret:
; getelementptr inbounds [16 x i32], [16 x i32]* %alloca, i32 1, i32 %b

; SI-PROMOTE: v_add_i32_e32 [[PTRREG:v[0-9]+]], vcc, 64
; SI-PROMOTE: ds_write_b32 [[PTRREG]]
define void @test_private_array_ptr_calc(i32 addrspace(1)* noalias %out, i32 addrspace(1)* noalias %inA, i32 addrspace(1)* noalias %inB) #0 {
  %alloca = alloca [16 x i32], align 16
  %mbcnt.lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0);
  %tid = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %mbcnt.lo)
  %a_ptr = getelementptr inbounds i32, i32 addrspace(1)* %inA, i32 %tid
  %b_ptr = getelementptr inbounds i32, i32 addrspace(1)* %inB, i32 %tid
  %a = load i32, i32 addrspace(1)* %a_ptr
  %b = load i32, i32 addrspace(1)* %b_ptr
  %result = add i32 %a, %b
  %alloca_ptr = getelementptr inbounds [16 x i32], [16 x i32]* %alloca, i32 1, i32 %b
  store i32 %result, i32* %alloca_ptr, align 4
  ; Dummy call
  call void @llvm.amdgcn.s.barrier()
  %reload = load i32, i32* %alloca_ptr, align 4
  %out_ptr = getelementptr inbounds i32, i32 addrspace(1)* %out, i32 %tid
  store i32 %reload, i32 addrspace(1)* %out_ptr, align 4
  ret void
}

attributes #0 = { nounwind "amdgpu-waves-per-eu"="1,1" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind convergent }
