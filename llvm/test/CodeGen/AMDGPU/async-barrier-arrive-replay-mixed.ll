; NOTE: Do not autogenerate. Check the two load identities and that only the replay load clears EXEC.
; RUN: llc -mtriple=amdgpu12.50-amd-amdhsa -stop-after=si-insert-async-barrier-arrive-workaround < %s | FileCheck %s --check-prefix=MIR
; RUN: llc -mtriple=amdgpu12.51-amd-amdhsa -stop-after=si-insert-async-barrier-arrive-workaround < %s | FileCheck %s --check-prefix=MIR
; RUN: llc -mtriple=amdgpu12.50-amd-amdhsa -global-isel=0 < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=amdgpu12.50-amd-amdhsa -global-isel=1 < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=amdgpu12.51-amd-amdhsa -global-isel=0 < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=amdgpu12.51-amd-amdhsa -global-isel=1 < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=amdgpu12.50-amd-amdhsa -amdgpu-xnack=0 < %s | FileCheck %s --check-prefix=NO-REPLAY
; RUN: llc -mtriple=amdgpu12.51-amd-amdhsa -amdgpu-xnack=0 < %s | FileCheck %s --check-prefix=NO-REPLAY

@lds = internal addrspace(3) global [16 x i8] poison, align 8

; MIR-LABEL: name: mixed_loads
; MIR: DS_ATOMIC_ASYNC_BARRIER_ARRIVE_B64
; MIR-NEXT: GLOBAL_LOAD_ASYNC_TO_LDS_B8_replay
; MIR: GLOBAL_LOAD_ASYNC_TO_LDS_B8{{(_SADDR)?}} {{.*}}

; ASM-LABEL: mixed_loads:
; ASM: ds_atomic_async_barrier_arrive_b64
; ASM: s_mov_b32 [[SAVED:s[0-9]+]], exec_lo
; ASM-NEXT: s_mov_b32 exec_lo, 0
; ASM-NEXT: global_load_async_to_lds_b8 v0, v[0:1], off
; ASM-NEXT: s_mov_b32 exec_lo, [[SAVED]]
; ASM-NOT: s_mov_b32 exec_lo, 0
; ASM: global_load_async_to_lds_b8
; ASM-NOT: global_load_async_to_lds_b8
; ASM: s_endpgm

; NO-REPLAY-LABEL: mixed_loads:
; NO-REPLAY: ds_atomic_async_barrier_arrive_b64
; NO-REPLAY-NOT: s_mov_b32 exec_lo, 0
; NO-REPLAY: global_load_async_to_lds_b8
; NO-REPLAY-NOT: global_load_async_to_lds_b8
; NO-REPLAY: s_endpgm

define amdgpu_kernel void @mixed_loads(ptr addrspace(1) %src) {
  store i64 0, ptr addrspace(3) @lds, align 8
  call void @llvm.amdgcn.ds.atomic.async.barrier.arrive.b64(ptr addrspace(3) @lds)
  %dst = getelementptr [16 x i8], ptr addrspace(3) @lds, i32 0, i32 8
  call void @llvm.amdgcn.global.load.async.to.lds.b8(ptr addrspace(1) %src,
                                                  ptr addrspace(3) %dst,
                                                  i32 0, i32 0)
  ret void
}
