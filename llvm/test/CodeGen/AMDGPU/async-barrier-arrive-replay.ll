; NOTE: Do not autogenerate. Check workaround ordering on affected targets and absence on unaffected targets.
; RUN: llc -global-isel=0 -mtriple=amdgpu12.50-amd-amdhsa < %s | FileCheck --check-prefix=AFFECTED %s
; RUN: llc -global-isel=1 -mtriple=amdgpu12.50-amd-amdhsa < %s | FileCheck --check-prefix=AFFECTED %s
; RUN: llc -global-isel=0 -mtriple=amdgpu12.51-amd-amdhsa < %s | FileCheck --check-prefix=AFFECTED %s
; RUN: llc -global-isel=1 -mtriple=amdgpu12.51-amd-amdhsa < %s | FileCheck --check-prefix=AFFECTED %s
; RUN: llc -global-isel=0 -mtriple=amdgpu13.10-amd-amdhsa -mattr=+lds-barrier-arrive-atomic < %s | FileCheck --check-prefix=UNAFFECTED %s
; RUN: llc -global-isel=0 -mtriple=amdgpu12.50-amd-amdhsa -amdgpu-xnack=0 < %s | FileCheck --check-prefix=UNAFFECTED %s
; RUN: llc -global-isel=0 -mtriple=amdgpu12.50-amd-amdhsa -amdgpu-xnack=1 < %s | FileCheck --check-prefix=AFFECTED %s
; RUN: llc -global-isel=1 -mtriple=amdgpu12.50-amd-amdhsa -amdgpu-xnack=0 < %s | FileCheck --check-prefix=UNAFFECTED %s
; RUN: llc -global-isel=1 -mtriple=amdgpu12.50-amd-amdhsa -amdgpu-xnack=1 < %s | FileCheck --check-prefix=AFFECTED %s
; RUN: llc -global-isel=0 -mtriple=amdgpu12.51-amd-amdhsa -amdgpu-xnack=0 < %s | FileCheck --check-prefix=UNAFFECTED %s
; RUN: llc -global-isel=0 -mtriple=amdgpu12.51-amd-amdhsa -amdgpu-xnack=1 < %s | FileCheck --check-prefix=AFFECTED %s
; RUN: llc -global-isel=1 -mtriple=amdgpu12.51-amd-amdhsa -amdgpu-xnack=0 < %s | FileCheck --check-prefix=UNAFFECTED %s
; RUN: llc -global-isel=1 -mtriple=amdgpu12.51-amd-amdhsa -amdgpu-xnack=1 < %s | FileCheck --check-prefix=AFFECTED %s
; RUN: llc -global-isel=0 -mtriple=amdgpu12.50s-amd-amdhsa -amdgpu-xnack=0 < %s | FileCheck --check-prefix=UNAFFECTED %s
; RUN: llc -global-isel=0 -mtriple=amdgpu12.50s-amd-amdhsa -amdgpu-xnack=1 < %s | FileCheck --check-prefix=AFFECTED %s
; RUN: llc -global-isel=1 -mtriple=amdgpu12.50s-amd-amdhsa -amdgpu-xnack=0 < %s | FileCheck --check-prefix=UNAFFECTED %s
; RUN: llc -global-isel=1 -mtriple=amdgpu12.50s-amd-amdhsa -amdgpu-xnack=1 < %s | FileCheck --check-prefix=AFFECTED %s
; RUN: llc -global-isel=0 -mtriple=amdgpu12.5-amd-amdhsa < %s | FileCheck --check-prefix=AFFECTED %s
; RUN: llc -global-isel=0 -mtriple=amdgpu12.50-amd-amdhsa -mattr=-async-barrier-arrive-replay-bug < %s | FileCheck --check-prefix=UNAFFECTED %s

; RUN: llc -O0 -global-isel=0 -mtriple=amdgpu12.50-amd-amdhsa -verify-machineinstrs < %s | FileCheck --check-prefix=AFFECTED %s
; RUN: llc -O0 -enable-new-pm -global-isel=1 -global-isel-abort=1 -mtriple=amdgpu12.50-amd-amdhsa -verify-machineinstrs < %s | FileCheck --check-prefix=AFFECTED %s

; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 < %s | FileCheck --check-prefix=AFFECTED %s
; RUN: llc -enable-new-pm -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1251 < %s | FileCheck --check-prefix=AFFECTED %s

; RUN: llc -mtriple=amdgpu12.50-amd-amdhsa -mattr=+dumpcode -filetype=obj < %s | llvm-objcopy --dump-section .AMDGPU.disasm=- - /dev/null | FileCheck --check-prefix=DUMP %s

; DUMP: ds_atomic_async_barrier_arrive_b64
; DUMP: s_mov_b32 [[SAVED:s[0-9]+]], exec_lo
; DUMP-NEXT: s_mov_b32 exec_lo, 0
; DUMP-NEXT: global_load_async_to_lds_b8 v0, v[0:1], off {{.*}}; {{[0-9A-F]{8} [0-9A-F]{8} ([0-9A-F]{8})}}
; DUMP-NEXT: s_mov_b32 exec_lo, [[SAVED]]

; Check an LDS barrier arrival followed by a global load.
@bar = internal addrspace(3) global i64 poison, align 8

; AFFECTED-LABEL: replay_window:
; AFFECTED: ds_atomic_async_barrier_arrive_b64
; AFFECTED-NOT: global_load_b32
; AFFECTED: s_mov_b32 [[SAVED:s[0-9]+]], exec_lo
; AFFECTED-NEXT: s_mov_b32 exec_lo, 0
; AFFECTED-NEXT: global_load_async_to_lds_b8
; AFFECTED-NEXT: s_mov_b32 exec_lo, [[SAVED]]
; AFFECTED: global_load_b32
;
; UNAFFECTED-LABEL: replay_window:
; UNAFFECTED: ds_atomic_async_barrier_arrive_b64
; UNAFFECTED-NOT: global_load_async_to_lds
; UNAFFECTED: global_load_b32
define amdgpu_kernel void @replay_window(ptr addrspace(1) %src,
                                         ptr addrspace(1) %dst) {
entry:
  store i64 0, ptr addrspace(3) @bar, align 8
  call void @llvm.amdgcn.ds.atomic.async.barrier.arrive.b64(ptr addrspace(3) @bar)
  %value = load volatile i32, ptr addrspace(1) %src, align 4
  store volatile i32 %value, ptr addrspace(1) %dst, align 4
  ret void
}

; A function can disable the bug feature on an affected module target.
; AFFECTED-LABEL: function_disabled:
; AFFECTED: ds_atomic_async_barrier_arrive_b64
; AFFECTED-NOT: global_load_async_to_lds
; AFFECTED: s_endpgm
; UNAFFECTED-LABEL: function_disabled:
; UNAFFECTED: ds_atomic_async_barrier_arrive_b64
; UNAFFECTED-NOT: global_load_async_to_lds
; UNAFFECTED: s_endpgm
define amdgpu_kernel void @function_disabled() #0 {
  call void @llvm.amdgcn.ds.atomic.async.barrier.arrive.b64(ptr addrspace(3) null)
  ret void
}

attributes #0 = { "target-features"="-async-barrier-arrive-replay-bug" }
