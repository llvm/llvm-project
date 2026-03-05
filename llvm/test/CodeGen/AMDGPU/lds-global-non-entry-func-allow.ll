; RUN: llc -global-isel=0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -o - \
; RUN:   -amdgpu-enable-lower-module-lds=false \
; RUN:   -amdgpu-allow-lds-in-non-entry-functions %s 2> %t \
; RUN:   | FileCheck -check-prefix=SDAG %s
; RUN: FileCheck -check-prefix=NOERR --allow-empty %s < %t
;
; RUN: llc -global-isel=1 -new-reg-bank-select -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -o - \
; RUN:   -amdgpu-enable-lower-module-lds=false \
; RUN:   -amdgpu-allow-lds-in-non-entry-functions %s 2> %t \
; RUN:   | FileCheck -check-prefix=GISEL %s
; RUN: FileCheck -check-prefix=NOERR --allow-empty %s < %t

; Verify that -amdgpu-allow-lds-in-non-entry-functions suppresses the
; "local memory global used by non-kernel function" diagnostic and allows
; normal LDS allocation in non-entry-point functions.

; NOERR-NOT: warning
; NOERR-NOT: local memory global used by non-kernel function

@lds = internal addrspace(3) global float poison, align 4

define void @func_use_lds_global() {
; SDAG-LABEL: func_use_lds_global:
; SDAG:       ; %bb.0:
; SDAG-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; SDAG-NEXT:    v_mov_b32_e32 v0, 0
; SDAG-NEXT:    ds_write_b32 v0, v0
; SDAG-NEXT:    s_waitcnt lgkmcnt(0)
; SDAG-NEXT:    s_setpc_b64 s[30:31]
;
; GISEL-LABEL: func_use_lds_global:
; GISEL:       ; %bb.0:
; GISEL-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GISEL-NEXT:    v_mov_b32_e32 v0, 0
; GISEL-NEXT:    ds_write_b32 v0, v0
; GISEL-NEXT:    s_waitcnt lgkmcnt(0)
; GISEL-NEXT:    s_setpc_b64 s[30:31]
  store volatile float 0.0, ptr addrspace(3) @lds, align 4
  ret void
}

; When the flag is set, the LDS address should be materialized as a constant
; and stored normally, without a trap.
define void @func_use_lds_global_constexpr_cast(ptr addrspace(1) %out) {
; SDAG-LABEL: func_use_lds_global_constexpr_cast:
; SDAG:       ; %bb.0:
; SDAG-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; SDAG-NEXT:    v_mov_b32_e32 v2, 0
; SDAG-NEXT:    global_store_dword v[0:1], v2, off
; SDAG-NEXT:    s_waitcnt vmcnt(0)
; SDAG-NEXT:    s_setpc_b64 s[30:31]
;
; GISEL-LABEL: func_use_lds_global_constexpr_cast:
; GISEL:       ; %bb.0:
; GISEL-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GISEL-NEXT:    v_mov_b32_e32 v2, 0
; GISEL-NEXT:    global_store_dword v[0:1], v2, off
; GISEL-NEXT:    s_waitcnt vmcnt(0)
; GISEL-NEXT:    s_setpc_b64 s[30:31]
  store i32 ptrtoint (ptr addrspace(3) @lds to i32), ptr addrspace(1) %out, align 4
  ret void
}
