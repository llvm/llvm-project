; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=kaveri -mattr=+flat-for-global < %s | FileCheck -check-prefix=HSA -check-prefix=HSA-DEFAULT -check-prefix=ALL %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=kaveri -mattr=-flat-for-global < %s | FileCheck -check-prefix=HSA -check-prefix=HSA-NODEFAULT -check-prefix=ALL %s
; RUN: llc -mtriple=amdgcn-- -mcpu=kaveri -mattr=-flat-for-global < %s | FileCheck -check-prefix=NOHSA-DEFAULT -check-prefix=ALL %s
; RUN: llc -mtriple=amdgcn-- -mcpu=kaveri -mattr=+flat-for-global < %s | FileCheck -check-prefix=NOHSA-NODEFAULT -check-prefix=ALL %s


; There are no stack objects even though flat is used by default, so
; flat_scratch_init should be disabled.

; ALL-LABEL: {{^}}test:
; HSA: .amd_kernel_code_t
; HSA: enable_sgpr_flat_scratch_init = 0
; HSA: .end_amd_kernel_code_t

; ALL-NOT: flat_scr

; HSA-DEFAULT: flat_store_dword
; HSA-NODEFAULT: buffer_store_dword

; NOHSA-DEFAULT: buffer_store_dword
; NOHSA-NODEFAULT: flat_store_dword
define void @test(i32 addrspace(1)* %out) {
entry:
  store i32 0, i32 addrspace(1)* %out
  ret void
}
