; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=kaveri --amdhsa-code-object-version=2 -mattr=+flat-for-global < %s | FileCheck -check-prefix=HSA -check-prefix=HSA-DEFAULT -check-prefix=ALL %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=kaveri --amdhsa-code-object-version=2 -mattr=-flat-for-global < %s | FileCheck -check-prefix=HSA -check-prefix=HSA-NODEFAULT -check-prefix=ALL %s
; RUN: llc -mtriple=amdgcn-- -mcpu=tonga < %s | FileCheck -check-prefix=HSA-NOADDR64 -check-prefix=ALL %s
; RUN: llc -mtriple=amdgcn-- -mcpu=kaveri -mattr=-flat-for-global < %s | FileCheck -check-prefix=NOHSA-DEFAULT -check-prefix=ALL %s
; RUN: llc -mtriple=amdgcn-- -mcpu=kaveri -mattr=+flat-for-global < %s | FileCheck -check-prefix=NOHSA-NODEFAULT -check-prefix=ALL %s
; RUN: llc -mtriple=amdgcn-- -mcpu=tonga < %s | FileCheck -check-prefix=NOHSA-NOADDR64 -check-prefix=ALL %s


; There are no stack objects even though flat is used by default, so
; flat_scratch_init should be disabled.

; ALL-LABEL: {{^}}test:
; HSA: .amd_kernel_code_t
; HSA: enable_sgpr_flat_scratch_init = 0
; HSA: .end_amd_kernel_code_t

; ALL-NOT: flat_scr

; HSA-DEFAULT: flat_store_dword
; HSA-NODEFAULT: buffer_store_dword
; HSA-NOADDR64: flat_store_dword

; NOHSA-DEFAULT: buffer_store_dword
; NOHSA-NODEFAULT: flat_store_dword
; NOHSA-NOADDR64: flat_store_dword
define amdgpu_kernel void @test(i32 addrspace(1)* %out) {
entry:
  store i32 0, i32 addrspace(1)* %out
  ret void
}

; HSA-DEFAULT: flat_store_dword
; HSA-NODEFAULT: buffer_store_dword
; HSA-NOADDR64: flat_store_dword

; NOHSA-DEFAULT: buffer_store_dword
; NOHSA-NODEFAULT: flat_store_dword
; NOHSA-NOADDR64: flat_store_dword
define amdgpu_kernel void @test_addr64(i32 addrspace(1)* %out) {
entry:
  %out.addr = alloca i32 addrspace(1)*, align 4, addrspace(5)

  store i32 addrspace(1)* %out, i32 addrspace(1)* addrspace(5)* %out.addr, align 4
  %ld0 = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(5)* %out.addr, align 4

  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %ld0, i32 0
  store i32 1, i32 addrspace(1)* %arrayidx, align 4

  %ld1 = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(5)* %out.addr, align 4
  %arrayidx1 = getelementptr inbounds i32, i32 addrspace(1)* %ld1, i32 1
  store i32 2, i32 addrspace(1)* %arrayidx1, align 4

  ret void
}
