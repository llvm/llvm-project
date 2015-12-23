; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=kaveri | FileCheck -check-prefix=HSA-DEFAULT %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=kaveri -mattr=-flat-for-global | FileCheck -check-prefix=HSA-NODEFAULT %s
; RUN: llc < %s -mtriple=amdgcn -mcpu=kaveri | FileCheck -check-prefix=NOHSA-DEFAULT %s
; RUN: llc < %s -mtriple=amdgcn -mcpu=kaveri -mattr=+flat-for-global | FileCheck -check-prefix=NOHSA-NODEFAULT %s


; HSA-DEFAULT: flat_store_dword
; HSA-NODEFAULT: buffer_store_dword
; NOHSA-DEFAULT: buffer_store_dword
; NOHSA-NODEFAULT: flat_store_dword
define void @test(i32 addrspace(1)* %out) {
entry:
  store i32 0, i32 addrspace(1)* %out
  ret void
}
