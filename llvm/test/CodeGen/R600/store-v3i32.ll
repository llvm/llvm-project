; XFAIL: *
; RUN: llc -verify-machineinstrs -march=amdgcn -mcpu=SI < %s | FileCheck -check-prefix=SI %s
; RUN: llc -verify-machineinstrs -march=amdgcn -mcpu=tonga < %s | FileCheck -check-prefix=SI %s

; 3 vectors have the same size and alignment as 4 vectors, so this
; should be done in a single store.

; SI-LABEL: {{^}}store_v3i32:
; SI: buffer_store_dwordx4
define void @store_v3i32(<3 x i32> addrspace(1)* %out, <3 x i32> %a) nounwind {
  store <3 x i32> %a, <3 x i32> addrspace(1)* %out, align 16
  ret void
}
