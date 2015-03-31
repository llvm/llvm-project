; XFAIL: *
; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs< %s | FileCheck -check-prefix=SI
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs< %s | FileCheck -check-prefix=SI

; SI-LABEL: {{^}}global_store_v3i64:
; SI: buffer_store_dwordx4
; SI: buffer_store_dwordx4
define void @global_store_v3i64(<3 x i64> addrspace(1)* %out, <3 x i64> %x) {
  store <3 x i64> %x, <3 x i64> addrspace(1)* %out, align 32
  ret void
}

; SI-LABEL: {{^}}global_store_v3i64_unaligned:
define void @global_store_v3i64_unaligned(<3 x i64> addrspace(1)* %out, <3 x i64> %x) {
  store <3 x i64> %x, <3 x i64> addrspace(1)* %out, align 1
  ret void
}

; SI-LABEL: {{^}}local_store_v3i64:
define void @local_store_v3i64(<3 x i64> addrspace(3)* %out, <3 x i64> %x) {
  store <3 x i64> %x, <3 x i64> addrspace(3)* %out, align 32
  ret void
}

; SI-LABEL: {{^}}local_store_v3i64_unaligned:
define void @local_store_v3i64_unaligned(<3 x i64> addrspace(1)* %out, <3 x i64> %x) {
  store <3 x i64> %x, <3 x i64> addrspace(1)* %out, align 1
  ret void
}
