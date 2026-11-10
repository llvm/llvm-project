; RUN: llc -mtriple=amdgpu9.0a-amd-amdhsa -mattr=-unaligned-access-mode -global-isel=0 -verify-machineinstrs -stop-after=finalize-isel < %s | FileCheck %s --implicit-check-not="= GLOBAL_LOAD_"
; RUN: llc -mtriple=amdgpu9.0a-amd-amdhsa -mattr=-unaligned-access-mode -global-isel=0 -verify-machineinstrs -stop-after=finalize-isel -O0 < %s | FileCheck %s --implicit-check-not="= GLOBAL_LOAD_"

; Type legalization can introduce misaligned loads after the early DAG combine.
; Both the dword vector and sub-dword tail must respect the original alignment.

define void @i80_align1(ptr addrspace(1) %p, ptr addrspace(1) %out) {
; CHECK-LABEL: name: i80_align1
; CHECK-COUNT-10: = GLOBAL_LOAD_UBYTE
  %v = load i80, ptr addrspace(1) %p, align 1
  store i80 %v, ptr addrspace(1) %out, align 16
  ret void
}

define void @i80_align2(ptr addrspace(1) %p, ptr addrspace(1) %out) {
; CHECK-LABEL: name: i80_align2
; CHECK-COUNT-5: = GLOBAL_LOAD_USHORT
  %v = load i80, ptr addrspace(1) %p, align 2
  store i80 %v, ptr addrspace(1) %out, align 16
  ret void
}

define void @i88_align1(ptr addrspace(1) %p, ptr addrspace(1) %out) {
; CHECK-LABEL: name: i88_align1
; CHECK-COUNT-11: = GLOBAL_LOAD_UBYTE
  %v = load i88, ptr addrspace(1) %p, align 1
  store i88 %v, ptr addrspace(1) %out, align 16
  ret void
}

define void @i136_align1(ptr addrspace(1) %p, ptr addrspace(1) %out) {
; CHECK-LABEL: name: i136_align1
; CHECK-COUNT-17: = GLOBAL_LOAD_UBYTE
  %v = load i136, ptr addrspace(1) %p, align 1
  store i136 %v, ptr addrspace(1) %out, align 16
  ret void
}

define void @i80_constant_align1(ptr addrspace(4) %p, ptr addrspace(1) %out) {
; CHECK-LABEL: name: i80_constant_align1
; CHECK-COUNT-10: = GLOBAL_LOAD_UBYTE
  %v = load i80, ptr addrspace(4) %p, align 1
  store i80 %v, ptr addrspace(1) %out, align 16
  ret void
}

define void @i64_align1(ptr addrspace(1) %p, ptr addrspace(1) %out) {
; CHECK-LABEL: name: i64_align1
; CHECK-COUNT-8: = GLOBAL_LOAD_UBYTE
  %v = load i64, ptr addrspace(1) %p, align 1
  store i64 %v, ptr addrspace(1) %out, align 16
  ret void
}
