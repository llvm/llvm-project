; RUN: llc < %s -march=amdgcn -verify-machineinstrs | FileCheck %s --check-prefix=SI
; RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck %s --check-prefix=SI
; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s --check-prefix=R600

; R600: {{^}}test:
; R600: MEM_RAT_CACHELESS STORE_RAW
; R600: MEM_RAT_CACHELESS STORE_RAW

; SI: {{^}}test:
; SI: v_mov_b32_e32 v[[V_ZERO:[0-9]]], 0{{$}}
; SI: buffer_store_dwordx2 v[0:[[V_ZERO]]{{\]}}
define void @test(i64 addrspace(1)* %out, i32 %a, i32 %b, i32 %c) {
entry:
  %0 = mul i32 %a, %b
  %1 = add i32 %0, %c
  %2 = zext i32 %1 to i64
  store i64 %2, i64 addrspace(1)* %out
  ret void
}

; SI-LABEL: {{^}}testi1toi32:
; SI: v_cndmask_b32
define void @testi1toi32(i32 addrspace(1)* %out, i32 %a, i32 %b) {
entry:
  %0 = icmp eq i32 %a, %b
  %1 = zext i1 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; SI-LABEL: {{^}}zext_i1_to_i64:
; SI: s_mov_b32 s{{[0-9]+}}, 0
; SI: v_cmp_eq_i32
; SI: v_cndmask_b32
define void @zext_i1_to_i64(i64 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %cmp = icmp eq i32 %a, %b
  %ext = zext i1 %cmp to i64
  store i64 %ext, i64 addrspace(1)* %out, align 8
  ret void
}
