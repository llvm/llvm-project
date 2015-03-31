; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s --check-prefix=R600-CHECK
; RUN: llc < %s -march=amdgcn -mcpu=SI -verify-machineinstrs | FileCheck %s --check-prefix=SI-CHECK
; RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck %s --check-prefix=SI-CHECK

; R600-CHECK: {{^}}test:
; R600-CHECK: MEM_RAT_CACHELESS STORE_RAW
; R600-CHECK: MEM_RAT_CACHELESS STORE_RAW

; SI-CHECK: {{^}}test:
; SI-CHECK: s_mov_b32 [[ZERO:s[0-9]]], 0{{$}}
; SI-CHECK: v_mov_b32_e32 v[[V_ZERO:[0-9]]], [[ZERO]]
; SI-CHECK: buffer_store_dwordx2 v[0:[[V_ZERO]]{{\]}}
define void @test(i64 addrspace(1)* %out, i32 %a, i32 %b, i32 %c) {
entry:
  %0 = mul i32 %a, %b
  %1 = add i32 %0, %c
  %2 = zext i32 %1 to i64
  store i64 %2, i64 addrspace(1)* %out
  ret void
}

; SI-CHECK-LABEL: {{^}}testi1toi32:
; SI-CHECK: v_cndmask_b32
define void @testi1toi32(i32 addrspace(1)* %out, i32 %a, i32 %b) {
entry:
  %0 = icmp eq i32 %a, %b
  %1 = zext i1 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; SI-CHECK-LABEL: {{^}}zext_i1_to_i64:
; SI-CHECK: v_cmp_eq_i32
; SI-CHECK: v_cndmask_b32
; SI-CHECK: s_mov_b32 s{{[0-9]+}}, 0
define void @zext_i1_to_i64(i64 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %cmp = icmp eq i32 %a, %b
  %ext = zext i1 %cmp to i64
  store i64 %ext, i64 addrspace(1)* %out, align 8
  ret void
}
