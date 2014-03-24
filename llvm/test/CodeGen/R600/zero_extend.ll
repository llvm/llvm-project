; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s --check-prefix=R600-CHECK
; RUN: llc < %s -march=r600 -mcpu=SI -verify-machineinstrs | FileCheck %s --check-prefix=SI-CHECK

; R600-CHECK: @test
; R600-CHECK: MEM_RAT_CACHELESS STORE_RAW
; R600-CHECK: MEM_RAT_CACHELESS STORE_RAW

; SI-CHECK: @test
; SI-CHECK: V_MOV_B32_e32 v[[ZERO:[0-9]]], 0
; SI-CHECK: BUFFER_STORE_DWORDX2 v[0:[[ZERO]]{{\]}}
define void @test(i64 addrspace(1)* %out, i32 %a, i32 %b, i32 %c) {
entry:
  %0 = mul i32 %a, %b
  %1 = add i32 %0, %c
  %2 = zext i32 %1 to i64
  store i64 %2, i64 addrspace(1)* %out
  ret void
}

; SI-CHECK-LABEL: @testi1toi32
; SI-CHECK: V_CNDMASK_B32
define void @testi1toi32(i32 addrspace(1)* %out, i32 %a, i32 %b) {
entry:
  %0 = icmp eq i32 %a, %b
  %1 = zext i1 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}
