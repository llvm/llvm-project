; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s --check-prefix=R600-CHECK
; RUN: llc < %s -march=amdgcn -mcpu=SI -verify-machineinstrs | FileCheck %s --check-prefix=SI-CHECK
; RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck %s --check-prefix=SI-CHECK

; R600-CHECK: {{^}}build_vector2:
; R600-CHECK: MOV
; R600-CHECK: MOV
; R600-CHECK-NOT: MOV
; SI-CHECK: {{^}}build_vector2:
; SI-CHECK-DAG: v_mov_b32_e32 v[[X:[0-9]]], 5
; SI-CHECK-DAG: v_mov_b32_e32 v[[Y:[0-9]]], 6
; SI-CHECK: buffer_store_dwordx2 v{{\[}}[[X]]:[[Y]]{{\]}}
define void @build_vector2 (<2 x i32> addrspace(1)* %out) {
entry:
  store <2 x i32> <i32 5, i32 6>, <2 x i32> addrspace(1)* %out
  ret void
}

; R600-CHECK: {{^}}build_vector4:
; R600-CHECK: MOV
; R600-CHECK: MOV
; R600-CHECK: MOV
; R600-CHECK: MOV
; R600-CHECK-NOT: MOV
; SI-CHECK: {{^}}build_vector4:
; SI-CHECK-DAG: v_mov_b32_e32 v[[X:[0-9]]], 5
; SI-CHECK-DAG: v_mov_b32_e32 v[[Y:[0-9]]], 6
; SI-CHECK-DAG: v_mov_b32_e32 v[[Z:[0-9]]], 7
; SI-CHECK-DAG: v_mov_b32_e32 v[[W:[0-9]]], 8
; SI-CHECK: buffer_store_dwordx4 v{{\[}}[[X]]:[[W]]{{\]}}
define void @build_vector4 (<4 x i32> addrspace(1)* %out) {
entry:
  store <4 x i32> <i32 5, i32 6, i32 7, i32 8>, <4 x i32> addrspace(1)* %out
  ret void
}
