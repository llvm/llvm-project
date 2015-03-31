; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s --check-prefix=R600-CHECK
; RUN: llc < %s -march=amdgcn -mcpu=SI -verify-machineinstrs | FileCheck %s --check-prefix=SI-CHECK
; RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck %s --check-prefix=SI-CHECK

; R600-CHECK: {{^}}v4i32_kernel_arg:
; R600-CHECK-DAG: MOV {{[* ]*}}T[[GPR:[0-9]]].X, KC0[3].Y
; R600-CHECK-DAG: MOV {{[* ]*}}T[[GPR]].Y, KC0[3].Z
; R600-CHECK-DAG: MOV {{[* ]*}}T[[GPR]].Z, KC0[3].W
; R600-CHECK-DAG: MOV {{[* ]*}}T[[GPR]].W, KC0[4].X
; SI-CHECK: {{^}}v4i32_kernel_arg:
; SI-CHECK: buffer_store_dwordx4
define void @v4i32_kernel_arg(<4 x i32> addrspace(1)* %out, <4 x i32>  %in) {
entry:
  store <4 x i32> %in, <4 x i32> addrspace(1)* %out
  ret void
}

; R600-CHECK: {{^}}v4f32_kernel_arg:
; R600-CHECK-DAG: MOV {{[* ]*}}T[[GPR:[0-9]]].X, KC0[3].Y
; R600-CHECK-DAG: MOV {{[* ]*}}T[[GPR]].Y, KC0[3].Z
; R600-CHECK-DAG: MOV {{[* ]*}}T[[GPR]].Z, KC0[3].W
; R600-CHECK-DAG: MOV {{[* ]*}}T[[GPR]].W, KC0[4].X
; SI-CHECK: {{^}}v4f32_kernel_arg:
; SI-CHECK: buffer_store_dwordx4
define void @v4f32_kernel_arg(<4 x float> addrspace(1)* %out, <4 x float>  %in) {
entry:
  store <4 x float> %in, <4 x float> addrspace(1)* %out
  ret void
}
