; RUN: llc -march=r600 -mcpu=cypress < %s | FileCheck -check-prefix=EG %s

; EG-LABEL: {{^}}and_setcc_setcc_i32:
; EG: AND_INT
; EG-NEXT: SETE_INT
define amdgpu_kernel void @and_setcc_setcc_i32(ptr addrspace(1) %out, i32 %a, i32 %b) {
  %cmp1 = icmp eq i32 %a, -1
  %cmp2 = icmp eq i32 %b, -1
  %and = and i1 %cmp1, %cmp2
  %ext = sext i1 %and to i32
  store i32 %ext, ptr addrspace(1) %out, align 4
  ret void
}

; EG-LABEL: {{^}}and_setcc_setcc_v4i32:
; EG: AND_INT
; EG: AND_INT
; EG: SETE_INT
; EG: AND_INT
; EG: SETE_INT
; EG: AND_INT
; EG: SETE_INT
define amdgpu_kernel void @and_setcc_setcc_v4i32(ptr addrspace(1) %out, <4 x i32> %a, <4 x i32> %b) {
  %cmp1 = icmp eq <4 x i32> %a, <i32 -1, i32 -1, i32 -1, i32 -1>
  %cmp2 = icmp eq <4 x i32> %b, <i32 -1, i32 -1, i32 -1, i32 -1>
  %and = and <4 x i1> %cmp1, %cmp2
  %ext = sext <4 x i1> %and to <4 x i32>
  store <4 x i32> %ext, ptr addrspace(1) %out, align 4
  ret void
}
