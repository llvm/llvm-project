; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN %s

; GCN-LABEL: {{^}}lsh8_or_and:
; GCN: v_mov_b32_e32 [[MASK:v[0-9]+]], 0x6050400
; GCN: v_perm_b32 v{{[0-9]+}}, {{[vs][0-9]+}}, {{[vs][0-9]+}}, [[MASK]]
define amdgpu_kernel void @lsh8_or_and(i32 addrspace(1)* nocapture %arg, i32 %arg1) {
bb:
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr i32, i32 addrspace(1)* %arg, i32 %id
  %tmp = load i32, i32 addrspace(1)* %gep, align 4
  %tmp2 = shl i32 %tmp, 8
  %tmp3 = and i32 %arg1, 255
  %tmp4 = or i32 %tmp2, %tmp3
  store i32 %tmp4, i32 addrspace(1)* %gep, align 4
  ret void
}

; GCN-LABEL: {{^}}lsr24_or_and:
; GCN: v_mov_b32_e32 [[MASK:v[0-9]+]], 0x7060503
; GCN: v_perm_b32 v{{[0-9]+}}, {{[vs][0-9]+}}, {{[vs][0-9]+}}, [[MASK]]
define amdgpu_kernel void @lsr24_or_and(i32 addrspace(1)* nocapture %arg, i32 %arg1) {
bb:
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr i32, i32 addrspace(1)* %arg, i32 %id
  %tmp = load i32, i32 addrspace(1)* %gep, align 4
  %tmp2 = lshr i32 %tmp, 24
  %tmp3 = and i32 %arg1, 4294967040 ; 0xffffff00
  %tmp4 = or i32 %tmp2, %tmp3
  store i32 %tmp4, i32 addrspace(1)* %gep, align 4
  ret void
}

; GCN-LABEL: {{^}}and_or_lsr24:
; GCN: v_mov_b32_e32 [[MASK:v[0-9]+]], 0x7060503
; GCN: v_perm_b32 v{{[0-9]+}}, {{[vs][0-9]+}}, {{[vs][0-9]+}}, [[MASK]]
define amdgpu_kernel void @and_or_lsr24(i32 addrspace(1)* nocapture %arg, i32 %arg1) {
bb:
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr i32, i32 addrspace(1)* %arg, i32 %id
  %tmp = load i32, i32 addrspace(1)* %gep, align 4
  %tmp2 = and i32 %tmp, 4294967040 ; 0xffffff00
  %tmp3 = lshr i32 %arg1, 24
  %tmp4 = or i32 %tmp2, %tmp3
  %tmp5 = xor i32 %tmp4, -2147483648
  store i32 %tmp5, i32 addrspace(1)* %gep, align 4
  ret void
}

; GCN-LABEL: {{^}}and_or_and:
; GCN: v_mov_b32_e32 [[MASK:v[0-9]+]], 0x7020500
; GCN: v_perm_b32 v{{[0-9]+}}, {{[vs][0-9]+}}, {{[vs][0-9]+}}, [[MASK]]
define amdgpu_kernel void @and_or_and(i32 addrspace(1)* nocapture %arg, i32 %arg1) {
bb:
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr i32, i32 addrspace(1)* %arg, i32 %id
  %tmp = load i32, i32 addrspace(1)* %gep, align 4
  %tmp2 = and i32 %tmp, -16711936
  %tmp3 = and i32 %arg1, 16711935
  %tmp4 = or i32 %tmp2, %tmp3
  store i32 %tmp4, i32 addrspace(1)* %gep, align 4
  ret void
}

; GCN-LABEL: {{^}}lsh8_or_lsr24:
; GCN: v_mov_b32_e32 [[MASK:v[0-9]+]], 0x6050403
; GCN: v_perm_b32 v{{[0-9]+}}, {{[vs][0-9]+}}, {{[vs][0-9]+}}, [[MASK]]
define amdgpu_kernel void @lsh8_or_lsr24(i32 addrspace(1)* nocapture %arg, i32 %arg1) {
bb:
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr i32, i32 addrspace(1)* %arg, i32 %id
  %tmp = load i32, i32 addrspace(1)* %gep, align 4
  %tmp2 = shl i32 %tmp, 8
  %tmp3 = lshr i32 %arg1, 24
  %tmp4 = or i32 %tmp2, %tmp3
  store i32 %tmp4, i32 addrspace(1)* %gep, align 4
  ret void
}

; GCN-LABEL: {{^}}lsh16_or_lsr24:
; GCN: v_mov_b32_e32 [[MASK:v[0-9]+]], 0x5040c03
; GCN: v_perm_b32 v{{[0-9]+}}, {{[vs][0-9]+}}, {{[vs][0-9]+}}, [[MASK]]
define amdgpu_kernel void @lsh16_or_lsr24(i32 addrspace(1)* nocapture %arg, i32 %arg1) {
bb:
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr i32, i32 addrspace(1)* %arg, i32 %id
  %tmp = load i32, i32 addrspace(1)* %gep, align 4
  %tmp2 = shl i32 %tmp, 16
  %tmp3 = lshr i32 %arg1, 24
  %tmp4 = or i32 %tmp2, %tmp3
  store i32 %tmp4, i32 addrspace(1)* %gep, align 4
  ret void
}

; GCN-LABEL: {{^}}and_xor_and:
; GCN: v_mov_b32_e32 [[MASK:v[0-9]+]], 0x7020104
; GCN: v_perm_b32 v{{[0-9]+}}, {{[vs][0-9]+}}, {{[vs][0-9]+}}, [[MASK]]
define amdgpu_kernel void @and_xor_and(i32 addrspace(1)* nocapture %arg, i32 %arg1) {
bb:
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr i32, i32 addrspace(1)* %arg, i32 %id
  %tmp = load i32, i32 addrspace(1)* %gep, align 4
  %tmp2 = and i32 %tmp, -16776961
  %tmp3 = and i32 %arg1, 16776960
  %tmp4 = xor i32 %tmp2, %tmp3
  store i32 %tmp4, i32 addrspace(1)* %gep, align 4
  ret void
}

; GCN-LABEL: {{^}}and_or_or_and:
; GCN: v_mov_b32_e32 [[MASK:v[0-9]+]], 0xffff0500
; GCN: v_perm_b32 v{{[0-9]+}}, {{[vs][0-9]+}}, {{[vs][0-9]+}}, [[MASK]]
define amdgpu_kernel void @and_or_or_and(i32 addrspace(1)* nocapture %arg, i32 %arg1) {
bb:
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr i32, i32 addrspace(1)* %arg, i32 %id
  %tmp = load i32, i32 addrspace(1)* %gep, align 4
  %and = and i32 %tmp, 16711935     ; 0x00ff00ff
  %tmp1 = and i32 %arg1, 4294967040 ; 0xffffff00
  %tmp2 = or i32 %tmp1, -65536
  %tmp3 = or i32 %tmp2, %and
  store i32 %tmp3, i32 addrspace(1)* %gep, align 4
  ret void
}

; GCN-LABEL: {{^}}and_or_and_shl:
; GCN: v_mov_b32_e32 [[MASK:v[0-9]+]], 0x50c0c00
; GCN: v_perm_b32 v{{[0-9]+}}, {{[vs][0-9]+}}, {{[vs][0-9]+}}, [[MASK]]
define amdgpu_kernel void @and_or_and_shl(i32 addrspace(1)* nocapture %arg, i32 %arg1) {
bb:
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr i32, i32 addrspace(1)* %arg, i32 %id
  %tmp = load i32, i32 addrspace(1)* %gep, align 4
  %tmp2 = shl i32 %tmp, 16
  %tmp3 = and i32 %arg1, 65535
  %tmp4 = or i32 %tmp2, %tmp3
  %and = and i32 %tmp4, 4278190335
  store i32 %and, i32 addrspace(1)* %gep, align 4
  ret void
}

; GCN-LABEL: {{^}}or_and_or:
; GCN: v_mov_b32_e32 [[MASK:v[0-9]+]], 0x7020104
; GCN: v_perm_b32 v{{[0-9]+}}, {{[vs][0-9]+}}, {{[vs][0-9]+}}, [[MASK]]
define amdgpu_kernel void @or_and_or(i32 addrspace(1)* nocapture %arg, i32 %arg1) {
bb:
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr i32, i32 addrspace(1)* %arg, i32 %id
  %tmp = load i32, i32 addrspace(1)* %gep, align 4
  %or1 = or i32 %tmp, 16776960    ; 0x00ffff00
  %or2 = or i32 %arg1, 4278190335 ; 0xff0000ff
  %and = and i32 %or1, %or2
  store i32 %and, i32 addrspace(1)* %gep, align 4
  ret void
}

; GCN-LABEL: {{^}}known_ffff0500:
; GCN-DAG: v_mov_b32_e32 [[MASK:v[0-9]+]], 0xffff0500
; GCN-DAG: v_mov_b32_e32 [[RES:v[0-9]+]], 0xffff8004
; GCN: v_perm_b32 v{{[0-9]+}}, {{[vs][0-9]+}}, {{[vs][0-9]+}}, [[MASK]]
; GCN: store_dword v[{{[0-9:]+}}], [[RES]]{{$}}
define amdgpu_kernel void @known_ffff0500(i32 addrspace(1)* nocapture %arg, i32 %arg1) {
bb:
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr i32, i32 addrspace(1)* %arg, i32 %id
  %load = load i32, i32 addrspace(1)* %gep, align 4
  %mask1 = or i32 %arg1, 32768 ; 0x8000
  %mask2 = or i32 %load, 4
  %and = and i32 %mask2, 16711935     ; 0x00ff00ff
  %tmp1 = and i32 %mask1, 4294967040 ; 0xffffff00
  %tmp2 = or i32 %tmp1, 4294901760   ; 0xffff0000
  %tmp3 = or i32 %tmp2, %and
  store i32 %tmp3, i32 addrspace(1)* %gep, align 4
  %v = and i32 %tmp3, 4294934532 ; 0xffff8004
  store i32 %v, i32 addrspace(1)* %arg, align 4
  ret void
}

; GCN-LABEL: {{^}}known_050c0c00:
; GCN-DAG: v_mov_b32_e32 [[MASK:v[0-9]+]], 0x50c0c00
; GCN-DAG: v_mov_b32_e32 [[RES:v[0-9]+]], 4{{$}}
; GCN: v_perm_b32 v{{[0-9]+}}, {{[vs][0-9]+}}, {{[vs][0-9]+}}, [[MASK]]
; GCN: store_dword v[{{[0-9:]+}}], [[RES]]{{$}}
define amdgpu_kernel void @known_050c0c00(i32 addrspace(1)* nocapture %arg, i32 %arg1) {
bb:
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr i32, i32 addrspace(1)* %arg, i32 %id
  %tmp = load i32, i32 addrspace(1)* %gep, align 4
  %tmp2 = shl i32 %tmp, 16
  %mask = or i32 %arg1, 4
  %tmp3 = and i32 %mask, 65535
  %tmp4 = or i32 %tmp2, %tmp3
  %and = and i32 %tmp4, 4278190335
  store i32 %and, i32 addrspace(1)* %gep, align 4
  %v = and i32 %and, 16776964
  store i32 %v, i32 addrspace(1)* %arg, align 4
  ret void
}

; GCN-LABEL: {{^}}known_ffff8004:
; GCN-DAG: v_mov_b32_e32 [[MASK:v[0-9]+]], 0xffff0500
; GCN-DAG: v_mov_b32_e32 [[RES:v[0-9]+]], 0xffff8004
; GCN: v_perm_b32 v{{[0-9]+}}, {{[vs][0-9]+}}, {{[vs][0-9]+}}, [[MASK]]
; GCN: store_dword v[{{[0-9:]+}}], [[RES]]{{$}}
define amdgpu_kernel void @known_ffff8004(i32 addrspace(1)* nocapture %arg, i32 %arg1) {
bb:
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr i32, i32 addrspace(1)* %arg, i32 %id
  %load = load i32, i32 addrspace(1)* %gep, align 4
  %mask1 = or i32 %arg1, 4
  %mask2 = or i32 %load, 32768 ; 0x8000
  %and = and i32 %mask1, 16711935     ; 0x00ff00ff
  %tmp1 = and i32 %mask2, 4294967040 ; 0xffffff00
  %tmp2 = or i32 %tmp1, 4294901760   ; 0xffff0000
  %tmp3 = or i32 %tmp2, %and
  store i32 %tmp3, i32 addrspace(1)* %gep, align 4
  %v = and i32 %tmp3, 4294934532 ; 0xffff8004
  store i32 %v, i32 addrspace(1)* %arg, align 4
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
