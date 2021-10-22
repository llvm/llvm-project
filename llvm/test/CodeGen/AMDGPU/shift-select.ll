; RUN: llc -march=amdgcn -mcpu=tahiti -stop-after=instruction-select -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX6 %s
; RUN: llc -march=amdgcn -mcpu=fiji -stop-after=instruction-select -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX8PLUS,GFX8 %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 -stop-after=instruction-select -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX8PLUS,GFX10 %s
; RUN: llc -march=amdgcn -mcpu=gfx1100 -stop-after=instruction-select -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX8PLUS,GFX11 %s

; GCN-LABEL: name:            s_shl_i32
; GCN: S_LSHL_B32
define amdgpu_kernel void @s_shl_i32(i32 addrspace(1)* %out, i32 %lhs, i32 %rhs) {
  %result = shl i32 %lhs, %rhs
  store i32 %result, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: name:            v_shl_i32
; GFX6: V_LSHL_B32_e32
; GFX8,GFX10: V_LSHLREV_B32_e32
; GFX11: V_DUAL_MOV_B32_e32_X_LSHLREV_B32_e32
define amdgpu_kernel void @v_shl_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %b_ptr = getelementptr i32, i32 addrspace(1)* %in, i32 %tid
  %a = load i32, i32 addrspace(1)* %in
  %b = load i32, i32 addrspace(1)* %b_ptr
  %result = shl i32 %a, %b
  store i32 %result, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: name:            s_lshr_i32
; GCN: S_LSHR_B32
define amdgpu_kernel void @s_lshr_i32(i32 addrspace(1)* %out, i32 %lhs, i32 %rhs) {
  %result = lshr i32 %lhs, %rhs
  store i32 %result, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: name:            v_lshr_i32
; GFX6: V_LSHR_B32_e32
; GFX8PLUS: V_LSHRREV_B32_e64
define amdgpu_kernel void @v_lshr_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %b_ptr = getelementptr i32, i32 addrspace(1)* %in, i32 %tid
  %a = load i32, i32 addrspace(1)* %in
  %b = load i32, i32 addrspace(1)* %b_ptr
  %result = lshr i32 %a, %b
  store i32 %result, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: name:            s_ashr_i32
; GCN: S_ASHR_I32
define amdgpu_kernel void @s_ashr_i32(i32 addrspace(1)* %out, i32 %lhs, i32 %rhs) #0 {
  %result = ashr i32 %lhs, %rhs
  store i32 %result, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: name:            v_ashr_i32
; GFX6: V_ASHR_I32_e32
; GFX8PLUS: V_ASHRREV_I32_e64
define amdgpu_kernel void @v_ashr_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %b_ptr = getelementptr i32, i32 addrspace(1)* %in, i32 %tid
  %a = load i32, i32 addrspace(1)* %in
  %b = load i32, i32 addrspace(1)* %b_ptr
  %result = ashr i32 %a, %b
  store i32 %result, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: name:            s_shl_i64
; GCN: S_LSHL_B64
define amdgpu_kernel void @s_shl_i64(i64 addrspace(1)* %out, i64 %lhs, i64 %rhs) {
  %result = shl i64 %lhs, %rhs
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: name:            v_shl_i64
; GFX6: V_LSHL_B64
; GFX8: V_LSHLREV_B64
define amdgpu_kernel void @v_shl_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %in) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %idx = zext i32 %tid to i64
  %b_ptr = getelementptr i64, i64 addrspace(1)* %in, i64 %idx
  %a = load i64, i64 addrspace(1)* %in
  %b = load i64, i64 addrspace(1)* %b_ptr
  %result = shl i64 %a, %b
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: name:            s_lshr_i64
; GCN: S_LSHR_B64
define amdgpu_kernel void @s_lshr_i64(i64 addrspace(1)* %out, i64 %lhs, i64 %rhs) {
  %result = lshr i64 %lhs, %rhs
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: name:            v_lshr_i64
; GFX6: V_LSHR_B64
; GFX8: V_LSHRREV_B64
define amdgpu_kernel void @v_lshr_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %in) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %idx = zext i32 %tid to i64
  %b_ptr = getelementptr i64, i64 addrspace(1)* %in, i64 %idx
  %a = load i64, i64 addrspace(1)* %in
  %b = load i64, i64 addrspace(1)* %b_ptr
  %result = lshr i64 %a, %b
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: name:            s_ashr_i64
; GCN: S_ASHR_I64
define amdgpu_kernel void @s_ashr_i64(i64 addrspace(1)* %out, i64 %lhs, i64 %rhs) {
  %result = ashr i64 %lhs, %rhs
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: name:            v_ashr_i64
; GFX6: V_ASHR_I64
; GFX8: V_ASHRREV_I64
define amdgpu_kernel void @v_ashr_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %in) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %idx = zext i32 %tid to i64
  %b_ptr = getelementptr i64, i64 addrspace(1)* %in, i64 %idx
  %a = load i64, i64 addrspace(1)* %in
  %b = load i64, i64 addrspace(1)* %b_ptr
  %result = ashr i64 %a, %b
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
