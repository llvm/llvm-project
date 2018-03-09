; RUN: llc -march=amdgcn -mcpu=fiji < %s | FileCheck %s

; Check transformation shl (or|add x, c2), c1 => or|add (shl x, c1), (c2 << c1)
; Only one shift if expected, GEP shall not produce a separate shift

; CHECK-LABEL: {{^}}add_const_offset:
; CHECK: v_lshlrev_b32_e32 v[[SHL:[0-9]+]], 4, v0
; CHECK: v_add_u32_e32 v[[ADD:[0-9]+]], vcc, 0xc80, v[[SHL]]
; CHECK-NOT: v_lshl
; CHECK: v_add_u32_e32 v[[ADDRLO:[0-9]+]], vcc, s{{[0-9]+}}, v[[ADD]]
; CHECK: load_dword v{{[0-9]+}}, v{{\[}}[[ADDRLO]]:
define amdgpu_kernel void @add_const_offset(i32 addrspace(1)* nocapture %arg) {
bb:
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %add = add i32 %id, 200
  %shl = shl i32 %add, 2
  %ptr = getelementptr inbounds i32, i32 addrspace(1)* %arg, i32 %shl
  %val = load i32, i32 addrspace(1)* %ptr, align 4
  store i32 %val, i32 addrspace(1)* %arg, align 4
  ret void
}

; CHECK-LABEL: {{^}}or_const_offset:
; CHECK: v_lshlrev_b32_e32 v[[SHL:[0-9]+]], 4, v0
; CHECK: v_or_b32_e32 v[[OR:[0-9]+]], 0x1000, v[[SHL]]
; CHECK-NOT: v_lshl
; CHECK: v_add_u32_e32 v[[ADDRLO:[0-9]+]], vcc, s{{[0-9]+}}, v[[OR]]
; CHECK: load_dword v{{[0-9]+}}, v{{\[}}[[ADDRLO]]:
define amdgpu_kernel void @or_const_offset(i32 addrspace(1)* nocapture %arg) {
bb:
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %add = or i32 %id, 256
  %shl = shl i32 %add, 2
  %ptr = getelementptr inbounds i32, i32 addrspace(1)* %arg, i32 %shl
  %val = load i32, i32 addrspace(1)* %ptr, align 4
  store i32 %val, i32 addrspace(1)* %arg, align 4
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
