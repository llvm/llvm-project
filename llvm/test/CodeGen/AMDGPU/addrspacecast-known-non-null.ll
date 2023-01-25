; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -o - %s | FileCheck %s
; RUN: llc -global-isel -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -o - %s | FileCheck %s

; Test that a null check is not emitted for lowered addrspacecast


define void @flat_user(ptr %ptr) {
  store i8 0, ptr %ptr
  ret void
}

; CHECK-LABEL: {{^}}cast_alloca:
; CHECK: s_mov_b64 s[{{[0-9]+}}:[[HIREG:[0-9]+]]], src_private_base
; CHECK: v_mov_b32_e32 v1, s[[HIREG]]
; CHECK-NOT: v0
; CHECK-NOT: v1
define void @cast_alloca() {
  %alloca = alloca i8, addrspace(5)
  %cast = addrspacecast ptr addrspace(5) %alloca to ptr
  call void @flat_user(ptr %cast)
  ret void
}

@lds = internal unnamed_addr addrspace(3) global i8 undef, align 4

; CHECK-LABEL: {{^}}cast_lds_gv:
; CHECK: s_mov_b64 s[{{[0-9]+}}:[[HIREG:[0-9]+]]], src_shared_base
; CHECK: v_mov_b32_e32 v0, 0
; CHECK: v_mov_b32_e32 v1, s[[HIREG]]
; CHECK-NOT: v0
; CHECK-NOT: v1
define amdgpu_kernel void @cast_lds_gv() {
  %cast = addrspacecast ptr addrspace(3) @lds to ptr
  call void @flat_user(ptr %cast)
  ret void
}

; CHECK-LABEL: {{^}}cast_constant_lds_neg1_gv:
; CHECK: v_mov_b32_e32 v0, 0
; CHECK: v_mov_b32_e32 v1, 0
define void @cast_constant_lds_neg1_gv() {
  call void @flat_user(ptr addrspacecast (ptr addrspace(3) inttoptr (i32 -1 to ptr addrspace(3)) to ptr))
  ret void
}

; CHECK-LABEL: {{^}}cast_constant_private_neg1_gv:
; CHECK: v_mov_b32_e32 v0, 0
; CHECK: v_mov_b32_e32 v1, 0
define void @cast_constant_private_neg1_gv() {
  call void @flat_user(ptr addrspacecast (ptr addrspace(5) inttoptr (i32 -1 to ptr addrspace(5)) to ptr))
  ret void
}

; CHECK-LABEL: {{^}}cast_constant_lds_other_gv:
; CHECK: s_mov_b64 s[{{[0-9]+}}:[[HIREG:[0-9]+]]], src_shared_base
; CHECK: v_mov_b32_e32 v0, 0x7b
; CHECK: v_mov_b32_e32 v1, s[[HIREG]]
define void @cast_constant_lds_other_gv() {
  call void @flat_user(ptr addrspacecast (ptr addrspace(3) inttoptr (i32 123 to ptr addrspace(3)) to ptr))
  ret void
}

; CHECK-LABEL: {{^}}cast_constant_private_other_gv:
; CHECK: s_mov_b64 s[{{[0-9]+}}:[[HIREG:[0-9]+]]], src_private_base
; CHECK: v_mov_b32_e32 v0, 0x7b
; CHECK: v_mov_b32_e32 v1, s[[HIREG]]
define void @cast_constant_private_other_gv() {
  call void @flat_user(ptr addrspacecast (ptr addrspace(5) inttoptr (i32 123 to ptr addrspace(5)) to ptr))
  ret void
}
