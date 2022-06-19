; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -o - %s | FileCheck %s
; RUN: llc -global-isel -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -o - %s | FileCheck %s

; Test that a null check is not emitted for lowered addrspacecast


define void @flat_user(i8* %ptr) {
  store i8 0, i8* %ptr
  ret void
}

; CHECK-LABEL: {{^}}cast_alloca:
; CHECK: s_getreg_b32 [[GETREG:s[0-9]+]], hwreg(HW_REG_SH_MEM_BASES, 0, 16)
; CHECK: s_lshl_b32 [[APERTURE:s[0-9]+]], [[GETREG]], 16
; CHECK: v_lshrrev_b32_e64 v0, 6, s33
; CHECK-NEXT: v_mov_b32_e32 v1, [[APERTURE]]
; CHECK-NOT: v0
; CHECK-NOT: v1
define void @cast_alloca() {
  %alloca = alloca i8, addrspace(5)
  %cast = addrspacecast i8 addrspace(5)* %alloca to i8*
  call void @flat_user(i8* %cast)
  ret void
}

@lds = internal unnamed_addr addrspace(3) global i8 undef, align 4

; CHECK-LABEL: {{^}}cast_lds_gv:
; CHECK: s_getreg_b32 [[GETREG:s[0-9]+]], hwreg(HW_REG_SH_MEM_BASES, 16, 16)
; CHECK: s_lshl_b32 [[APERTURE:s[0-9]+]], [[GETREG]], 16
; CHECK: v_mov_b32_e32 v0, 0
; CHECK: v_mov_b32_e32 v1, [[APERTURE]]
; CHECK-NOT: v0
; CHECK-NOT: v1
define void @cast_lds_gv() {
  %cast = addrspacecast i8 addrspace(3)* @lds to i8*
  call void @flat_user(i8* %cast)
  ret void
}

; CHECK-LABEL: {{^}}cast_constant_lds_neg1_gv:
; CHECK: v_mov_b32_e32 v0, 0
; CHECK: v_mov_b32_e32 v1, 0
define void @cast_constant_lds_neg1_gv() {
  call void @flat_user(i8* addrspacecast (i8 addrspace(3)* inttoptr (i32 -1 to i8 addrspace(3)*) to i8*))
  ret void
}

; CHECK-LABEL: {{^}}cast_constant_private_neg1_gv:
; CHECK: v_mov_b32_e32 v0, 0
; CHECK: v_mov_b32_e32 v1, 0
define void @cast_constant_private_neg1_gv() {
  call void @flat_user(i8* addrspacecast (i8 addrspace(5)* inttoptr (i32 -1 to i8 addrspace(5)*) to i8*))
  ret void
}

; CHECK-LABEL: {{^}}cast_constant_lds_other_gv:
; CHECK: s_getreg_b32 [[GETREG:s[0-9]+]], hwreg(HW_REG_SH_MEM_BASES, 16, 16)
; CHECK: s_lshl_b32 [[APERTURE:s[0-9]+]], [[GETREG]], 16
; CHECK: v_mov_b32_e32 v0, 0x7b
; CHECK: v_mov_b32_e32 v1, [[APERTURE]]
define void @cast_constant_lds_other_gv() {
  call void @flat_user(i8* addrspacecast (i8 addrspace(3)* inttoptr (i32 123 to i8 addrspace(3)*) to i8*))
  ret void
}

; CHECK-LABEL: {{^}}cast_constant_private_other_gv:
; CHECK: s_getreg_b32 [[GETREG:s[0-9]+]], hwreg(HW_REG_SH_MEM_BASES, 0, 16)
; CHECK: s_lshl_b32 [[APERTURE:s[0-9]+]], [[GETREG]], 16
; CHECK: v_mov_b32_e32 v0, 0x7b
; CHECK: v_mov_b32_e32 v1, [[APERTURE]]
define void @cast_constant_private_other_gv() {
  call void @flat_user(i8* addrspacecast (i8 addrspace(5)* inttoptr (i32 123 to i8 addrspace(5)*) to i8*))
  ret void
}
