; RUN: opt -S -mtriple=amdgcn-unknown-amdhsa -mcpu=kaveri -amdgpu-promote-alloca < %s | FileCheck %s

; This normally would be fixed by instcombine to be compare to the GEP
; indices

; CHECK-LABEL: @lds_promoted_alloca_icmp_same_derived_pointer(
; CHECK: [[ARRAYGEP:%[0-9]+]] = getelementptr inbounds [256 x [16 x i32]], [256 x [16 x i32]] addrspace(3)* @lds_promoted_alloca_icmp_same_derived_pointer.alloca, i32 0, i32 %{{[0-9]+}}
; CHECK: %ptr0 = getelementptr inbounds [16 x i32], [16 x i32] addrspace(3)* [[ARRAYGEP]], i32 0, i32 %a
; CHECK: %ptr1 = getelementptr inbounds [16 x i32], [16 x i32] addrspace(3)* [[ARRAYGEP]], i32 0, i32 %b
; CHECK: %cmp = icmp eq i32 addrspace(3)* %ptr0, %ptr1
define void @lds_promoted_alloca_icmp_same_derived_pointer(i32 addrspace(1)* %out, i32 %a, i32 %b) #0 {
  %alloca = alloca [16 x i32], align 4
  %ptr0 = getelementptr inbounds [16 x i32], [16 x i32]* %alloca, i32 0, i32 %a
  %ptr1 = getelementptr inbounds [16 x i32], [16 x i32]* %alloca, i32 0, i32 %b
  %cmp = icmp eq i32* %ptr0, %ptr1
  %zext = zext i1 %cmp to i32
  store volatile i32 %zext, i32 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: @lds_promoted_alloca_icmp_null_rhs(
; CHECK: [[ARRAYGEP:%[0-9]+]] = getelementptr inbounds [256 x [16 x i32]], [256 x [16 x i32]] addrspace(3)* @lds_promoted_alloca_icmp_null_rhs.alloca, i32 0, i32 %{{[0-9]+}}
; CHECK: %ptr0 = getelementptr inbounds [16 x i32], [16 x i32] addrspace(3)* [[ARRAYGEP]], i32 0, i32 %a
; CHECK: %cmp = icmp eq i32 addrspace(3)* %ptr0, null
define void @lds_promoted_alloca_icmp_null_rhs(i32 addrspace(1)* %out, i32 %a, i32 %b) #0 {
  %alloca = alloca [16 x i32], align 4
  %ptr0 = getelementptr inbounds [16 x i32], [16 x i32]* %alloca, i32 0, i32 %a
  %cmp = icmp eq i32* %ptr0, null
  %zext = zext i1 %cmp to i32
  store volatile i32 %zext, i32 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: @lds_promoted_alloca_icmp_null_lhs(
; CHECK: [[ARRAYGEP:%[0-9]+]] = getelementptr inbounds [256 x [16 x i32]], [256 x [16 x i32]] addrspace(3)* @lds_promoted_alloca_icmp_null_lhs.alloca, i32 0, i32 %{{[0-9]+}}
; CHECK: %ptr0 = getelementptr inbounds [16 x i32], [16 x i32] addrspace(3)* [[ARRAYGEP]], i32 0, i32 %a
; CHECK: %cmp = icmp eq i32 addrspace(3)* null, %ptr0
define void @lds_promoted_alloca_icmp_null_lhs(i32 addrspace(1)* %out, i32 %a, i32 %b) #0 {
  %alloca = alloca [16 x i32], align 4
  %ptr0 = getelementptr inbounds [16 x i32], [16 x i32]* %alloca, i32 0, i32 %a
  %cmp = icmp eq i32* null, %ptr0
  %zext = zext i1 %cmp to i32
  store volatile i32 %zext, i32 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: @lds_promoted_alloca_icmp_unknown_ptr(
; CHECK: %alloca = alloca [16 x i32], align 4
; CHECK: %ptr0 = getelementptr inbounds [16 x i32], [16 x i32]* %alloca, i32 0, i32 %a
; CHECK: %ptr1 = call i32* @get_unknown_pointer()
; CHECK: %cmp = icmp eq i32* %ptr0, %ptr1
define void @lds_promoted_alloca_icmp_unknown_ptr(i32 addrspace(1)* %out, i32 %a, i32 %b) #0 {
  %alloca = alloca [16 x i32], align 4
  %ptr0 = getelementptr inbounds [16 x i32], [16 x i32]* %alloca, i32 0, i32 %a
  %ptr1 = call i32* @get_unknown_pointer()
  %cmp = icmp eq i32* %ptr0, %ptr1
  %zext = zext i1 %cmp to i32
  store volatile i32 %zext, i32 addrspace(1)* %out
  ret void
}

declare i32* @get_unknown_pointer() #0

attributes #0 = { nounwind "amdgpu-waves-per-eu"="1,1" }
