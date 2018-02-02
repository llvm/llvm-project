; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -infer-address-spaces %s | FileCheck %s

; Trivial optimization of generic addressing

; CHECK-LABEL: @load_global_from_flat(
; CHECK-NEXT: %tmp0 = addrspacecast float* %generic_scalar to float addrspace(1)*
; CHECK-NEXT: %tmp1 = load float, float addrspace(1)* %tmp0
; CHECK-NEXT: ret float %tmp1
define float @load_global_from_flat(float* %generic_scalar) #0 {
  %tmp0 = addrspacecast float* %generic_scalar to float addrspace(1)*
  %tmp1 = load float, float addrspace(1)* %tmp0
  ret float %tmp1
}

; CHECK-LABEL: @load_constant_from_flat(
; CHECK-NEXT: %tmp0 = addrspacecast float* %generic_scalar to float addrspace(2)*
; CHECK-NEXT: %tmp1 = load float, float addrspace(2)* %tmp0
; CHECK-NEXT: ret float %tmp1
define float @load_constant_from_flat(float* %generic_scalar) #0 {
  %tmp0 = addrspacecast float* %generic_scalar to float addrspace(2)*
  %tmp1 = load float, float addrspace(2)* %tmp0
  ret float %tmp1
}

; CHECK-LABEL: @load_group_from_flat(
; CHECK-NEXT: %tmp0 = addrspacecast float* %generic_scalar to float addrspace(3)*
; CHECK-NEXT: %tmp1 = load float, float addrspace(3)* %tmp0
; CHECK-NEXT: ret float %tmp1
define float @load_group_from_flat(float* %generic_scalar) #0 {
  %tmp0 = addrspacecast float* %generic_scalar to float addrspace(3)*
  %tmp1 = load float, float addrspace(3)* %tmp0
  ret float %tmp1
}

; CHECK-LABEL: @load_private_from_flat(
; CHECK-NEXT: %tmp0 = addrspacecast float* %generic_scalar to float addrspace(5)*
; CHECK-NEXT: %tmp1 = load float, float addrspace(5)* %tmp0
; CHECK-NEXT: ret float %tmp1
define float @load_private_from_flat(float* %generic_scalar) #0 {
  %tmp0 = addrspacecast float* %generic_scalar to float addrspace(5)*
  %tmp1 = load float, float addrspace(5)* %tmp0
  ret float %tmp1
}

; CHECK-LABEL: @store_global_from_flat(
; CHECK-NEXT: %tmp0 = addrspacecast float* %generic_scalar to float addrspace(1)*
; CHECK-NEXT: store float 0.000000e+00, float addrspace(1)* %tmp0
define amdgpu_kernel void @store_global_from_flat(float* %generic_scalar) #0 {
  %tmp0 = addrspacecast float* %generic_scalar to float addrspace(1)*
  store float 0.0, float addrspace(1)* %tmp0
  ret void
}

; CHECK-LABEL: @store_group_from_flat(
; CHECK-NEXT: %tmp0 = addrspacecast float* %generic_scalar to float addrspace(3)*
; CHECK-NEXT: store float 0.000000e+00, float addrspace(3)* %tmp0
define amdgpu_kernel void @store_group_from_flat(float* %generic_scalar) #0 {
  %tmp0 = addrspacecast float* %generic_scalar to float addrspace(3)*
  store float 0.0, float addrspace(3)* %tmp0
  ret void
}

; CHECK-LABEL: @store_private_from_flat(
; CHECK-NEXT: %tmp0 = addrspacecast float* %generic_scalar to float addrspace(5)*
; CHECK-NEXT: store float 0.000000e+00, float addrspace(5)* %tmp0
define amdgpu_kernel void @store_private_from_flat(float* %generic_scalar) #0 {
  %tmp0 = addrspacecast float* %generic_scalar to float addrspace(5)*
  store float 0.0, float addrspace(5)* %tmp0
  ret void
}

; optimized to global load/store.
; CHECK-LABEL: @load_store_global(
; CHECK-NEXT: %val = load i32, i32 addrspace(1)* %input, align 4
; CHECK-NEXT: store i32 %val, i32 addrspace(1)* %output, align 4
; CHECK-NEXT: ret void
define amdgpu_kernel void @load_store_global(i32 addrspace(1)* nocapture %input, i32 addrspace(1)* nocapture %output) #0 {
  %tmp0 = addrspacecast i32 addrspace(1)* %input to i32*
  %tmp1 = addrspacecast i32 addrspace(1)* %output to i32*
  %val = load i32, i32* %tmp0, align 4
  store i32 %val, i32* %tmp1, align 4
  ret void
}

; Optimized to group load/store.
; CHECK-LABEL: @load_store_group(
; CHECK-NEXT: %val = load i32, i32 addrspace(3)* %input, align 4
; CHECK-NEXT: store i32 %val, i32 addrspace(3)* %output, align 4
; CHECK-NEXT: ret void
define amdgpu_kernel void @load_store_group(i32 addrspace(3)* nocapture %input, i32 addrspace(3)* nocapture %output) #0 {
  %tmp0 = addrspacecast i32 addrspace(3)* %input to i32*
  %tmp1 = addrspacecast i32 addrspace(3)* %output to i32*
  %val = load i32, i32* %tmp0, align 4
  store i32 %val, i32* %tmp1, align 4
  ret void
}

; Optimized to private load/store.
; CHECK-LABEL: @load_store_private(
; CHECK-NEXT: %val = load i32, i32 addrspace(5)* %input, align 4
; CHECK-NEXT: store i32 %val, i32 addrspace(5)* %output, align 4
; CHECK-NEXT: ret void
define amdgpu_kernel void @load_store_private(i32 addrspace(5)* nocapture %input, i32 addrspace(5)* nocapture %output) #0 {
  %tmp0 = addrspacecast i32 addrspace(5)* %input to i32*
  %tmp1 = addrspacecast i32 addrspace(5)* %output to i32*
  %val = load i32, i32* %tmp0, align 4
  store i32 %val, i32* %tmp1, align 4
  ret void
}

; No optimization. flat load/store.
; CHECK-LABEL: @load_store_flat(
; CHECK-NEXT: %val = load i32, i32* %input, align 4
; CHECK-NEXT: store i32 %val, i32* %output, align 4
; CHECK-NEXT: ret void
define amdgpu_kernel void @load_store_flat(i32* nocapture %input, i32* nocapture %output) #0 {
  %val = load i32, i32* %input, align 4
  store i32 %val, i32* %output, align 4
  ret void
}

; CHECK-LABEL: @store_addrspacecast_ptr_value(
; CHECK: %cast = addrspacecast i32 addrspace(1)* %input to i32*
; CHECK-NEXT: store i32* %cast, i32* addrspace(1)* %output, align 4
define amdgpu_kernel void @store_addrspacecast_ptr_value(i32 addrspace(1)* nocapture %input, i32* addrspace(1)* nocapture %output) #0 {
  %cast = addrspacecast i32 addrspace(1)* %input to i32*
  store i32* %cast, i32* addrspace(1)* %output, align 4
  ret void
}

; CHECK-LABEL: @atomicrmw_add_global_to_flat(
; CHECK-NEXT: %ret = atomicrmw add i32 addrspace(1)* %global.ptr, i32 %y seq_cst
define i32 @atomicrmw_add_global_to_flat(i32 addrspace(1)* %global.ptr, i32 %y) #0 {
  %cast = addrspacecast i32 addrspace(1)* %global.ptr to i32*
  %ret = atomicrmw add i32* %cast, i32 %y seq_cst
  ret i32 %ret
}

; CHECK-LABEL: @atomicrmw_add_group_to_flat(
; CHECK-NEXT: %ret = atomicrmw add i32 addrspace(3)* %group.ptr, i32 %y seq_cst
define i32 @atomicrmw_add_group_to_flat(i32 addrspace(3)* %group.ptr, i32 %y) #0 {
  %cast = addrspacecast i32 addrspace(3)* %group.ptr to i32*
  %ret = atomicrmw add i32* %cast, i32 %y seq_cst
  ret i32 %ret
}

; CHECK-LABEL: @cmpxchg_global_to_flat(
; CHECK: %ret = cmpxchg i32 addrspace(1)* %global.ptr, i32 %cmp, i32 %val seq_cst monotonic
define { i32, i1 } @cmpxchg_global_to_flat(i32 addrspace(1)* %global.ptr, i32 %cmp, i32 %val) #0 {
  %cast = addrspacecast i32 addrspace(1)* %global.ptr to i32*
  %ret = cmpxchg i32* %cast, i32 %cmp, i32 %val seq_cst monotonic
  ret { i32, i1 } %ret
}

; CHECK-LABEL: @cmpxchg_group_to_flat(
; CHECK: %ret = cmpxchg i32 addrspace(3)* %group.ptr, i32 %cmp, i32 %val seq_cst monotonic
define { i32, i1 } @cmpxchg_group_to_flat(i32 addrspace(3)* %group.ptr, i32 %cmp, i32 %val) #0 {
  %cast = addrspacecast i32 addrspace(3)* %group.ptr to i32*
  %ret = cmpxchg i32* %cast, i32 %cmp, i32 %val seq_cst monotonic
  ret { i32, i1 } %ret
}

; Not pointer operand
; CHECK-LABEL: @cmpxchg_group_to_flat_wrong_operand(
; CHECK: %cast.cmp = addrspacecast i32 addrspace(3)* %cmp.ptr to i32*
; CHECK: %ret = cmpxchg i32* addrspace(3)* %cas.ptr, i32* %cast.cmp, i32* %val seq_cst monotonic
define { i32*, i1 } @cmpxchg_group_to_flat_wrong_operand(i32* addrspace(3)* %cas.ptr, i32 addrspace(3)* %cmp.ptr, i32* %val) #0 {
  %cast.cmp = addrspacecast i32 addrspace(3)* %cmp.ptr to i32*
  %ret = cmpxchg i32* addrspace(3)* %cas.ptr, i32* %cast.cmp, i32* %val seq_cst monotonic
  ret { i32*, i1 } %ret
}

; Null pointer in local addr space
; CHECK-LABEL: @local_nullptr
; CHECK: icmp ne i8 addrspace(3)* %a, addrspacecast (i8 addrspace(5)* null to i8 addrspace(3)*)
; CHECK-NOT: i8 addrspace(3)* null
define void @local_nullptr(i32 addrspace(1)* nocapture %results, i8 addrspace(3)* %a) {
entry:
  %tobool = icmp ne i8 addrspace(3)* %a, addrspacecast (i8 addrspace(5)* null to i8 addrspace(3)*)
  %conv = zext i1 %tobool to i32
  store i32 %conv, i32 addrspace(1)* %results, align 4
  ret void
}

attributes #0 = { nounwind }
