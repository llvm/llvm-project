; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -passes=infer-address-spaces %s | FileCheck %s

; Trivial optimization of generic addressing

; CHECK-LABEL: @load_global_from_flat(
; CHECK-NEXT: %tmp0 = addrspacecast ptr %generic_scalar to ptr addrspace(1)
; CHECK-NEXT: %tmp1 = load float, ptr addrspace(1) %tmp0
; CHECK-NEXT: ret float %tmp1
define float @load_global_from_flat(ptr %generic_scalar) #0 {
  %tmp0 = addrspacecast ptr %generic_scalar to ptr addrspace(1)
  %tmp1 = load float, ptr addrspace(1) %tmp0
  ret float %tmp1
}

; CHECK-LABEL: @load_constant_from_flat(
; CHECK-NEXT: %tmp0 = addrspacecast ptr %generic_scalar to ptr addrspace(4)
; CHECK-NEXT: %tmp1 = load float, ptr addrspace(4) %tmp0
; CHECK-NEXT: ret float %tmp1
define float @load_constant_from_flat(ptr %generic_scalar) #0 {
  %tmp0 = addrspacecast ptr %generic_scalar to ptr addrspace(4)
  %tmp1 = load float, ptr addrspace(4) %tmp0
  ret float %tmp1
}

; CHECK-LABEL: @load_group_from_flat(
; CHECK-NEXT: %tmp0 = addrspacecast ptr %generic_scalar to ptr addrspace(3)
; CHECK-NEXT: %tmp1 = load float, ptr addrspace(3) %tmp0
; CHECK-NEXT: ret float %tmp1
define float @load_group_from_flat(ptr %generic_scalar) #0 {
  %tmp0 = addrspacecast ptr %generic_scalar to ptr addrspace(3)
  %tmp1 = load float, ptr addrspace(3) %tmp0
  ret float %tmp1
}

; CHECK-LABEL: @load_private_from_flat(
; CHECK-NEXT: %tmp0 = addrspacecast ptr %generic_scalar to ptr addrspace(5)
; CHECK-NEXT: %tmp1 = load float, ptr addrspace(5) %tmp0
; CHECK-NEXT: ret float %tmp1
define float @load_private_from_flat(ptr %generic_scalar) #0 {
  %tmp0 = addrspacecast ptr %generic_scalar to ptr addrspace(5)
  %tmp1 = load float, ptr addrspace(5) %tmp0
  ret float %tmp1
}

; CHECK-LABEL: @store_global_from_flat(
; CHECK-NEXT: %tmp0 = addrspacecast ptr %generic_scalar to ptr addrspace(1)
; CHECK-NEXT: store float 0.000000e+00, ptr addrspace(1) %tmp0
define amdgpu_kernel void @store_global_from_flat(ptr %generic_scalar) #0 {
  %tmp0 = addrspacecast ptr %generic_scalar to ptr addrspace(1)
  store float 0.0, ptr addrspace(1) %tmp0
  ret void
}

; CHECK-LABEL: @store_group_from_flat(
; CHECK-NEXT: %tmp0 = addrspacecast ptr %generic_scalar to ptr addrspace(3)
; CHECK-NEXT: store float 0.000000e+00, ptr addrspace(3) %tmp0
define amdgpu_kernel void @store_group_from_flat(ptr %generic_scalar) #0 {
  %tmp0 = addrspacecast ptr %generic_scalar to ptr addrspace(3)
  store float 0.0, ptr addrspace(3) %tmp0
  ret void
}

; CHECK-LABEL: @store_private_from_flat(
; CHECK-NEXT: %tmp0 = addrspacecast ptr %generic_scalar to ptr addrspace(5)
; CHECK-NEXT: store float 0.000000e+00, ptr addrspace(5) %tmp0
define amdgpu_kernel void @store_private_from_flat(ptr %generic_scalar) #0 {
  %tmp0 = addrspacecast ptr %generic_scalar to ptr addrspace(5)
  store float 0.0, ptr addrspace(5) %tmp0
  ret void
}

; optimized to global load/store.
; CHECK-LABEL: @load_store_global(
; CHECK-NEXT: %val = load i32, ptr addrspace(1) %input, align 4
; CHECK-NEXT: store i32 %val, ptr addrspace(1) %output, align 4
; CHECK-NEXT: ret void
define amdgpu_kernel void @load_store_global(ptr addrspace(1) nocapture %input, ptr addrspace(1) nocapture %output) #0 {
  %tmp0 = addrspacecast ptr addrspace(1) %input to ptr
  %tmp1 = addrspacecast ptr addrspace(1) %output to ptr
  %val = load i32, ptr %tmp0, align 4
  store i32 %val, ptr %tmp1, align 4
  ret void
}

; Optimized to group load/store.
; CHECK-LABEL: @load_store_group(
; CHECK-NEXT: %val = load i32, ptr addrspace(3) %input, align 4
; CHECK-NEXT: store i32 %val, ptr addrspace(3) %output, align 4
; CHECK-NEXT: ret void
define amdgpu_kernel void @load_store_group(ptr addrspace(3) nocapture %input, ptr addrspace(3) nocapture %output) #0 {
  %tmp0 = addrspacecast ptr addrspace(3) %input to ptr
  %tmp1 = addrspacecast ptr addrspace(3) %output to ptr
  %val = load i32, ptr %tmp0, align 4
  store i32 %val, ptr %tmp1, align 4
  ret void
}

; Optimized to private load/store.
; CHECK-LABEL: @load_store_private(
; CHECK-NEXT: %val = load i32, ptr addrspace(5) %input, align 4
; CHECK-NEXT: store i32 %val, ptr addrspace(5) %output, align 4
; CHECK-NEXT: ret void
define amdgpu_kernel void @load_store_private(ptr addrspace(5) nocapture %input, ptr addrspace(5) nocapture %output) #0 {
  %tmp0 = addrspacecast ptr addrspace(5) %input to ptr
  %tmp1 = addrspacecast ptr addrspace(5) %output to ptr
  %val = load i32, ptr %tmp0, align 4
  store i32 %val, ptr %tmp1, align 4
  ret void
}

; No optimization. flat load/store.
; CHECK-LABEL: @load_store_flat(
; CHECK-NEXT: %val = load i32, ptr %input, align 4
; CHECK-NEXT: store i32 %val, ptr %output, align 4
; CHECK-NEXT: ret void
define amdgpu_kernel void @load_store_flat(ptr nocapture %input, ptr nocapture %output) #0 {
  %val = load i32, ptr %input, align 4
  store i32 %val, ptr %output, align 4
  ret void
}

; CHECK-LABEL: @store_addrspacecast_ptr_value(
; CHECK: %cast = addrspacecast ptr addrspace(1) %input to ptr
; CHECK-NEXT: store ptr %cast, ptr addrspace(1) %output, align 4
define amdgpu_kernel void @store_addrspacecast_ptr_value(ptr addrspace(1) nocapture %input, ptr addrspace(1) nocapture %output) #0 {
  %cast = addrspacecast ptr addrspace(1) %input to ptr
  store ptr %cast, ptr addrspace(1) %output, align 4
  ret void
}

; CHECK-LABEL: @atomicrmw_add_global_to_flat(
; CHECK-NEXT: %ret = atomicrmw add ptr addrspace(1) %global.ptr, i32 %y seq_cst
define i32 @atomicrmw_add_global_to_flat(ptr addrspace(1) %global.ptr, i32 %y) #0 {
  %cast = addrspacecast ptr addrspace(1) %global.ptr to ptr
  %ret = atomicrmw add ptr %cast, i32 %y seq_cst
  ret i32 %ret
}

; CHECK-LABEL: @atomicrmw_add_group_to_flat(
; CHECK-NEXT: %ret = atomicrmw add ptr addrspace(3) %group.ptr, i32 %y seq_cst
define i32 @atomicrmw_add_group_to_flat(ptr addrspace(3) %group.ptr, i32 %y) #0 {
  %cast = addrspacecast ptr addrspace(3) %group.ptr to ptr
  %ret = atomicrmw add ptr %cast, i32 %y seq_cst
  ret i32 %ret
}

; CHECK-LABEL: @cmpxchg_global_to_flat(
; CHECK: %ret = cmpxchg ptr addrspace(1) %global.ptr, i32 %cmp, i32 %val seq_cst monotonic
define { i32, i1 } @cmpxchg_global_to_flat(ptr addrspace(1) %global.ptr, i32 %cmp, i32 %val) #0 {
  %cast = addrspacecast ptr addrspace(1) %global.ptr to ptr
  %ret = cmpxchg ptr %cast, i32 %cmp, i32 %val seq_cst monotonic
  ret { i32, i1 } %ret
}

; CHECK-LABEL: @cmpxchg_group_to_flat(
; CHECK: %ret = cmpxchg ptr addrspace(3) %group.ptr, i32 %cmp, i32 %val seq_cst monotonic
define { i32, i1 } @cmpxchg_group_to_flat(ptr addrspace(3) %group.ptr, i32 %cmp, i32 %val) #0 {
  %cast = addrspacecast ptr addrspace(3) %group.ptr to ptr
  %ret = cmpxchg ptr %cast, i32 %cmp, i32 %val seq_cst monotonic
  ret { i32, i1 } %ret
}

; Not pointer operand
; CHECK-LABEL: @cmpxchg_group_to_flat_wrong_operand(
; CHECK: %cast.cmp = addrspacecast ptr addrspace(3) %cmp.ptr to ptr
; CHECK: %ret = cmpxchg ptr addrspace(3) %cas.ptr, ptr %cast.cmp, ptr %val seq_cst monotonic
define { ptr, i1 } @cmpxchg_group_to_flat_wrong_operand(ptr addrspace(3) %cas.ptr, ptr addrspace(3) %cmp.ptr, ptr %val) #0 {
  %cast.cmp = addrspacecast ptr addrspace(3) %cmp.ptr to ptr
  %ret = cmpxchg ptr addrspace(3) %cas.ptr, ptr %cast.cmp, ptr %val seq_cst monotonic
  ret { ptr, i1 } %ret
}

; Null pointer in local addr space
; CHECK-LABEL: @local_nullptr
; CHECK: icmp ne ptr addrspace(3) %a, addrspacecast (ptr addrspace(5) null to ptr addrspace(3))
; CHECK-NOT: ptr addrspace(3) null
define void @local_nullptr(ptr addrspace(1) nocapture %results, ptr addrspace(3) %a) {
entry:
  %tobool = icmp ne ptr addrspace(3) %a, addrspacecast (ptr addrspace(5) null to ptr addrspace(3))
  %conv = zext i1 %tobool to i32
  store i32 %conv, ptr addrspace(1) %results, align 4
  ret void
}

attributes #0 = { nounwind }
