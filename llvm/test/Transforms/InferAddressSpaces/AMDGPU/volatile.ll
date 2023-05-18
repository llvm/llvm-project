; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -passes=infer-address-spaces %s | FileCheck %s

; Check that volatile users of addrspacecast are not replaced.

; CHECK-LABEL: @volatile_load_flat_from_global(
; CHECK: load volatile i32, ptr
; CHECK: store i32 %val, ptr addrspace(1)
define amdgpu_kernel void @volatile_load_flat_from_global(ptr addrspace(1) nocapture %input, ptr addrspace(1) nocapture %output) #0 {
  %tmp0 = addrspacecast ptr addrspace(1) %input to ptr
  %tmp1 = addrspacecast ptr addrspace(1) %output to ptr
  %val = load volatile i32, ptr %tmp0, align 4
  store i32 %val, ptr %tmp1, align 4
  ret void
}

; CHECK-LABEL: @volatile_load_flat_from_constant(
; CHECK: load volatile i32, ptr
; CHECK: store i32 %val, ptr addrspace(1)
define amdgpu_kernel void @volatile_load_flat_from_constant(ptr addrspace(4) nocapture %input, ptr addrspace(1) nocapture %output) #0 {
  %tmp0 = addrspacecast ptr addrspace(4) %input to ptr
  %tmp1 = addrspacecast ptr addrspace(1) %output to ptr
  %val = load volatile i32, ptr %tmp0, align 4
  store i32 %val, ptr %tmp1, align 4
  ret void
}

; CHECK-LABEL: @volatile_load_flat_from_group(
; CHECK: load volatile i32, ptr
; CHECK: store i32 %val, ptr addrspace(3)
define amdgpu_kernel void @volatile_load_flat_from_group(ptr addrspace(3) nocapture %input, ptr addrspace(3) nocapture %output) #0 {
  %tmp0 = addrspacecast ptr addrspace(3) %input to ptr
  %tmp1 = addrspacecast ptr addrspace(3) %output to ptr
  %val = load volatile i32, ptr %tmp0, align 4
  store i32 %val, ptr %tmp1, align 4
  ret void
}

; CHECK-LABEL: @volatile_load_flat_from_private(
; CHECK: load volatile i32, ptr
; CHECK: store i32 %val, ptr addrspace(5)
define amdgpu_kernel void @volatile_load_flat_from_private(ptr addrspace(5) nocapture %input, ptr addrspace(5) nocapture %output) #0 {
  %tmp0 = addrspacecast ptr addrspace(5) %input to ptr
  %tmp1 = addrspacecast ptr addrspace(5) %output to ptr
  %val = load volatile i32, ptr %tmp0, align 4
  store i32 %val, ptr %tmp1, align 4
  ret void
}

; CHECK-LABEL: @volatile_store_flat_to_global(
; CHECK: load i32, ptr addrspace(1)
; CHECK: store volatile i32 %val, ptr
define amdgpu_kernel void @volatile_store_flat_to_global(ptr addrspace(1) nocapture %input, ptr addrspace(1) nocapture %output) #0 {
  %tmp0 = addrspacecast ptr addrspace(1) %input to ptr
  %tmp1 = addrspacecast ptr addrspace(1) %output to ptr
  %val = load i32, ptr %tmp0, align 4
  store volatile i32 %val, ptr %tmp1, align 4
  ret void
}

; CHECK-LABEL: @volatile_store_flat_to_group(
; CHECK: load i32, ptr addrspace(3)
; CHECK: store volatile i32 %val, ptr
define amdgpu_kernel void @volatile_store_flat_to_group(ptr addrspace(3) nocapture %input, ptr addrspace(3) nocapture %output) #0 {
  %tmp0 = addrspacecast ptr addrspace(3) %input to ptr
  %tmp1 = addrspacecast ptr addrspace(3) %output to ptr
  %val = load i32, ptr %tmp0, align 4
  store volatile i32 %val, ptr %tmp1, align 4
  ret void
}

; CHECK-LABEL: @volatile_store_flat_to_private(
; CHECK: load i32, ptr addrspace(5)
; CHECK: store volatile i32 %val, ptr
define amdgpu_kernel void @volatile_store_flat_to_private(ptr addrspace(5) nocapture %input, ptr addrspace(5) nocapture %output) #0 {
  %tmp0 = addrspacecast ptr addrspace(5) %input to ptr
  %tmp1 = addrspacecast ptr addrspace(5) %output to ptr
  %val = load i32, ptr %tmp0, align 4
  store volatile i32 %val, ptr %tmp1, align 4
  ret void
}

; CHECK-LABEL: @volatile_atomicrmw_add_group_to_flat(
; CHECK: addrspacecast ptr addrspace(3) %group.ptr to ptr
; CHECK: atomicrmw volatile add ptr
define i32 @volatile_atomicrmw_add_group_to_flat(ptr addrspace(3) %group.ptr, i32 %y) #0 {
  %cast = addrspacecast ptr addrspace(3) %group.ptr to ptr
  %ret = atomicrmw volatile add ptr %cast, i32 %y seq_cst
  ret i32 %ret
}

; CHECK-LABEL: @volatile_atomicrmw_add_global_to_flat(
; CHECK: addrspacecast ptr addrspace(1) %global.ptr to ptr
; CHECK: %ret = atomicrmw volatile add ptr
define i32 @volatile_atomicrmw_add_global_to_flat(ptr addrspace(1) %global.ptr, i32 %y) #0 {
  %cast = addrspacecast ptr addrspace(1) %global.ptr to ptr
  %ret = atomicrmw volatile add ptr %cast, i32 %y seq_cst
  ret i32 %ret
}

; CHECK-LABEL: @volatile_cmpxchg_global_to_flat(
; CHECK: addrspacecast ptr addrspace(1) %global.ptr to ptr
; CHECK: cmpxchg volatile ptr
define { i32, i1 } @volatile_cmpxchg_global_to_flat(ptr addrspace(1) %global.ptr, i32 %cmp, i32 %val) #0 {
  %cast = addrspacecast ptr addrspace(1) %global.ptr to ptr
  %ret = cmpxchg volatile ptr %cast, i32 %cmp, i32 %val seq_cst monotonic
  ret { i32, i1 } %ret
}

; CHECK-LABEL: @volatile_cmpxchg_group_to_flat(
; CHECK: addrspacecast ptr addrspace(3) %group.ptr to ptr
; CHECK: cmpxchg volatile ptr
define { i32, i1 } @volatile_cmpxchg_group_to_flat(ptr addrspace(3) %group.ptr, i32 %cmp, i32 %val) #0 {
  %cast = addrspacecast ptr addrspace(3) %group.ptr to ptr
  %ret = cmpxchg volatile ptr %cast, i32 %cmp, i32 %val seq_cst monotonic
  ret { i32, i1 } %ret
}

; CHECK-LABEL: @volatile_memset_group_to_flat(
; CHECK: %cast = addrspacecast ptr addrspace(3) %group.ptr to ptr
; CHECK: call void @llvm.memset.p0.i64(ptr align 4 %cast, i8 4, i64 32, i1 true)
define amdgpu_kernel void @volatile_memset_group_to_flat(ptr addrspace(3) %group.ptr, i32 %y) #0 {
  %cast = addrspacecast ptr addrspace(3) %group.ptr to ptr
  call void @llvm.memset.p0.i64(ptr align 4 %cast, i8 4, i64 32, i1 true)
  ret void
}

; CHECK-LABEL: @volatile_memset_global_to_flat(
; CHECK: %cast = addrspacecast ptr addrspace(1) %global.ptr to ptr
; CHECK: call void @llvm.memset.p0.i64(ptr align 4 %cast, i8 4, i64 32, i1 true)
define amdgpu_kernel void @volatile_memset_global_to_flat(ptr addrspace(1) %global.ptr, i32 %y) #0 {
  %cast = addrspacecast ptr addrspace(1) %global.ptr to ptr
  call void @llvm.memset.p0.i64(ptr align 4 %cast, i8 4, i64 32, i1 true)
  ret void
}

declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1) #1

attributes #0 = { nounwind }
attributes #1 = { argmemonly nounwind }
