; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -infer-address-spaces %s | FileCheck %s

; Check that volatile users of addrspacecast are not replaced.

; CHECK-LABEL: @volatile_load_flat_from_global(
; CHECK: load volatile i32, i32*
; CHECK: store i32 %val, i32 addrspace(1)*
define amdgpu_kernel void @volatile_load_flat_from_global(i32 addrspace(1)* nocapture %input, i32 addrspace(1)* nocapture %output) #0 {
  %tmp0 = addrspacecast i32 addrspace(1)* %input to i32*
  %tmp1 = addrspacecast i32 addrspace(1)* %output to i32*
  %val = load volatile i32, i32* %tmp0, align 4
  store i32 %val, i32* %tmp1, align 4
  ret void
}

; CHECK-LABEL: @volatile_load_flat_from_constant(
; CHECK: load volatile i32, i32*
; CHECK: store i32 %val, i32 addrspace(1)*
define amdgpu_kernel void @volatile_load_flat_from_constant(i32 addrspace(4)* nocapture %input, i32 addrspace(1)* nocapture %output) #0 {
  %tmp0 = addrspacecast i32 addrspace(4)* %input to i32*
  %tmp1 = addrspacecast i32 addrspace(1)* %output to i32*
  %val = load volatile i32, i32* %tmp0, align 4
  store i32 %val, i32* %tmp1, align 4
  ret void
}

; CHECK-LABEL: @volatile_load_flat_from_group(
; CHECK: load volatile i32, i32*
; CHECK: store i32 %val, i32 addrspace(3)*
define amdgpu_kernel void @volatile_load_flat_from_group(i32 addrspace(3)* nocapture %input, i32 addrspace(3)* nocapture %output) #0 {
  %tmp0 = addrspacecast i32 addrspace(3)* %input to i32*
  %tmp1 = addrspacecast i32 addrspace(3)* %output to i32*
  %val = load volatile i32, i32* %tmp0, align 4
  store i32 %val, i32* %tmp1, align 4
  ret void
}

; CHECK-LABEL: @volatile_load_flat_from_private(
; CHECK: load volatile i32, i32*
; CHECK: store i32 %val, i32 addrspace(5)*
define amdgpu_kernel void @volatile_load_flat_from_private(i32 addrspace(5)* nocapture %input, i32 addrspace(5)* nocapture %output) #0 {
  %tmp0 = addrspacecast i32 addrspace(5)* %input to i32*
  %tmp1 = addrspacecast i32 addrspace(5)* %output to i32*
  %val = load volatile i32, i32* %tmp0, align 4
  store i32 %val, i32* %tmp1, align 4
  ret void
}

; CHECK-LABEL: @volatile_store_flat_to_global(
; CHECK: load i32, i32 addrspace(1)*
; CHECK: store volatile i32 %val, i32*
define amdgpu_kernel void @volatile_store_flat_to_global(i32 addrspace(1)* nocapture %input, i32 addrspace(1)* nocapture %output) #0 {
  %tmp0 = addrspacecast i32 addrspace(1)* %input to i32*
  %tmp1 = addrspacecast i32 addrspace(1)* %output to i32*
  %val = load i32, i32* %tmp0, align 4
  store volatile i32 %val, i32* %tmp1, align 4
  ret void
}

; CHECK-LABEL: @volatile_store_flat_to_group(
; CHECK: load i32, i32 addrspace(3)*
; CHECK: store volatile i32 %val, i32*
define amdgpu_kernel void @volatile_store_flat_to_group(i32 addrspace(3)* nocapture %input, i32 addrspace(3)* nocapture %output) #0 {
  %tmp0 = addrspacecast i32 addrspace(3)* %input to i32*
  %tmp1 = addrspacecast i32 addrspace(3)* %output to i32*
  %val = load i32, i32* %tmp0, align 4
  store volatile i32 %val, i32* %tmp1, align 4
  ret void
}

; CHECK-LABEL: @volatile_store_flat_to_private(
; CHECK: load i32, i32 addrspace(5)*
; CHECK: store volatile i32 %val, i32*
define amdgpu_kernel void @volatile_store_flat_to_private(i32 addrspace(5)* nocapture %input, i32 addrspace(5)* nocapture %output) #0 {
  %tmp0 = addrspacecast i32 addrspace(5)* %input to i32*
  %tmp1 = addrspacecast i32 addrspace(5)* %output to i32*
  %val = load i32, i32* %tmp0, align 4
  store volatile i32 %val, i32* %tmp1, align 4
  ret void
}

; CHECK-LABEL: @volatile_atomicrmw_add_group_to_flat(
; CHECK: addrspacecast i32 addrspace(3)* %group.ptr to i32*
; CHECK: atomicrmw volatile add i32*
define i32 @volatile_atomicrmw_add_group_to_flat(i32 addrspace(3)* %group.ptr, i32 %y) #0 {
  %cast = addrspacecast i32 addrspace(3)* %group.ptr to i32*
  %ret = atomicrmw volatile add i32* %cast, i32 %y seq_cst
  ret i32 %ret
}

; CHECK-LABEL: @volatile_atomicrmw_add_global_to_flat(
; CHECK: addrspacecast i32 addrspace(1)* %global.ptr to i32*
; CHECK: %ret = atomicrmw volatile add i32*
define i32 @volatile_atomicrmw_add_global_to_flat(i32 addrspace(1)* %global.ptr, i32 %y) #0 {
  %cast = addrspacecast i32 addrspace(1)* %global.ptr to i32*
  %ret = atomicrmw volatile add i32* %cast, i32 %y seq_cst
  ret i32 %ret
}

; CHECK-LABEL: @volatile_cmpxchg_global_to_flat(
; CHECK: addrspacecast i32 addrspace(1)* %global.ptr to i32*
; CHECK: cmpxchg volatile i32*
define { i32, i1 } @volatile_cmpxchg_global_to_flat(i32 addrspace(1)* %global.ptr, i32 %cmp, i32 %val) #0 {
  %cast = addrspacecast i32 addrspace(1)* %global.ptr to i32*
  %ret = cmpxchg volatile i32* %cast, i32 %cmp, i32 %val seq_cst monotonic
  ret { i32, i1 } %ret
}

; CHECK-LABEL: @volatile_cmpxchg_group_to_flat(
; CHECK: addrspacecast i32 addrspace(3)* %group.ptr to i32*
; CHECK: cmpxchg volatile i32*
define { i32, i1 } @volatile_cmpxchg_group_to_flat(i32 addrspace(3)* %group.ptr, i32 %cmp, i32 %val) #0 {
  %cast = addrspacecast i32 addrspace(3)* %group.ptr to i32*
  %ret = cmpxchg volatile i32* %cast, i32 %cmp, i32 %val seq_cst monotonic
  ret { i32, i1 } %ret
}

; FIXME: Shouldn't be losing names
; CHECK-LABEL: @volatile_memset_group_to_flat(
; CHECK: addrspacecast i8 addrspace(3)* %group.ptr to i8*
; CHECK: call void @llvm.memset.p0i8.i64(i8* align 4 %1, i8 4, i64 32, i1 true)
define amdgpu_kernel void @volatile_memset_group_to_flat(i8 addrspace(3)* %group.ptr, i32 %y) #0 {
  %cast = addrspacecast i8 addrspace(3)* %group.ptr to i8*
  call void @llvm.memset.p0i8.i64(i8* align 4 %cast, i8 4, i64 32, i1 true)
  ret void
}

; CHECK-LABEL: @volatile_memset_global_to_flat(
; CHECK: addrspacecast i8 addrspace(1)* %global.ptr to i8*
; CHECK: call void @llvm.memset.p0i8.i64(i8* align 4 %1, i8 4, i64 32, i1 true)
define amdgpu_kernel void @volatile_memset_global_to_flat(i8 addrspace(1)* %global.ptr, i32 %y) #0 {
  %cast = addrspacecast i8 addrspace(1)* %global.ptr to i8*
  call void @llvm.memset.p0i8.i64(i8* align 4 %cast, i8 4, i64 32, i1 true)
  ret void
}

declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1) #1

attributes #0 = { nounwind }
attributes #1 = { argmemonly nounwind }
