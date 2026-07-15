; RUN: opt < %s -passes=amdgpu-sw-lower-lds -amdgpu-asan-instrument-lds=false -S -mtriple=amdgcn-amd-amdhsa | FileCheck %s

; A non-kernel receives lowered LDS storage as a flat (generic) pointer and
; re-derives an LDS pointer via a flat -> addrspace(3) round-trip. After SW LDS
; lowering the storage lives in global memory, so the addrspace(3) access would
; hit dead LDS. The round-trip must be collapsed into a flat access on the
; incoming argument, which still points at the global backing buffer. Check the
; store, load, atomicrmw and cmpxchg cases.

@lds_var = internal addrspace(3) global [64 x i32] poison, align 4

define void @store_case(ptr %storage) sanitize_address {
; CHECK-LABEL: define void @store_case(
; CHECK-SAME: ptr [[STORAGE:%.*]]){{.*}} {
; CHECK-NOT:     addrspace(3)
; CHECK:         store i32 42, ptr [[STORAGE]], align 4
; CHECK-NOT:     addrspace(3)
; CHECK:         ret void
  %cast = addrspacecast ptr %storage to ptr addrspace(3)
  store i32 42, ptr addrspace(3) %cast, align 4
  ret void
}

define i32 @load_case(ptr %storage) sanitize_address {
; CHECK-LABEL: define i32 @load_case(
; CHECK-SAME: ptr [[STORAGE:%.*]]){{.*}} {
; CHECK-NOT:     addrspace(3)
; CHECK:         [[VAL:%.*]] = load i32, ptr [[STORAGE]], align 4
; CHECK-NOT:     addrspace(3)
; CHECK:         ret i32 [[VAL]]
  %cast = addrspacecast ptr %storage to ptr addrspace(3)
  %val = load i32, ptr addrspace(3) %cast, align 4
  ret i32 %val
}

define i32 @atomicrmw_case(ptr %storage) sanitize_address {
; CHECK-LABEL: define i32 @atomicrmw_case(
; CHECK-SAME: ptr [[STORAGE:%.*]]){{.*}} {
; CHECK-NOT:     addrspace(3)
; CHECK:         [[OLD:%.*]] = atomicrmw add ptr [[STORAGE]], i32 1 seq_cst, align 4
; CHECK-NOT:     addrspace(3)
; CHECK:         ret i32 [[OLD]]
  %cast = addrspacecast ptr %storage to ptr addrspace(3)
  %old = atomicrmw add ptr addrspace(3) %cast, i32 1 seq_cst, align 4
  ret i32 %old
}

define i32 @cmpxchg_case(ptr %storage) sanitize_address {
; CHECK-LABEL: define i32 @cmpxchg_case(
; CHECK-SAME: ptr [[STORAGE:%.*]]){{.*}} {
; CHECK-NOT:     addrspace(3)
; CHECK:         [[RES:%.*]] = cmpxchg ptr [[STORAGE]], i32 0, i32 1 seq_cst seq_cst, align 4
; CHECK-NOT:     addrspace(3)
; CHECK:         [[VAL:%.*]] = extractvalue { i32, i1 } [[RES]], 0
; CHECK:         ret i32 [[VAL]]
  %cast = addrspacecast ptr %storage to ptr addrspace(3)
  %res = cmpxchg ptr addrspace(3) %cast, i32 0, i32 1 seq_cst seq_cst, align 4
  %val = extractvalue { i32, i1 } %res, 0
  ret i32 %val
}

define amdgpu_kernel void @kernel() sanitize_address {
; CHECK-LABEL: define amdgpu_kernel void @kernel(
  %flat = addrspacecast ptr addrspace(3) @lds_var to ptr
  call void @store_case(ptr %flat)
  %l = call i32 @load_case(ptr %flat)
  %a = call i32 @atomicrmw_case(ptr %flat)
  %c = call i32 @cmpxchg_case(ptr %flat)
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 4, !"nosanitize_address", i32 1}
