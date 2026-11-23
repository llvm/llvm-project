; RUN: opt -S -mtriple=amdgpu-amd-amdhsa -passes=infer-address-spaces %s | FileCheck %s

; Test that pure addrspacecast instructions not directly connected to
; a memory operation are inferred.

; CHECK-LABEL: @addrspacecast_gep_addrspacecast(
; CHECK: %gep0 = getelementptr i32, ptr addrspace(3) %ptr, i64 9
; CHECK-NEXT: store i32 8, ptr addrspace(3) %gep0, align 8
; CHECK-NEXT: ret void
define void @addrspacecast_gep_addrspacecast(ptr addrspace(3) %ptr) {
  %asc0 = addrspacecast ptr addrspace(3) %ptr to ptr
  %gep0 = getelementptr i32, ptr %asc0, i64 9
  %asc1 = addrspacecast ptr %gep0 to ptr addrspace(3)
  store i32 8, ptr addrspace(3) %asc1, align 8
  ret void
}

; CHECK-LABEL: @addrspacecast_different_pointee_type(
; CHECK: [[GEP:%.*]] = getelementptr i32, ptr addrspace(3) %ptr, i64 9
; CHECK-NEXT: store i8 8, ptr addrspace(3) [[GEP]], align 8
; CHECK-NEXT: ret void
define void @addrspacecast_different_pointee_type(ptr addrspace(3) %ptr) {
  %asc0 = addrspacecast ptr addrspace(3) %ptr to ptr
  %gep0 = getelementptr i32, ptr %asc0, i64 9
  %asc1 = addrspacecast ptr %gep0 to ptr addrspace(3)
  store i8 8, ptr addrspace(3) %asc1, align 8
  ret void
}

; CHECK-LABEL: @addrspacecast_to_memory(
; CHECK: %gep0 = getelementptr i32, ptr addrspace(3) %ptr, i64 9
; CHECK-NEXT: store volatile ptr addrspace(3) %gep0, ptr addrspace(1) poison
; CHECK-NEXT: ret void
define void @addrspacecast_to_memory(ptr addrspace(3) %ptr) {
  %asc0 = addrspacecast ptr addrspace(3) %ptr to ptr
  %gep0 = getelementptr i32, ptr %asc0, i64 9
  %asc1 = addrspacecast ptr %gep0 to ptr addrspace(3)
  store volatile ptr addrspace(3) %asc1, ptr addrspace(1) poison
  ret void
}

; CHECK-LABEL: @multiuse_addrspacecast_gep_addrspacecast(
; CHECK: %asc0 = addrspacecast ptr addrspace(3) %ptr to ptr
; CHECK-NEXT: store volatile ptr %asc0, ptr addrspace(1) poison
; CHECK-NEXT: %gep0 = getelementptr i32, ptr addrspace(3) %ptr, i64 9
; CHECK-NEXT: store i32 8, ptr addrspace(3) %gep0, align 8
; CHECK-NEXT: ret void
define void @multiuse_addrspacecast_gep_addrspacecast(ptr addrspace(3) %ptr) {
  %asc0 = addrspacecast ptr addrspace(3) %ptr to ptr
  store volatile ptr %asc0, ptr addrspace(1) poison
  %gep0 = getelementptr i32, ptr %asc0, i64 9
  %asc1 = addrspacecast ptr %gep0 to ptr addrspace(3)
  store i32 8, ptr addrspace(3) %asc1, align 8
  ret void
}

; nonnull is dropped when the cast is rebuilt on the sunk select operand.
; CHECK-LABEL: @rebuilt_addrspacecast_drops_nonnull(
; CHECK: %sel = select i1 %c, ptr addrspace(3) %p, ptr addrspace(3) %q
; CHECK-NEXT: %1 = addrspacecast ptr addrspace(3) %sel to ptr
; CHECK-NOT: nonnull
; CHECK-NEXT: call void @use_flat(ptr %1)
; CHECK-NEXT: %v = load i32, ptr addrspace(3) %sel, align 4
; CHECK-NEXT: ret i32 %v
define i32 @rebuilt_addrspacecast_drops_nonnull(i1 %c, ptr addrspace(3) %p, ptr addrspace(3) %q) {
  %pf = addrspacecast nonnull ptr addrspace(3) %p to ptr
  %qf = addrspacecast nonnull ptr addrspace(3) %q to ptr
  %sel = select i1 %c, ptr %pf, ptr %qf
  call void @use_flat(ptr %sel)
  %v = load i32, ptr %sel
  ret i32 %v
}

declare void @use_flat(ptr)
