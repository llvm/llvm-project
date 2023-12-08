; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -passes=infer-address-spaces %s | FileCheck %s

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
; CHECK-NEXT: store volatile ptr addrspace(3) %gep0, ptr addrspace(1) undef
; CHECK-NEXT: ret void
define void @addrspacecast_to_memory(ptr addrspace(3) %ptr) {
  %asc0 = addrspacecast ptr addrspace(3) %ptr to ptr
  %gep0 = getelementptr i32, ptr %asc0, i64 9
  %asc1 = addrspacecast ptr %gep0 to ptr addrspace(3)
  store volatile ptr addrspace(3) %asc1, ptr addrspace(1) undef
  ret void
}

; CHECK-LABEL: @multiuse_addrspacecast_gep_addrspacecast(
; CHECK: %asc0 = addrspacecast ptr addrspace(3) %ptr to ptr
; CHECK-NEXT: store volatile ptr %asc0, ptr addrspace(1) undef
; CHECK-NEXT: %gep0 = getelementptr i32, ptr addrspace(3) %ptr, i64 9
; CHECK-NEXT: store i32 8, ptr addrspace(3) %gep0, align 8
; CHECK-NEXT: ret void
define void @multiuse_addrspacecast_gep_addrspacecast(ptr addrspace(3) %ptr) {
  %asc0 = addrspacecast ptr addrspace(3) %ptr to ptr
  store volatile ptr %asc0, ptr addrspace(1) undef
  %gep0 = getelementptr i32, ptr %asc0, i64 9
  %asc1 = addrspacecast ptr %gep0 to ptr addrspace(3)
  store i32 8, ptr addrspace(3) %asc1, align 8
  ret void
}
