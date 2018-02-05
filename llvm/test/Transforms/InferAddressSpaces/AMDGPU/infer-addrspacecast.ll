; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -infer-address-spaces %s | FileCheck %s

; Test that pure addrspacecast instructions not directly connected to
; a memory operation are inferred.

; CHECK-LABEL: @addrspacecast_gep_addrspacecast(
; CHECK: %gep0 = getelementptr i32, i32 addrspace(3)* %ptr, i64 9
; CHECK-NEXT: store i32 8, i32 addrspace(3)* %gep0, align 8
; CHECK-NEXT: ret void
define void @addrspacecast_gep_addrspacecast(i32 addrspace(3)* %ptr) {
  %asc0 = addrspacecast i32 addrspace(3)* %ptr to i32*
  %gep0 = getelementptr i32, i32* %asc0, i64 9
  %asc1 = addrspacecast i32* %gep0 to i32 addrspace(3)*
  store i32 8, i32 addrspace(3)* %asc1, align 8
  ret void
}

; CHECK-LABEL: @addrspacecast_different_pointee_type(
; CHECK: [[GEP:%.*]] = getelementptr i32, i32 addrspace(3)* %ptr, i64 9
; CHECK: [[CAST:%.*]] = bitcast i32 addrspace(3)* [[GEP]] to i8 addrspace(3)*
; CHECK-NEXT: store i8 8, i8 addrspace(3)* [[CAST]], align 8
; CHECK-NEXT: ret void
define void @addrspacecast_different_pointee_type(i32 addrspace(3)* %ptr) {
  %asc0 = addrspacecast i32 addrspace(3)* %ptr to i32*
  %gep0 = getelementptr i32, i32* %asc0, i64 9
  %asc1 = addrspacecast i32* %gep0 to i8 addrspace(3)*
  store i8 8, i8 addrspace(3)* %asc1, align 8
  ret void
}

; CHECK-LABEL: @addrspacecast_to_memory(
; CHECK: %gep0 = getelementptr i32, i32 addrspace(3)* %ptr, i64 9
; CHECK-NEXT: store volatile i32 addrspace(3)* %gep0, i32 addrspace(3)* addrspace(1)* undef
; CHECK-NEXT: ret void
define void @addrspacecast_to_memory(i32 addrspace(3)* %ptr) {
  %asc0 = addrspacecast i32 addrspace(3)* %ptr to i32*
  %gep0 = getelementptr i32, i32* %asc0, i64 9
  %asc1 = addrspacecast i32* %gep0 to i32 addrspace(3)*
  store volatile i32 addrspace(3)* %asc1, i32 addrspace(3)* addrspace(1)* undef
  ret void
}

; CHECK-LABEL: @multiuse_addrspacecast_gep_addrspacecast(
; CHECK: %1 = addrspacecast i32 addrspace(3)* %ptr to i32*
; CHECK-NEXT: store volatile i32* %1, i32* addrspace(1)* undef
; CHECK-NEXT: %gep0 = getelementptr i32, i32 addrspace(3)* %ptr, i64 9
; CHECK-NEXT: store i32 8, i32 addrspace(3)* %gep0, align 8
; CHECK-NEXT: ret void
define void @multiuse_addrspacecast_gep_addrspacecast(i32 addrspace(3)* %ptr) {
  %asc0 = addrspacecast i32 addrspace(3)* %ptr to i32*
  store volatile i32* %asc0, i32* addrspace(1)* undef
  %gep0 = getelementptr i32, i32* %asc0, i64 9
  %asc1 = addrspacecast i32* %gep0 to i32 addrspace(3)*
  store i32 8, i32 addrspace(3)* %asc1, align 8
  ret void
}
