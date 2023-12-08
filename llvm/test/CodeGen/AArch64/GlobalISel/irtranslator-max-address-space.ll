; RUN: llc -O0 -mtriple=aarch64-apple-ios -global-isel -stop-after=irtranslator %s -o - | FileCheck %s

; CHECK-LABEL: name: store_max_address_space
; CHECK: %0:_(p16777215) = COPY $x0
; CHECK: G_STORE %1(s32), %0(p16777215) :: (store (s32) into %ir.ptr, addrspace 16777215)
define void @store_max_address_space(ptr addrspace(16777215) %ptr) {
  store i32 0, ptr addrspace(16777215) %ptr
  ret void
}

; CHECK-LABEL: name: store_max_address_space_vector
; CHECK: %0:_(<2 x p16777215>) = COPY $q0
; CHECK: %1:_(p16777215) = G_EXTRACT_VECTOR_ELT %0(<2 x p16777215>), %2(s64)
; CHECK: %1(p16777215) :: (store (s32) into %ir.elt0, addrspace 16777215)
define void @store_max_address_space_vector(<2 x ptr addrspace(16777215)> %vptr) {
  %elt0 = extractelement <2 x ptr addrspace(16777215)> %vptr, i32 0
  store i32 0, ptr addrspace(16777215) %elt0
  ret void
}

; CHECK-LABEL: name: max_address_space_vector_max_num_elts
; CHECK: %0:_(<65535 x p16777215>) = G_LOAD %1(p0) :: (volatile load (<65535 x p16777215>) from `ptr undef`, align 524288)
define void @max_address_space_vector_max_num_elts() {
  %load = load volatile <65535 x ptr addrspace(16777215)>, ptr undef
  ret void
}
