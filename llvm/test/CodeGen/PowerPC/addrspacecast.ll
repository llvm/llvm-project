; RUN: llc -mtriple=powerpc64le-unknown-linux-gnu %s -o - | FileCheck %s

; Check that codegen for an addrspace cast succeeds without error.
define <4 x ptr addrspace(1)> @f (<4 x ptr> %x) {
  %1 = addrspacecast <4 x ptr> %x to <4 x ptr addrspace(1)>
  ret <4 x ptr addrspace(1)> %1
  ; CHECK-LABEL: @f
}

; Check that fairly complicated addrspace cast and operations succeed without error.
%struct = type opaque
define void @g (ptr %x) {
  %1 = load ptr addrspace(10), ptr %x
  %2 = addrspacecast ptr addrspace(10) %1 to ptr addrspace(11)
  %3 = getelementptr i8, ptr addrspace(11) %2, i64 16
  %4 = load ptr addrspace(10), ptr addrspace(11) %3
  store ptr addrspace(10) %4, ptr undef
  ret void
  ; CHECK-LABEL: @g
}
