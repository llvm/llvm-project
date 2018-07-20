; RUN: llc -mtriple=powerpc64le-unknown-linux-gnu %s -o - | FileCheck %s

; Check that codegen for an addrspace cast succeeds without error.
define <4 x i32 addrspace(1)*> @f (<4 x i32*> %x) {
  %1 = addrspacecast <4 x i32*> %x to <4 x i32 addrspace(1)*>
  ret <4 x i32 addrspace(1)*> %1
  ; CHECK-LABEL: @f
}

; Check that fairly complicated addrspace cast and operations succeed without error.
%struct = type opaque
define void @g (%struct addrspace(10)** %x) {
  %1 = load %struct addrspace(10)*, %struct addrspace(10)** %x
  %2 = addrspacecast %struct addrspace(10)* %1 to %struct addrspace(11)*
  %3 = bitcast %struct addrspace(11)* %2 to i8 addrspace(11)*
  %4 = getelementptr i8, i8 addrspace(11)* %3, i64 16
  %5 = bitcast i8 addrspace(11)* %4 to %struct addrspace(10)* addrspace(11)*
  %6 = load %struct addrspace(10)*, %struct addrspace(10)* addrspace(11)* %5
  store %struct addrspace(10)* %6, %struct addrspace(10)** undef
  ret void
  ; CHECK-LABEL: @g
}
