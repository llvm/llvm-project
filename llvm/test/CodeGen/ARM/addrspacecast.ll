; RUN: llc < %s -march=arm

; Check that codegen for an addrspace cast succeeds without error.
define <4 x i32 addrspace(1)*> @f (<4 x i32*> %x) {
  %1 = addrspacecast <4 x i32*> %x to <4 x i32 addrspace(1)*>
  ret <4 x i32 addrspace(1)*> %1
}
