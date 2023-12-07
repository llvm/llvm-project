; RUN: llc -mtriple=arm-eabi %s -o /dev/null

; Check that codegen for an addrspace cast succeeds without error.
define <4 x ptr addrspace(1)> @f (<4 x ptr> %x) {
  %1 = addrspacecast <4 x ptr> %x to <4 x ptr addrspace(1)>
  ret <4 x ptr addrspace(1)> %1
}
