; RUN: not --crash opt -S -mtriple=amdgcn-- -amdgpu-lower-module-lds --amdgpu-lower-module-lds-strategy=module < %s 2>&1 | FileCheck %s
; RUN: not --crash opt -S -mtriple=amdgcn-- -passes=amdgpu-lower-module-lds --amdgpu-lower-module-lds-strategy=module < %s 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: cannot lower LDS 'var0' because it has a zero-sized type
@var0 = internal addrspace(3) global [0 x float] poison, align 4

define amdgpu_kernel void @kernel() {
  load float, ptr addrspace(3) @var0
  ret void
}
