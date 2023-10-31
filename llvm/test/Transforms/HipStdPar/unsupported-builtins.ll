; RUN: not opt -S -mtriple=amdgcn-amd-amdhsa -passes=hipstdpar-select-accelerator-code \
; RUN: %s 2>&1 | FileCheck %s

; CHECK: error: {{.*}} in function foo void (): Accelerator does not support the __builtin_ia32_pause function
define amdgpu_kernel void @foo() {
entry:
  call void @__builtin_ia32_pause__hipstdpar_unsupported()
  ret void
}

declare void @__builtin_ia32_pause__hipstdpar_unsupported()