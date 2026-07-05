; RUN: not opt -S -mtriple=amdgcn-amd-amdhsa -passes=hipstdpar-select-accelerator-code \
; RUN: %s 2>&1 | FileCheck %s

; CHECK: error: {{.*}} in function foo void (): Accelerator does not support the ASM block:
; CHECK-NEXT: {{.*}}Invalid ASM block{{.*}}
define amdgpu_kernel void @foo() {
entry:
  call void @__ASM__hipstdpar_unsupported([18 x i8] c"Invalid ASM block\00")
  ret void
}

declare void @__ASM__hipstdpar_unsupported([18 x i8])