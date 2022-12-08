; RUN: opt < %s -S -mtriple=nvptx-nvidia-cuda -passes=nvvm-reflect | FileCheck %s
; RUN: opt < %s -S -mtriple=nvptx-nvidia-cuda -passes=nvvm-reflect | FileCheck %s

declare i32 @__nvvm_reflect(i8*)
@str = private unnamed_addr addrspace(1) constant [11 x i8] c"__CUDA_FTZ\00"

define i32 @foo() {
  %call = call i32 @__nvvm_reflect(i8* addrspacecast (i8 addrspace(1)* getelementptr inbounds ([11 x i8], [11 x i8] addrspace(1)* @str, i32 0, i32 0) to i8*))
  ; CHECK: ret i32 42
  ret i32 %call
}

!llvm.module.flags = !{!0}
!0 = !{i32 4, !"nvvm-reflect-ftz", i32 42}
