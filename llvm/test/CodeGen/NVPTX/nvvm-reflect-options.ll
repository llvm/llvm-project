; Verify that when passing in command-line options to NVVMReflect, that reflect calls are replaced with
; the appropriate command line values.

declare i32 @__nvvm_reflect(ptr)
@ftz = private unnamed_addr addrspace(1) constant [11 x i8] c"__CUDA_FTZ\00"
@arch = private unnamed_addr addrspace(1) constant [12 x i8] c"__CUDA_ARCH\00"

; RUN: opt -passes=nvvm-reflect -mtriple=nvptx-nvidia-cuda -nvvm-reflect-add __CUDA_FTZ=1 -nvvm-reflect-add __CUDA_ARCH=350 %s -S | FileCheck %s --check-prefix=CHECK-FTZ1-ARCH350
; RUN: opt -passes=nvvm-reflect -mtriple=nvptx-nvidia-cuda -nvvm-reflect-add __CUDA_FTZ=0 -nvvm-reflect-add __CUDA_ARCH=520 %s -S | FileCheck %s --check-prefix=CHECK-FTZ0-ARCH520

; Verify that if we have module metadata that sets __CUDA_FTZ=1, that gets overridden by the command line arguments

; RUN: cat %s > %t.options
; RUN: echo '!llvm.module.flags = !{!0}' >> %t.options
; RUN: echo '!0 = !{i32 4, !"nvvm-reflect-ftz", i32 1}' >> %t.options
; RUN: opt -passes=nvvm-reflect -mtriple=nvptx-nvidia-cuda -nvvm-reflect-add __CUDA_FTZ=0 -nvvm-reflect-add __CUDA_ARCH=520 %t.options -S | FileCheck %s --check-prefix=CHECK-FTZ0-ARCH520

define i32 @options() {
  %1 = call i32 @__nvvm_reflect(ptr addrspacecast (ptr addrspace(1) @ftz to ptr))
  %2 = call i32 @__nvvm_reflect(ptr addrspacecast (ptr addrspace(1) @arch to ptr))
  %3 = add i32 %1, %2
  ret i32 %3
}

; CHECK-FTZ1-ARCH350: ret i32 351
; CHECK-FTZ0-ARCH520: ret i32 520