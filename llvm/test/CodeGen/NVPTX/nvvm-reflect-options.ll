; Test the NVVM reflect pass functionality: verifying that reflect calls are replaced with 
; appropriate values based on command-line options. Verify that we can handle custom reflect arguments
; that aren't __CUDA_ARCH or __CUDA_FTZ. If that argument is given a value on the command-line, the reflect call should be replaced with that value.
; Otherwise, the reflect call should be replaced with 0.

; RUN: opt -passes=nvvm-reflect -mtriple=nvptx-nvidia-cuda -nvvm-reflect-add __CUDA_FTZ=1 -nvvm-reflect-add __CUDA_ARCH=350 %s -S | FileCheck %s --check-prefix=CHECK-FTZ1 --check-prefix=CHECK-ARCH350 --check-prefix=CHECK-CUSTOM-ABSENT
; RUN: opt -passes=nvvm-reflect -mtriple=nvptx-nvidia-cuda -nvvm-reflect-add __CUDA_FTZ=0 -nvvm-reflect-add __CUDA_ARCH=520 %s -S | FileCheck %s --check-prefix=CHECK-FTZ0 --check-prefix=CHECK-ARCH520 --check-prefix=CHECK-CUSTOM-ABSENT
; RUN: opt -passes=nvvm-reflect -mtriple=nvptx-nvidia-cuda -nvvm-reflect-add __CUDA_FTZ=0 -nvvm-reflect-add __CUDA_ARCH=520 -nvvm-reflect-add __CUSTOM_VALUE=42 %s -S | FileCheck %s --check-prefix=CHECK-CUSTOM-PRESENT

; To ensure that command line options override module options, create a copy of this test file with module options appended and rerun some tests.
;
; RUN: cat %s > %t.options
; RUN: echo '!llvm.module.flags = !{!0}' >> %t.options
; RUN: echo '!0 = !{i32 4, !"nvvm-reflect-ftz", i32 1}' >> %t.options
; RUN: opt -passes=nvvm-reflect -mtriple=nvptx-nvidia-cuda -nvvm-reflect-add __CUDA_FTZ=0 -nvvm-reflect-add __CUDA_ARCH=520 %t.options -S | FileCheck %s --check-prefix=CHECK-FTZ0 --check-prefix=CHECK-ARCH520

declare i32 @__nvvm_reflect(ptr)
@ftz = private unnamed_addr addrspace(1) constant [11 x i8] c"__CUDA_FTZ\00"
@arch = private unnamed_addr addrspace(1) constant [12 x i8] c"__CUDA_ARCH\00"
@custom = private unnamed_addr addrspace(1) constant [15 x i8] c"__CUSTOM_VALUE\00"

; Test handling of __CUDA_FTZ reflect value
define i32 @test_ftz() {
  %1 = call i32 @__nvvm_reflect(ptr addrspacecast (ptr addrspace(1) @ftz to ptr))
  ret i32 %1
}

; CHECK-FTZ1: define i32 @test_ftz()
; CHECK-FTZ1: ret i32 1
; CHECK-FTZ0: define i32 @test_ftz()
; CHECK-FTZ0: ret i32 0

; Test handling of __CUDA_ARCH reflect value
define i32 @test_arch() {
  %1 = call i32 @__nvvm_reflect(ptr addrspacecast (ptr addrspace(1) @arch to ptr))
  ret i32 %1
}

; CHECK-ARCH350: define i32 @test_arch()
; CHECK-ARCH350: ret i32 350
; CHECK-ARCH520: define i32 @test_arch()
; CHECK-ARCH520: ret i32 520

; Test handling of a custom reflect value that's not built into the pass
define i32 @test_custom() {
  %1 = call i32 @__nvvm_reflect(ptr addrspacecast (ptr addrspace(1) @custom to ptr))
  ret i32 %1
}

; CHECK-CUSTOM-ABSENT: define i32 @test_custom()
; CHECK-CUSTOM-ABSENT: ret i32 0
; CHECK-CUSTOM-PRESENT: define i32 @test_custom()
; CHECK-CUSTOM-PRESENT: ret i32 42
