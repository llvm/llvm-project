; Test the NVVM reflect pass functionality: verifying that reflect calls are replaced with 
; appropriate values based on command-line options. Verify that we can handle custom reflect arguments
; that aren't __CUDA_ARCH or __CUDA_FTZ. If that argument is given a value on the command-line,
; the reflect call should be replaced with that value. Otherwise, the reflect call should be replaced with 0.

; RUN: opt -passes=nvvm-reflect -mtriple=nvptx-nvidia-cuda \
; RUN:   -nvvm-reflect-add __CUDA_FTZ=1 -nvvm-reflect-add __CUDA_ARCH=350 %s -S \
; RUN:   | FileCheck %s --check-prefixes=COMMON,FTZ1,ARCH350,CUSTOM-ABSENT
; RUN: opt -passes=nvvm-reflect -mtriple=nvptx-nvidia-cuda \
; RUN:   -nvvm-reflect-add __CUDA_FTZ=0 -nvvm-reflect-add __CUDA_ARCH=520 %s -S \
; RUN:   | FileCheck %s --check-prefixes=COMMON,FTZ0,ARCH520,CUSTOM-ABSENT
; RUN: opt -passes=nvvm-reflect -mtriple=nvptx-nvidia-cuda \
; RUN:   -nvvm-reflect-add __CUDA_FTZ=0 -nvvm-reflect-add __CUDA_ARCH=520 \
; RUN:   -nvvm-reflect-add __CUSTOM_VALUE=42 %s -S \
; RUN:   | FileCheck %s --check-prefixes=COMMON,CUSTOM-PRESENT

; To ensure that command line options override module options, create a copy of this test file 
; with module options appended and rerun some tests.

; RUN: cat %s > %t.options
; RUN: echo '!llvm.module.flags = !{!0}' >> %t.options
; RUN: echo '!0 = !{i32 4, !"nvvm-reflect-ftz", i32 1}' >> %t.options
; RUN: opt -passes=nvvm-reflect -mtriple=nvptx-nvidia-cuda \
; RUN:   -nvvm-reflect-add __CUDA_FTZ=0 -nvvm-reflect-add __CUDA_ARCH=520 %t.options -S \
; RUN:   | FileCheck %s --check-prefixes=COMMON,FTZ0,ARCH520

declare i32 @__nvvm_reflect(ptr)
@ftz = private unnamed_addr addrspace(1) constant [11 x i8] c"__CUDA_FTZ\00"
@arch = private unnamed_addr addrspace(1) constant [12 x i8] c"__CUDA_ARCH\00"
@custom = private unnamed_addr addrspace(1) constant [15 x i8] c"__CUSTOM_VALUE\00"

; COMMON-LABEL: define i32 @test_ftz()
; FTZ1: ret i32 1
; FTZ0: ret i32 0
define i32 @test_ftz() {
  %1 = call i32 @__nvvm_reflect(ptr addrspacecast (ptr addrspace(1) @ftz to ptr))
  ret i32 %1
}

; COMMON-LABEL: define i32 @test_arch()
; ARCH350: ret i32 350
; ARCH520: ret i32 520
define i32 @test_arch() {
  %1 = call i32 @__nvvm_reflect(ptr addrspacecast (ptr addrspace(1) @arch to ptr))
  ret i32 %1
}

; COMMON-LABEL: define i32 @test_custom()
; CUSTOM-ABSENT: ret i32 0
; CUSTOM-PRESENT: ret i32 42
define i32 @test_custom() {
  %1 = call i32 @__nvvm_reflect(ptr addrspacecast (ptr addrspace(1) @custom to ptr))
  ret i32 %1
}
