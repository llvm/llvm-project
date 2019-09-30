; RUN: llc < %s -mtriple=arm-none-none-eabi -mcpu=cortex-a7 -O0 | FileCheck %s
; RUN: llc < %s -mtriple=arm-none-none-eabi -mcpu=cortex-a7 -O1 | FileCheck %s
; RUN: llc < %s -mtriple=arm-none-none-eabi -mcpu=cortex-a7 -O3 | FileCheck %s

; RUN: llc < %s -mtriple=arm-none-none-eabi -mcpu=cortex-a7 -O0 -filetype obj -o - | llvm-readobj --arch-specific - | FileCheck %s
; RUN: llc < %s -mtriple=arm-none-none-eabi -mcpu=cortex-a7 -O1 -filetype obj -o - | llvm-readobj --arch-specific - | FileCheck %s
; RUN: llc < %s -mtriple=arm-none-none-eabi -mcpu=cortex-a7 -O3 -filetype obj -o - | llvm-readobj --arch-specific - | FileCheck %s

; CHECK-NOT: .eabi_attribute 30
; CHECK-NOT: Tag_ABI_optimization_goals

define i32 @f(i64 %z) #0 {
    ret i32 0
}

define i32 @g(i64 %z) #1 {
    ret i32 1
}

attributes #0 = { noinline optnone }

attributes #1 = { minsize optsize }

