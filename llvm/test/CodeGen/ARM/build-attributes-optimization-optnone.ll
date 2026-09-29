; RUN: llc < %s -mtriple=arm-none-none-eabi -mcpu=cortex-a7 -O0 | FileCheck %s
; RUN: llc < %s -mtriple=arm-none-none-eabi -mcpu=cortex-a7 -O1 | FileCheck %s
; RUN: llc < %s -mtriple=arm-none-none-eabi -mcpu=cortex-a7 -O3 | FileCheck %s

; RUN: llc < %s -mtriple=arm-none-none-eabi -mcpu=cortex-a7 -O0 -filetype obj -o - | llvm-readobj --arch-specific - | FileCheck %s --check-prefix=CHECK-OBJ
; RUN: llc < %s -mtriple=arm-none-none-eabi -mcpu=cortex-a7 -O1 -filetype obj -o - | llvm-readobj --arch-specific - | FileCheck %s --check-prefix=CHECK-OBJ
; RUN: llc < %s -mtriple=arm-none-none-eabi -mcpu=cortex-a7 -O3 -filetype obj -o - | llvm-readobj --arch-specific - | FileCheck %s --check-prefix=CHECK-OBJ

; CHECK: .eabi_attribute 30, 6	@ Tag_ABI_optimization_goals
; CHECK-OBJ:          TagName: ABI_optimization_goals
; CHECK-OBJ-NEXT:     Description: Best Debugging

define i32 @f(i64 %z) #0 {
    ret i32 0
}

attributes #0 = { noinline optnone }

