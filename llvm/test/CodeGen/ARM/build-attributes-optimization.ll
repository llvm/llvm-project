; RUN: llc < %s -mtriple=arm-none-none-eabi -mcpu=cortex-a7 -O0 | FileCheck %s --check-prefix=NONE
; RUN: llc < %s -mtriple=arm-none-none-eabi -mcpu=cortex-a7 -O1 | FileCheck %s --check-prefix=SPEED
; RUN: llc < %s -mtriple=arm-none-none-eabi -mcpu=cortex-a7 -O3 | FileCheck %s --check-prefix=MAXSPEED

; RUN: llc < %s -mtriple=arm-none-none-eabi -mcpu=cortex-a7 -O0 -filetype obj -o - | llvm-readobj -arm-attributes - | FileCheck %s --check-prefix=NONE-OBJ
; RUN: llc < %s -mtriple=arm-none-none-eabi -mcpu=cortex-a7 -O1 -filetype obj -o - | llvm-readobj -arm-attributes - | FileCheck %s --check-prefix=SPEED-OBJ
; RUN: llc < %s -mtriple=arm-none-none-eabi -mcpu=cortex-a7 -O3 -filetype obj -o - | llvm-readobj -arm-attributes - | FileCheck %s --check-prefix=MAXSPEED-OBJ

; NONE:     .eabi_attribute 30, 5	@ Tag_ABI_optimization_goals
; SPEED:    .eabi_attribute 30, 1	@ Tag_ABI_optimization_goals
; MAXSPEED: .eabi_attribute 30, 2	@ Tag_ABI_optimization_goals

; NONE-OBJ:          TagName: ABI_optimization_goals
; NONE-OBJ-NEXT:     Description: Debugging
; SPEED-OBJ:         TagName: ABI_optimization_goals
; SPEED-OBJ-NEXT:    Description: Speed
; MAXSPEED-OBJ:      TagName: ABI_optimization_goals
; MAXSPEED-OBJ-NEXT: Description: Aggressive Speed

define i32 @f(i64 %z) {
    ret i32 0
}

