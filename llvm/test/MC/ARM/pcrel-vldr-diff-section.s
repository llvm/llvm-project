@ RUN: not llvm-mc -filetype=obj -o /dev/null %s 2>&1 -triple=armv8.2a-eabi     | FileCheck %s
@ RUN: not llvm-mc -filetype=obj -o /dev/null %s 2>&1 -triple=armebv8.2a-eabi   | FileCheck %s
@ RUN: not llvm-mc -filetype=obj -o /dev/null %s 2>&1 -triple=thumbv8.2a-eabi   | FileCheck %s
@ RUN: not llvm-mc -filetype=obj -o /dev/null %s 2>&1 -triple=thumbebv8.2a-eabi | FileCheck %s

  .arch_extension fp16

vldr s0, foo     @ arm_pcrel_10 / t2_pcrel_10
vldr d0, foo     @ arm_pcrel_10 / t2_pcrel_10
vldr.16 s0,foo   @ arm_pcrel_9 / t2_pcrel_9

@ CHECK: :[[#@LINE-4]]:1: error: unsupported relocation type
@ CHECK: :[[#@LINE-4]]:1: error: unsupported relocation type
@ CHECK: :[[#@LINE-4]]:1: error: unsupported relocation type
