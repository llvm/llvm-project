@ RUN: llvm-mc -mcpu=cortex-a9 -triple armv7-arm-none-eabi -filetype obj -o - %s \
@ RUN:   | llvm-objdump -mcpu=cortex-a9 -triple armv7-arm-none-eabi -d - \
@ RUN:   | FileCheck %s -check-prefix CHECK-ARM

@ RUN: llvm-mc -mcpu=cortex-m3 -triple thumbv7m-arm-none-eabi -filetype obj -o - %s \
@ RUN:   | llvm-objdump -mcpu=cortex-m3 -triple thumbv7m-arm-none-eabi -d - \
@ RUN:   | FileCheck %s -check-prefix CHECK-THUMB

b.w .Lbranch
@ CHECK-ARM: b #4 <$a.0+0xC>
@ CHECK-THUMB: b.w #8 <$t.0+0xC>
adds r0, r1, #42
adds r1, r2, #42
.Lbranch:
movs r2, r3
