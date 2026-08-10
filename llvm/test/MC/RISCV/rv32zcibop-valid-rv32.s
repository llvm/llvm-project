# RUN: llvm-mc %s -triple=riscv32 -mattr=+zicbop -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+zicbop < %s \
# RUN:     | llvm-objdump --no-print-imm-hex --mattr=+zicbop -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

## This test checks that 32-bit hex immediates are accepted for the `prefetch.*`
## instructions on rv32.

# CHECK-ASM-AND-OBJ: prefetch.i -2048(t0)
# CHECK-ASM: encoding: [0x13,0xe0,0x02,0x80]
prefetch.i 0xfffff800(t0)
# CHECK-ASM-AND-OBJ: prefetch.r -2048(t1)
# CHECK-ASM: encoding: [0x13,0x60,0x13,0x80]
prefetch.r 0xfffff800(t1)
# CHECK-ASM-AND-OBJ: prefetch.w -2048(t2)
# CHECK-ASM: encoding: [0x13,0xe0,0x33,0x80]
prefetch.w 0xfffff800(t2)
