# RUN: llvm-mc %s -triple=riscv32 -mattr=+xtheadcmo -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+xtheadcmo < %s \
# RUN:     | llvm-objdump --mattr=+xtheadcmo -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+xtheadcmo -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+xtheadcmo < %s \
# RUN:     | llvm-objdump --mattr=+xtheadcmo -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: th.dcache.csw a6
# CHECK-ASM: encoding: [0x0b,0x00,0x18,0x02]
th.dcache.csw a6

# CHECK-ASM-AND-OBJ: th.dcache.isw t0
# CHECK-ASM: encoding: [0x0b,0x80,0x22,0x02]
th.dcache.isw t0

# CHECK-ASM-AND-OBJ: th.dcache.cisw a7
# CHECK-ASM: encoding: [0x0b,0x80,0x38,0x02]
th.dcache.cisw a7

# CHECK-ASM-AND-OBJ: th.dcache.cval1 t2
# CHECK-ASM: encoding: [0x0b,0x80,0x43,0x02]
th.dcache.cval1 t2

# CHECK-ASM-AND-OBJ: th.dcache.cva a3
# CHECK-ASM: encoding: [0x0b,0x80,0x56,0x02]
th.dcache.cva a3

# CHECK-ASM-AND-OBJ: th.dcache.iva a5
# CHECK-ASM: encoding: [0x0b,0x80,0x67,0x02]
th.dcache.iva a5

# CHECK-ASM-AND-OBJ: th.dcache.civa a4
# CHECK-ASM: encoding: [0x0b,0x00,0x77,0x02]
th.dcache.civa a4

# CHECK-ASM-AND-OBJ: th.dcache.cpal1 t1
# CHECK-ASM: encoding: [0x0b,0x00,0x83,0x02]
th.dcache.cpal1 t1

# CHECK-ASM-AND-OBJ: th.dcache.cpa a0
# CHECK-ASM: encoding: [0x0b,0x00,0x95,0x02]
th.dcache.cpa a0

# CHECK-ASM-AND-OBJ: th.dcache.ipa a2
# CHECK-ASM: encoding: [0x0b,0x00,0xa6,0x02]
th.dcache.ipa a2

# CHECK-ASM-AND-OBJ: th.dcache.cipa a1
# CHECK-ASM: encoding: [0x0b,0x80,0xb5,0x02]
th.dcache.cipa a1

# CHECK-ASM-AND-OBJ: th.icache.iva t4
# CHECK-ASM: encoding: [0x0b,0x80,0x0e,0x03]
th.icache.iva t4

# CHECK-ASM-AND-OBJ: th.icache.ipa t3
# CHECK-ASM: encoding: [0x0b,0x00,0x8e,0x03]
th.icache.ipa t3

# CHECK-ASM-AND-OBJ: th.dcache.call
# CHECK-ASM: encoding: [0x0b,0x00,0x10,0x00]
th.dcache.call

# CHECK-ASM-AND-OBJ: th.dcache.iall
# CHECK-ASM: encoding: [0x0b,0x00,0x20,0x00]
th.dcache.iall

# CHECK-ASM-AND-OBJ: th.dcache.ciall
# CHECK-ASM: encoding: [0x0b,0x00,0x30,0x00]
th.dcache.ciall

# CHECK-ASM-AND-OBJ: th.icache.iall
# CHECK-ASM: encoding: [0x0b,0x00,0x00,0x01]
th.icache.iall

# CHECK-ASM-AND-OBJ: th.icache.ialls
# CHECK-ASM: encoding: [0x0b,0x00,0x10,0x01]
th.icache.ialls

# CHECK-ASM-AND-OBJ: th.l2cache.call
# CHECK-ASM: encoding: [0x0b,0x00,0x50,0x01]
th.l2cache.call

# CHECK-ASM-AND-OBJ: th.l2cache.iall
# CHECK-ASM: encoding: [0x0b,0x00,0x60,0x01]
th.l2cache.iall

# CHECK-ASM-AND-OBJ: th.l2cache.ciall
# CHECK-ASM: encoding: [0x0b,0x00,0x70,0x01]
th.l2cache.ciall
