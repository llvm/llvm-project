# With Bitmanip CRC extension:
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-zbr -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+experimental-zbr < %s \
# RUN:     | llvm-objdump --mattr=+experimental-zbr -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: crc32.b t0, t1
# CHECK-ASM: encoding: [0x93,0x12,0x03,0x61]
crc32.b	t0, t1
# CHECK-ASM-AND-OBJ: crc32.h t0, t1
# CHECK-ASM: encoding: [0x93,0x12,0x13,0x61]
crc32.h	t0, t1
# CHECK-ASM-AND-OBJ: crc32.w t0, t1
# CHECK-ASM: encoding: [0x93,0x12,0x23,0x61]
crc32.w	t0, t1
# CHECK-ASM-AND-OBJ: crc32c.b t0, t1
# CHECK-ASM: encoding: [0x93,0x12,0x83,0x61]
crc32c.b t0, t1
# CHECK-ASM-AND-OBJ: crc32c.h t0, t1
# CHECK-ASM: encoding: [0x93,0x12,0x93,0x61]
crc32c.h t0, t1
# CHECK-ASM-AND-OBJ: crc32c.w t0, t1
# CHECK-ASM: encoding: [0x93,0x12,0xa3,0x61]
crc32c.w t0, t1
