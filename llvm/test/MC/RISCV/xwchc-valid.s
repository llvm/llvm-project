# RUN: llvm-mc %s -triple=riscv32 -mattr=+xwchc -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+xwchc < %s \
# RUN:     | llvm-objdump --mattr=+xwchc --no-print-imm-hex -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: qk.c.lbu s0, 0(s0)
# CHECK-ASM: encoding: [0x00,0x20]
qk.c.lbu s0, 0(s0)
# CHECK-ASM-AND-OBJ: qk.c.lbu s0, 1(s0)
# CHECK-ASM: encoding: [0x00,0x30]
qk.c.lbu s0, 1(s0)
# CHECK-ASM-AND-OBJ: qk.c.lbu s0, 2(s0)
# CHECK-ASM: encoding: [0x20,0x20]
qk.c.lbu s0, 2(s0)
# CHECK-ASM-AND-OBJ: qk.c.lbu s0, 4(s0)
# CHECK-ASM: encoding: [0x40,0x20]
qk.c.lbu s0, 4(s0)
# CHECK-ASM-AND-OBJ: qk.c.lbu s0, 8(s0)
# CHECK-ASM: encoding: [0x00,0x24]
qk.c.lbu s0, 8(s0)
# CHECK-ASM-AND-OBJ: qk.c.lbu s0, 16(s0)
# CHECK-ASM: encoding: [0x00,0x28]
qk.c.lbu s0, 16(s0)

# CHECK-ASM-AND-OBJ: qk.c.sb s0, 0(s0)
# CHECK-ASM: encoding: [0x00,0xa0]
qk.c.sb s0, 0(s0)
# CHECK-ASM-AND-OBJ: qk.c.sb s0, 1(s0)
# CHECK-ASM: encoding: [0x00,0xb0]
qk.c.sb s0, 1(s0)
# CHECK-ASM-AND-OBJ: qk.c.sb s0, 2(s0)
# CHECK-ASM: encoding: [0x20,0xa0]
qk.c.sb s0, 2(s0)
# CHECK-ASM-AND-OBJ: qk.c.sb s0, 4(s0)
# CHECK-ASM: encoding: [0x40,0xa0]
qk.c.sb s0, 4(s0)
# CHECK-ASM-AND-OBJ: qk.c.sb s0, 8(s0)
# CHECK-ASM: encoding: [0x00,0xa4]
qk.c.sb s0, 8(s0)
# CHECK-ASM-AND-OBJ: qk.c.sb s0, 16(s0)
# CHECK-ASM: encoding: [0x00,0xa8]
qk.c.sb s0, 16(s0)

# CHECK-ASM-AND-OBJ: qk.c.lhu s0, 0(s0)
# CHECK-ASM: encoding: [0x02,0x20]
qk.c.lhu s0, 0(s0)
# CHECK-ASM-AND-OBJ: qk.c.lhu s0, 2(s0)
# CHECK-ASM: encoding: [0x22,0x20]
qk.c.lhu s0, 2(s0)
# CHECK-ASM-AND-OBJ: qk.c.lhu s0, 4(s0)
# CHECK-ASM: encoding: [0x42,0x20]
qk.c.lhu s0, 4(s0)
# CHECK-ASM-AND-OBJ: qk.c.lhu s0, 8(s0)
# CHECK-ASM: encoding: [0x02,0x24]
qk.c.lhu s0, 8(s0)
# CHECK-ASM-AND-OBJ: qk.c.lhu s0, 16(s0)
# CHECK-ASM: encoding: [0x02,0x28]
qk.c.lhu s0, 16(s0)
# CHECK-ASM-AND-OBJ: qk.c.lhu s0, 32(s0)
# CHECK-ASM: encoding: [0x02,0x30]
qk.c.lhu s0, 32(s0)

# CHECK-ASM-AND-OBJ: qk.c.sh s0, 0(s0)
# CHECK-ASM: encoding: [0x02,0xa0]
qk.c.sh s0, 0(s0)
# CHECK-ASM-AND-OBJ: qk.c.sh s0, 2(s0)
# CHECK-ASM: encoding: [0x22,0xa0]
qk.c.sh s0, 2(s0)
# CHECK-ASM-AND-OBJ: qk.c.sh s0, 4(s0)
# CHECK-ASM: encoding: [0x42,0xa0]
qk.c.sh s0, 4(s0)
# CHECK-ASM-AND-OBJ: qk.c.sh s0, 8(s0)
# CHECK-ASM: encoding: [0x02,0xa4]
qk.c.sh s0, 8(s0)
# CHECK-ASM-AND-OBJ: qk.c.sh s0, 16(s0)
# CHECK-ASM: encoding: [0x02,0xa8]
qk.c.sh s0, 16(s0)
# CHECK-ASM-AND-OBJ: qk.c.sh s0, 32(s0)
# CHECK-ASM: encoding: [0x02,0xb0]
qk.c.sh s0, 32(s0)

# CHECK-ASM-AND-OBJ: qk.c.lbusp s0, 0(sp)
# CHECK-ASM: encoding: [0x00,0x80]
qk.c.lbusp s0, 0(sp)
# CHECK-ASM-AND-OBJ: qk.c.lbusp s0, 1(sp)
# CHECK-ASM: encoding: [0x80,0x80]
qk.c.lbusp s0, 1(sp)
# CHECK-ASM-AND-OBJ: qk.c.lbusp s0, 2(sp)
# CHECK-ASM: encoding: [0x00,0x81]
qk.c.lbusp s0, 2(sp)
# CHECK-ASM-AND-OBJ: qk.c.lbusp s0, 4(sp)
# CHECK-ASM: encoding: [0x00,0x82]
qk.c.lbusp s0, 4(sp)
# CHECK-ASM-AND-OBJ: qk.c.lbusp s0, 8(sp)
# CHECK-ASM: encoding: [0x00,0x84]
qk.c.lbusp s0, 8(sp)

# CHECK-ASM-AND-OBJ: qk.c.sbsp s0, 0(sp)
# CHECK-ASM: encoding: [0x40,0x80]
qk.c.sbsp s0, 0(sp)
# CHECK-ASM-AND-OBJ: qk.c.sbsp s0, 1(sp)
# CHECK-ASM: encoding: [0xc0,0x80]
qk.c.sbsp s0, 1(sp)
# CHECK-ASM-AND-OBJ: qk.c.sbsp s0, 2(sp)
# CHECK-ASM: encoding: [0x40,0x81]
qk.c.sbsp s0, 2(sp)
# CHECK-ASM-AND-OBJ: qk.c.sbsp s0, 4(sp)
# CHECK-ASM: encoding: [0x40,0x82]
qk.c.sbsp s0, 4(sp)
# CHECK-ASM-AND-OBJ: qk.c.sbsp s0, 8(sp)
# CHECK-ASM: encoding: [0x40,0x84]
qk.c.sbsp s0, 8(sp)

# CHECK-ASM-AND-OBJ: qk.c.lhusp s0, 0(sp)
# CHECK-ASM: encoding: [0x20,0x80]
qk.c.lhusp s0, 0(sp)
# CHECK-ASM-AND-OBJ: qk.c.lhusp s0, 2(sp)
# CHECK-ASM: encoding: [0x20,0x81]
qk.c.lhusp s0, 2(sp)
# CHECK-ASM-AND-OBJ: qk.c.lhusp s0, 4(sp)
# CHECK-ASM: encoding: [0x20,0x82]
qk.c.lhusp s0, 4(sp)
# CHECK-ASM-AND-OBJ: qk.c.lhusp s0, 8(sp)
# CHECK-ASM: encoding: [0x20,0x84]
qk.c.lhusp s0, 8(sp)
# CHECK-ASM-AND-OBJ: qk.c.lhusp s0, 16(sp)
# CHECK-ASM: encoding: [0xa0,0x80]
qk.c.lhusp s0, 16(sp)

# CHECK-ASM-AND-OBJ: qk.c.shsp s0, 0(sp)
# CHECK-ASM: encoding: [0x60,0x80]
qk.c.shsp s0, 0(sp)
# CHECK-ASM-AND-OBJ: qk.c.shsp s0, 2(sp)
# CHECK-ASM: encoding: [0x60,0x81]
qk.c.shsp s0, 2(sp)
# CHECK-ASM-AND-OBJ: qk.c.shsp s0, 4(sp)
# CHECK-ASM: encoding: [0x60,0x82]
qk.c.shsp s0, 4(sp)
# CHECK-ASM-AND-OBJ: qk.c.shsp s0, 8(sp)
# CHECK-ASM: encoding: [0x60,0x84]
qk.c.shsp s0, 8(sp)
# CHECK-ASM-AND-OBJ: qk.c.shsp s0, 16(sp)
# CHECK-ASM: encoding: [0xe0,0x80]
qk.c.shsp s0, 16(sp)
