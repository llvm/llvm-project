# RUN: llvm-mc %s -triple=riscv64 -mattr=+xventanacondops -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+xventanacondops < %s \
# RUN:     | llvm-objdump --mattr=+xventanacondops -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: vt.maskc zero, zero, zero
# CHECK-ASM: encoding: [0x7b,0x60,0x00,0x00]
vt.maskc x0, x0, x0

# CHECK-ASM-AND-OBJ: vt.maskcn zero, zero, zero
# CHECK-ASM: encoding: [0x7b,0x70,0x00,0x00]
vt.maskcn x0, x0, x0

# CHECK-ASM-AND-OBJ: vt.maskc ra, sp, gp
# CHECK-ASM: encoding: [0xfb,0x60,0x31,0x00]
vt.maskc x1, x2, x3

# CHECK-ASM-AND-OBJ: vt.maskcn ra, sp, gp
# CHECK-ASM: encoding: [0xfb,0x70,0x31,0x00]
vt.maskcn x1, x2, x3

