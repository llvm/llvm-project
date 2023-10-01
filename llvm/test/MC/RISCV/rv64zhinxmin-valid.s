# RUN: llvm-mc %s -triple=riscv64 -mattr=+zhinx,+zdinx -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+zhinx,+zdinx %s \
# RUN:     | llvm-objdump --mattr=+zhinx,+zdinx -M no-aliases -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: fcvt.d.h a0, a2
# CHECK-ASM: encoding: [0x53,0x05,0x26,0x42]
fcvt.d.h a0, a2
# CHECK-ASM-AND-OBJ: fcvt.d.h a0, a2, rup
# CHECK-ASM: encoding: [0x53,0x35,0x26,0x42]
fcvt.d.h a0, a2, rup

# CHECK-ASM-AND-OBJ: fcvt.h.d a0, a2, dyn
# CHECK-ASM: encoding: [0x53,0x75,0x16,0x44]
fcvt.h.d a0, a2, dyn
