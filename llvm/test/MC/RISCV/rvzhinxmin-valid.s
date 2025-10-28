# RUN: llvm-mc %s -triple=riscv32 -mattr=+zhinxmin -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+zhinxmin -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+zhinxmin %s \
# RUN:     | llvm-objdump --mattr=+zhinxmin -M no-aliases -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+zhinxmin %s \
# RUN:     | llvm-objdump --mattr=+zhinxmin -M no-aliases -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: fcvt.s.h a0, a1
# CHECK-ASM: encoding: [0x53,0x85,0x25,0x40]
fcvt.s.h a0, a1
# CHECK-ASM-AND-OBJ: fcvt.s.h a0, a1, rup
# CHECK-ASM: encoding: [0x53,0xb5,0x25,0x40]
fcvt.s.h a0, a1, rup

# CHECK-ASM-AND-OBJ: fcvt.h.s a0, a1, dyn
# CHECK-ASM: encoding: [0x53,0xf5,0x05,0x44]
fcvt.h.s a0, a1
