# RUN: llvm-mc %s -triple=riscv32 -mattr=+q,+zfhmin -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+q,+zfhmin -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+q,+zfhmin < %s \
# RUN:     | llvm-objdump --mattr=+q,+zfhmin --no-print-imm-hex -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+q,+zfhmin < %s \
# RUN:     | llvm-objdump --mattr=+q,+zfhmin --no-print-imm-hex -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: fcvt.q.h fa0, ft0
# CHECK-ASM: encoding: [0x53,0x05,0x20,0x46]
fcvt.q.h fa0, ft0
# CHECK-ASM-AND-OBJ: fcvt.q.h fa0, ft0, rup
# CHECK-ASM: encoding: [0x53,0x35,0x20,0x46]
fcvt.q.h fa0, ft0, rup

# CHECK-ASM-AND-OBJ: fcvt.h.q ft2, fa2
# CHECK-ASM: encoding: [0x53,0x71,0x36,0x44]
fcvt.h.q ft2, fa2
# CHECK-ASM-AND-OBJ: fcvt.h.q ft2, fa2, rup
# CHECK-ASM: encoding: [0x53,0x31,0x36,0x44]
fcvt.h.q ft2, fa2, rup
