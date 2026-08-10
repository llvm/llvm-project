# RUN: llvm-mc %s -triple=csky -show-encoding -csky-no-aliases -mattr=+e1 \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=csky -mattr=+e1 < %s \
# RUN:     | llvm-objdump --mattr=+e1 -M no-aliases -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: ipush
# CHECK-ASM: encoding: [0x62,0x14]
ipush16

# CHECK-ASM-AND-OBJ: ipop
# CHECK-ASM: encoding: [0x63,0x14]
ipop16

# RUN: not llvm-mc -triple csky -mattr=+e1 --defsym=ERR=1 < %s 2>&1 | FileCheck %s

.ifdef ERR
ipush16 1 # CHECK: :[[#@LINE]]:9: error: invalid operand for instruction
.endif
