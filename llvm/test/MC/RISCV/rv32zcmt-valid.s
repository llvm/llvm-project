# RUN: llvm-mc %s -triple=riscv32 -mattr=+zcmt\
# RUN:  -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+zcmt\
# RUN:  -mattr=m < %s \
# RUN:     | llvm-objdump --mattr=+zcmt --no-print-imm-hex \
# RUN:  -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefixes=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+zcmt\
# RUN:  -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+zcmt\
# RUN:  -mattr=m < %s \
# RUN:     | llvm-objdump --mattr=+zcmt --no-print-imm-hex \
# RUN:  -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefixes=CHECK-ASM-AND-OBJ %s
#
# RUN: not llvm-mc -triple riscv32 \
# RUN:     -riscv-no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-NO-EXT %s
# RUN: not llvm-mc -triple riscv64 \
# RUN:     -riscv-no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-NO-EXT %s

# CHECK-ASM-AND-OBJ: cm.jt 1
# CHECK-ASM: encoding: [0x06,0xa0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zcmt' (table jump instuctions for code-size reduction){{$}}
cm.jt 1

# CHECK-ASM-AND-OBJ: cm.jalt 32
# CHECK-ASM: encoding: [0x82,0xa0]
# CHECK-NO-EXT: error: instruction requires the following: 'Zcmt' (table jump instuctions for code-size reduction){{$}}
cm.jalt 32
