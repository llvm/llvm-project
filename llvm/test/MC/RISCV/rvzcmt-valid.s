# RUN: llvm-mc %s -triple=riscv32 -mattr=+zcmt\
# RUN:  -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+zcmt\
# RUN:  -mattr=m < %s \
# RUN:     | llvm-objdump --mattr=+zcmt --no-print-imm-hex \
# RUN:  -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefixes=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+zcmt\
# RUN:  -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+zcmt\
# RUN:  -mattr=m < %s \
# RUN:     | llvm-objdump --mattr=+zcmt --no-print-imm-hex \
# RUN:  -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefixes=CHECK-ASM-AND-OBJ %s
#
# RUN: llvm-mc --verify -triple riscv32 -M no-aliases -show-encoding < %s
# RUN: llvm-mc --verify -triple riscv64 -M no-aliases -show-encoding < %s

# CHECK-ASM-AND-OBJ: cm.jt 1
# CHECK-ASM: encoding: [0x06,0xa0]
cm.jt 1 # expected-error {{instruction requires the following: 'Zcmt' (table jump instructions for code-size reduction)}}

# CHECK-ASM-AND-OBJ: cm.jalt 32
# CHECK-ASM: encoding: [0x82,0xa0]
cm.jalt 32 # expected-error {{instruction requires the following: 'Zcmt' (table jump instructions for code-size reduction)}}
