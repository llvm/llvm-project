# RUN: llvm-mc %s -triple=riscv32 -mattr=+smrnmi -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+smrnmi -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+smrnmi < %s \
# RUN:     | llvm-objdump --mattr=+smrnmi -M no-aliases -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+smrnmi < %s \
# RUN:     | llvm-objdump --mattr=+smrnmi -M no-aliases -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s

# CHECK-INST: mnret
# CHECK: encoding: [0x73,0x00,0x20,0x70]
mnret
