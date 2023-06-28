# SCIE - SiFive Custom Instructions Extension.
# RUN: llvm-mc %s -triple=riscv32 -mattr=+xsfcie -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+xsfcie -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+xsfcie < %s \
# RUN:     | llvm-objdump --mattr=+xsfcie -M no-aliases -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+xsfcie < %s \
# RUN:     | llvm-objdump --mattr=+xsfcie -M no-aliases -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv64 -mcpu=sifive-s76 -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mcpu=sifive-s76 < %s \
# RUN:     | llvm-objdump --mcpu=sifive-s76 -M no-aliases -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s

# CHECK-INST: cflush.d.l1     zero
# CHECK-ENC: encoding: [0x73,0x00,0x00,0xfc]
# CHECK-INST: cflush.d.l1     zero
# CHECK-ENC: encoding: [0x73,0x00,0x00,0xfc]
cflush.d.l1 x0
cflush.d.l1

# CHECK-INST: cflush.d.l1     t2
# CHECK-ENC: encoding: [0x73,0x80,0x03,0xfc]
cflush.d.l1 x7

# CHECK-INST: cdiscard.d.l1   zero
# CHECK-ENC: encoding: [0x73,0x00,0x20,0xfc]
# CHECK-INST: cdiscard.d.l1     zero
# CHECK-ENC: encoding: [0x73,0x00,0x20,0xfc]
cdiscard.d.l1 x0
cdiscard.d.l1

# CHECK-INST: cdiscard.d.l1   t2
# CHECK-ENC: encoding: [0x73,0x80,0x23,0xfc]
cdiscard.d.l1 x7

# CHECK-INST: cease
# CHECK-ENC: encoding: [0x73,0x00,0x50,0x30]
cease
