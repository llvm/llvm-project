# RUN: llvm-mc %s -triple=riscv32 -mattr=+xsifivecdiscarddlone,+xsifivecflushdlone,+xsfcease -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+xsifivecdiscarddlone,+xsifivecflushdlone,+xsfcease -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+xsifivecdiscarddlone,+xsifivecflushdlone,+xsfcease < %s \
# RUN:     | llvm-objdump --mattr=+xsifivecdiscarddlone,+xsifivecflushdlone,+xsfcease -M no-aliases -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+xsifivecdiscarddlone,+xsifivecflushdlone,+xsfcease < %s \
# RUN:     | llvm-objdump --mattr=+xsifivecdiscarddlone,+xsifivecflushdlone,+xsfcease -M no-aliases -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s

# CHECK-INST: sf.cflush.d.l1     zero
# CHECK-ENC: encoding: [0x73,0x00,0x00,0xfc]
sf.cflush.d.l1 x0
# CHECK-INST: sf.cflush.d.l1     zero
# CHECK-ENC: encoding: [0x73,0x00,0x00,0xfc]
sf.cflush.d.l1

# CHECK-INST: sf.cflush.d.l1     t2
# CHECK-ENC: encoding: [0x73,0x80,0x03,0xfc]
sf.cflush.d.l1 x7

# CHECK-INST: sf.cdiscard.d.l1   zero
# CHECK-ENC: encoding: [0x73,0x00,0x20,0xfc]
sf.cdiscard.d.l1 x0
# CHECK-INST: sf.cdiscard.d.l1   zero
# CHECK-ENC: encoding: [0x73,0x00,0x20,0xfc]
sf.cdiscard.d.l1

# CHECK-INST: sf.cdiscard.d.l1   t2
# CHECK-ENC: encoding: [0x73,0x80,0x23,0xfc]
sf.cdiscard.d.l1 x7

# CHECK-INST: sf.cease
# CHECK-ENC: encoding: [0x73,0x00,0x50,0x30]
sf.cease
