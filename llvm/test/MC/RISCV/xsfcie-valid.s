# SCIE - SiFive Custom Instructions Extension.
# RUN: llvm-mc %s -triple=riscv32 -mattr=+xsfcie -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+xsfcie -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv32 -mattr=+xsfcie -riscv-no-aliases -show-encoding 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-WARN %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+xsfcie -riscv-no-aliases -show-encoding 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-WARN %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+xsfcie < %s \
# RUN:     | llvm-objdump --mattr=+xsfcie -M no-aliases -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+xsfcie < %s \
# RUN:     | llvm-objdump --mattr=+xsfcie -M no-aliases -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv64 -mcpu=sifive-s76 -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv64 -mcpu=sifive-s76 -riscv-no-aliases -show-encoding 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-WARN %s
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

# mbpm
# name
# CHECK-INST: csrrs t2, mbpm, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x00,0x7c]
# uimm12
# CHECK-INST: csrrs t2, mbpm, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x00,0x7c]
# name
csrrs t2, mbpm, zero
# uimm12
csrrs t2, 0x7C0, zero

# mfd
# name
# CHECK-INST: csrrs t2, mfd, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x10,0x7c]
# uimm12
# CHECK-INST: csrrs t2, mfd, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x10,0x7c]
# name
csrrs t2, mfd, zero
# uimm12
csrrs t2, 0x7C1, zero

# mpd
# name
# CHECK-INST: csrrs t2, mpd, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x80,0x7c]
# uimm12
# CHECK-INST: csrrs t2, mpd, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x80,0x7c]
# name
csrrs t2, mpd, zero
# uimm12
csrrs t2, 0x7C8, zero

# mnscratch
# name
# CHECK-INST: csrrs t1, mnscratch, zero
# CHECK-ENC: encoding: [0x73,0x23,0x00,0x35]
# CHECK-WARN: warning: 'miselect' CSR is not available on the current subtarget. Instead 'mnscratch' CSR will be used.
# uimm12
# CHECK-INST: csrrs t2, mnscratch, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x00,0x35]
# name
csrrs t1, mnscratch, zero
csrrs t1, miselect, zero
# uimm12
csrrs t2, 0x350, zero

# mnepc
# name
# CHECK-INST: csrrs t1, mnepc, zero
# CHECK-ENC: encoding: [0x73,0x23,0x10,0x35]
# CHECK-WARN: warning: 'mireg' CSR is not available on the current subtarget. Instead 'mnepc' CSR will be used.
# uimm12
# CHECK-INST: csrrs t2, mnepc, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x10,0x35]
# name
csrrs t1, mnepc, zero
csrrs t1, mireg, zero
# uimm12
csrrs t2, 0x351, zero

# mncause
# name
# CHECK-INST: csrrs t1, mncause, zero
# CHECK-ENC: encoding: [0x73,0x23,0x20,0x35]
# uimm12
# CHECK-INST: csrrs t2, mncause, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x20,0x35]
# name
csrrs t1, mncause, zero
# uimm12
csrrs t2, 0x352, zero

# mnstatus
# name
# CHECK-INST: csrrs t1, mnstatus, zero
# CHECK-ENC: encoding: [0x73,0x23,0x30,0x35]
# uimm12
# CHECK-INST: csrrs t2, mnstatus, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x30,0x35]
# name
csrrs t1, mnstatus, zero
# uimm12
csrrs t2, 0x353, zero
