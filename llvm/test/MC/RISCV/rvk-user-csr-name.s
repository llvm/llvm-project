# RUN: llvm-mc %s -triple=riscv32 -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-INST,CHECK-ENC %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+zkr < %s \
# RUN:     | llvm-objdump -d --mattr=+zkr - \
# RUN:     | FileCheck -check-prefix=CHECK-INST-ALIAS %s
#
# RUN: llvm-mc %s -triple=riscv64 -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-INST,CHECK-ENC %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+zkr < %s \
# RUN:     | llvm-objdump -d --mattr=+zkr - \
# RUN:     | FileCheck -check-prefix=CHECK-INST-ALIAS %s

##################################
# Entropy Source CSR
##################################

# seed
# name
# CHECK-INST: csrrs t1, seed, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x50,0x01]
# CHECK-INST-ALIAS: csrr t1, seed
# uimm12
# CHECK-INST: csrrs t2, seed, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x50,0x01]
# CHECK-INST-ALIAS: csrr t2, seed
# name
csrrs t1, seed, zero
# uimm12
csrrs t2, 0x015, zero
