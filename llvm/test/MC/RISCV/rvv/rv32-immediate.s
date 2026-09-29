# RUN: llvm-mc -triple=riscv32 -show-encoding --mattr=+v %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=riscv32 -filetype=obj --mattr=+v %s \
# RUN:        | llvm-objdump -d --mattr=+v --no-print-imm-hex - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

# For rv32, allow 32 bit constants that contains a simm5 value.

vadd.vi v8, v4, 0xfffffff0
# CHECK-INST: vadd.vi v8, v4, -16
# CHECK-ENCODING: [0x57,0x34,0x48,0x02]

vmsltu.vi v8, v4, 0xfffffff1
# CHECK-INST: vmsleu.vi v8, v4, -16
# CHECK-ENCODING: [0x57,0x34,0x48,0x72]
