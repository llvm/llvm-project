# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-sscsps \
# RUN:     -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-INST,CHECK-ENC %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 \
# RUN:     -mattr=+experimental-sscsps < %s \
# RUN:     | llvm-objdump -d --mattr=+experimental-sscsps -M no-aliases - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+experimental-sscsps \
# RUN:     -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-INST,CHECK-ENC %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 \
# RUN:     -mattr=+experimental-sscsps < %s \
# RUN:     | llvm-objdump -d --mattr=+experimental-sscsps -M no-aliases - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s

# CHECK-INST: mcspspush sp, sp
# CHECK-ENC: encoding: [0x73,0x01,0x91,0x30]
mcspspush x2, x2

# CHECK-INST: mcspspop sp, sp
# CHECK-ENC: encoding: [0x73,0x01,0xc1,0x30]
mcspspop x2, x2

# CHECK-INST: scspspush sp, sp
# CHECK-ENC: encoding: [0x73,0x01,0x91,0x10]
scspspush x2, x2

# CHECK-INST: scspspop sp, sp
# CHECK-ENC: encoding: [0x73,0x01,0xc1,0x10]
scspspop x2, x2
