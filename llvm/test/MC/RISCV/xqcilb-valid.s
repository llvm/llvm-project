# Xqcilb - Qualcomm uC Long Branch Extension
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xqcilb -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST,CHECK-NOALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqcilb < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xqcilb -M no-aliases --no-print-imm-hex -d - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ,CHECK-OBJ-NOALIAS %s
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xqcilb -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST,CHECK-ALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqcilb < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xqcilb --no-print-imm-hex -d - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ,CHECK-OBJ-ALIAS %s

# CHECK-INST: qc.e.j  -2147483648
# CHECK-OBJ: qc.e.j  0x80000000
# CHECK-ENC: encoding: [0x1f,0x40,0x00,0x00,0x00,0x80]
qc.e.j  -2147483648

# CHECK-INST: qc.e.jal 2147483640
# CHECK-OBJ: qc.e.jal 0x7ffffffe
# CHECK-ENC: encoding: [0x9f,0xcc,0x0e,0xfe,0xff,0x7f]
qc.e.jal 2147483640

# Check that compress patterns work as expected

# CHECK-NOALIAS: c.jal -116
# CHECK-ALIAS: jal -116
# CHECK-OBJ-NOALIAS: c.jal 0xffffff98
# CHECK-OBJ-ALIAS: jal 0xffffff98
# CHECK-ENC: encoding: [0x71,0x37]
qc.e.jal 0xffffff8c

# CHECK-NOALIAS: c.j 1024
# CHECK-ALIAS: j 1024
# CHECK-OBJ-NOALIAS: c.j 0x40e
# CHECK-OBJ-ALIAS: j 0x40e
# CHECK-ENC: encoding: [0x01,0xa1]
qc.e.j 1024
