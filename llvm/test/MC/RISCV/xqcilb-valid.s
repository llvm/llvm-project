# Xqcilb - Qualcomm uC Long Branch Extension
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xqcilb -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqcilb < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xqcilb -M no-aliases --no-print-imm-hex -d - \
# RUN:     | FileCheck -check-prefix=CHECK-OBJ %s
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xqcilb -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqcilb < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xqcilb --no-print-imm-hex -d - \
# RUN:     | FileCheck -check-prefix=CHECK-OBJ %s

# CHECK-INST: qc.e.j  -2147483648
# CHECK-OBJ: qc.e.j  0x80000000
# CHECK-ENC: encoding: [0x1f,0x40,0x00,0x00,0x00,0x80]
qc.e.j  -2147483648

# CHECK-INST: qc.e.jal 2147483640
# CHECK-OBJ: qc.e.jal 0x7ffffffe
# CHECK-ENC: encoding: [0x9f,0xcc,0x0e,0xfe,0xff,0x7f]
qc.e.jal 2147483640

# CHECK-INST: qc.e.jal -116
# CHECK-OBJ: qc.e.jal 0xffffff98
# CHECK-ENC: encoding: [0x9f,0xc6,0x0e,0xf8,0xff,0xff]
qc.e.jal 0xffffff8c
