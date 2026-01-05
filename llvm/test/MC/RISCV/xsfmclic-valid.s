# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xsfmclic -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-INST,CHECK-ENC %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xsfmclic < %s \
# RUN:     | llvm-objdump -d  --mattr=+experimental-xsfmclic -M no-aliases - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
#
# RUN: llvm-mc %s -triple=riscv64 -mattr=+experimental-xsfmclic -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-INST,CHECK-ENC %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+experimental-xsfmclic < %s \
# RUN:     | llvm-objdump -d --mattr=+experimental-xsfmclic -M no-aliases - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s

# CHECK-INST: csrrs t1, sf.mtvt, zero
# CHECK-ENC: encoding: [0x73,0x23,0x70,0x30]
csrrs t1, sf.mtvt, zero
# CHECK-INST: csrrs t2, sf.mtvt, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x70,0x30]
csrrs t2, 0x307, zero

# CHECK-INST: csrrs t1, sf.mnxti, zero
# CHECK-ENC: encoding: [0x73,0x23,0x50,0x34]
csrrs t1, sf.mnxti, zero
# CHECK-INST: csrrs t2, sf.mnxti, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x50,0x34]
csrrs t2, 0x345, zero

# CHECK-INST: csrrs t1, sf.mintstatus, zero
# CHECK-ENC: encoding: [0x73,0x23,0x60,0x34]
csrrs t1, sf.mintstatus, zero
# CHECK-INST: csrrs t2, sf.mintstatus, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x60,0x34]
csrrs t2, 0x346, zero

# CHECK-INST: csrrs t1, sf.mscratchcsw, zero
# CHECK-ENC: encoding: [0x73,0x23,0x80,0x34]
csrrs t1, sf.mscratchcsw, zero
# CHECK-INST: csrrs t2, sf.mscratchcsw, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x80,0x34]
csrrs t2, 0x348, zero

# CHECK-INST: csrrs t1, sf.mscratchcswl, zero
# CHECK-ENC: encoding: [0x73,0x23,0x90,0x34]
csrrs t1, sf.mscratchcswl, zero
# CHECK-INST: csrrs t2, sf.mscratchcswl, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x90,0x34]
csrrs t2, 0x349, zero
