# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xsfsclic -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-INST,CHECK-ENC %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xsfsclic < %s \
# RUN:     | llvm-objdump -d  --mattr=+experimental-xsfsclic -M no-aliases - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
#
# RUN: llvm-mc %s -triple=riscv64 -mattr=+experimental-xsfsclic -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-INST,CHECK-ENC %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+experimental-xsfsclic < %s \
# RUN:     | llvm-objdump -d --mattr=+experimental-xsfsclic -M no-aliases - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s

# CHECK-INST: csrrs t1, stvt, zero
# CHECK-ENC: encoding: [0x73,0x23,0x70,0x10]
csrrs t1, stvt, zero
# CHECK-INST: csrrs t2, stvt, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x70,0x10]
csrrs t2, 0x107, zero

# CHECK-INST: csrrs t1, snxti, zero
# CHECK-ENC: encoding: [0x73,0x23,0x50,0x14]
csrrs t1, snxti, zero
# CHECK-INST: csrrs t2, snxti, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x50,0x14]
csrrs t2, 0x145, zero

# CHECK-INST: csrrs t1, sintstatus, zero
# CHECK-ENC: encoding: [0x73,0x23,0x60,0x14]
csrrs t1, sintstatus, zero
# CHECK-INST: csrrs t2, sintstatus, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x60,0x14]
csrrs t2, 0x146, zero

# CHECK-INST: csrrs t1, sscratchcsw, zero
# CHECK-ENC: encoding: [0x73,0x23,0x80,0x14]
csrrs t1, sscratchcsw, zero
# CHECK-INST: csrrs t2, sscratchcsw, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x80,0x14]
csrrs t2, 0x148, zero

# CHECK-INST: csrrs t1, sscratchcswl, zero
# CHECK-ENC: encoding: [0x73,0x23,0x90,0x14]
csrrs t1, sscratchcswl, zero
# CHECK-INST: csrrs t2, sscratchcswl, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x90,0x14]
csrrs t2, 0x149, zero
