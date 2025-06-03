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

# CHECK-INST: csrrs t1, sf.stvt, zero
# CHECK-ENC: encoding: [0x73,0x23,0x70,0x10]
csrrs t1, sf.stvt, zero
# CHECK-INST: csrrs t2, sf.stvt, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x70,0x10]
csrrs t2, 0x107, zero

# CHECK-INST: csrrs t1, sf.snxti, zero
# CHECK-ENC: encoding: [0x73,0x23,0x50,0x14]
csrrs t1, sf.snxti, zero
# CHECK-INST: csrrs t2, sf.snxti, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x50,0x14]
csrrs t2, 0x145, zero

# CHECK-INST: csrrs t1, sf.sintstatus, zero
# CHECK-ENC: encoding: [0x73,0x23,0x60,0x14]
csrrs t1, sf.sintstatus, zero
# CHECK-INST: csrrs t2, sf.sintstatus, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x60,0x14]
csrrs t2, 0x146, zero

# CHECK-INST: csrrs t1, sf.sscratchcsw, zero
# CHECK-ENC: encoding: [0x73,0x23,0x80,0x14]
csrrs t1, sf.sscratchcsw, zero
# CHECK-INST: csrrs t2, sf.sscratchcsw, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x80,0x14]
csrrs t2, 0x148, zero

# CHECK-INST: csrrs t1, sf.sscratchcswl, zero
# CHECK-ENC: encoding: [0x73,0x23,0x90,0x14]
csrrs t1, sf.sscratchcswl, zero
# CHECK-INST: csrrs t2, sf.sscratchcswl, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x90,0x14]
csrrs t2, 0x149, zero
