# RUN: llvm-mc %s -triple=riscv32 -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv64 -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 < %s \
# RUN:     | llvm-objdump -d - | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv64 < %s \
# RUN:     | llvm-objdump -d - | FileCheck -check-prefix=CHECK-INST %s

# CHECK-INST: uret
# CHECK: encoding: [0x73,0x00,0x20,0x00]
uret

# CHECK-INST: sret
# CHECK: encoding: [0x73,0x00,0x20,0x10]
sret

# CHECK-INST: mret
# CHECK: encoding: [0x73,0x00,0x20,0x30]
mret

# CHECK-INST: wfi
# CHECK: encoding: [0x73,0x00,0x50,0x10]
wfi

# CHECK-INST: sfence.vma zero, zero
# CHECK: encoding: [0x73,0x00,0x00,0x12]
sfence.vma zero, zero

# CHECK-INST: sfence.vma a0, a1
# CHECK: encoding: [0x73,0x00,0xb5,0x12]
sfence.vma a0, a1
