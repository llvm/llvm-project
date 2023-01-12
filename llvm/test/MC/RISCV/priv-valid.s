# RUN: llvm-mc %s -triple=riscv32 -mattr=+svinval -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+svinval -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+svinval < %s \
# RUN:     | llvm-objdump --mattr=+svinval -M no-aliases -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+svinval < %s \
# RUN:     | llvm-objdump --mattr=+svinval -M no-aliases -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s

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

# CHECK-INST: sinval.vma zero, zero
# CHECK: encoding: [0x73,0x00,0x00,0x16]
sinval.vma zero, zero

# CHECK-INST: sinval.vma a0, a1
# CHECK: encoding: [0x73,0x00,0xb5,0x16]
sinval.vma a0, a1

# CHECK-INST: sfence.w.inval
# CHECK: encoding: [0x73,0x00,0x00,0x18]
sfence.w.inval

# CHECK-INST: sfence.inval.ir
# CHECK: encoding: [0x73,0x00,0x10,0x18]
sfence.inval.ir
