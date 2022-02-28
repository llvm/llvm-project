# RUN: llvm-mc %s -triple=riscv32 -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv64 -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 < %s \
# RUN:     | llvm-objdump -M no-aliases -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv64 < %s \
# RUN:     | llvm-objdump -M no-aliases -d - \
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

# CHECK-INST: hfence.vvma zero, zero
# CHECK: encoding: [0x73,0x00,0x00,0x22]
hfence.vvma zero, zero

# CHECK-INST: hfence.vvma a0, a1
# CHECK: encoding: [0x73,0x00,0xb5,0x22]
hfence.vvma a0, a1

# CHECK-INST: hfence.gvma zero, zero
# CHECK: encoding: [0x73,0x00,0x00,0x62]
hfence.gvma zero, zero

# CHECK-INST: hfence.gvma a0, a1
# CHECK: encoding: [0x73,0x00,0xb5,0x62]
hfence.gvma a0, a1

# CHECK-INST: hinval.vvma zero, zero
# CHECK: encoding: [0x73,0x00,0x00,0x26]
hinval.vvma zero, zero

# CHECK-INST: hinval.vvma a0, a1
# CHECK: encoding: [0x73,0x00,0xb5,0x26]
hinval.vvma a0, a1

# CHECK-INST: hinval.gvma zero, zero
# CHECK: encoding: [0x73,0x00,0x00,0x66]
hinval.gvma zero, zero

# CHECK-INST: hinval.gvma a0, a1
# CHECK: encoding: [0x73,0x00,0xb5,0x66]
hinval.gvma a0, a1
