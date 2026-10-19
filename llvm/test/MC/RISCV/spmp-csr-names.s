# RUN: llvm-mc %s -triple=riscv32 -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-INST,CHECK-ENC %s
# RUN: llvm-mc -filetype=obj -triple riscv32 < %s \
# RUN:     | llvm-objdump -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST-ALIAS %s
# RUN: llvm-mc %s -triple=riscv64 -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-INST,CHECK-ENC %s
# RUN: llvm-mc -filetype=obj -triple riscv64 < %s \
# RUN:     | llvm-objdump -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST-ALIAS %s
# RUN: llvm-mc %s -triple=riscv32 -defsym=RV32=1 -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=RV32-INST,RV32-ENC %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -defsym=RV32=1 < %s \
# RUN:     | llvm-objdump -d - \
# RUN:     | FileCheck -check-prefix=RV32-ALIAS %s
# RUN: not llvm-mc %s -triple=riscv64 -defsym=RV32=1 2>&1 \
# RUN:     | FileCheck -check-prefix=CHECK-RV64-ERR %s

# spmpen
# CHECK-INST: csrrs t1, spmpen, zero
# CHECK-ENC: encoding: [0x73,0x23,0x30,0x18]
# CHECK-INST-ALIAS: csrr t1, spmpen
# CHECK-INST: csrrs t2, spmpen, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x30,0x18]
# CHECK-INST-ALIAS: csrr t2, spmpen
csrrs t1, spmpen, zero
csrrs t2, 0x183, zero

# mpmpdeleg
# CHECK-INST: csrrs t1, mpmpdeleg, zero
# CHECK-ENC: encoding: [0x73,0x23,0x60,0x31]
# CHECK-INST-ALIAS: csrr t1, mpmpdeleg
# CHECK-INST: csrrs t2, mpmpdeleg, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x60,0x31]
# CHECK-INST-ALIAS: csrr t2, mpmpdeleg
csrrs t1, mpmpdeleg, zero
csrrs t2, 0x316, zero

.ifdef RV32
# spmpenh
# RV32-INST: csrrs t1, spmpenh, zero
# RV32-ENC: encoding: [0x73,0x23,0x30,0x19]
# RV32-ALIAS: csrr t1, spmpenh
# RV32-INST: csrrs t2, spmpenh, zero
# RV32-ENC: encoding: [0xf3,0x23,0x30,0x19]
# RV32-ALIAS: csrr t2, spmpenh
# CHECK-RV64-ERR: error: system register 'spmpenh' is RV32 only
csrrs t1, spmpenh, zero
csrrs t2, 0x193, zero
.endif
