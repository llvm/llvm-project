# Xqcilo - Qualcomm uC Large Offset Load Store extension
# RUN: llvm-mc %s -triple=riscv32 -mattr=+xqcilo,+xqcili -M no-aliases \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-BASE %s
# RUN: llvm-mc %s -triple=riscv32 -mattr=+xqcilo,+xqcili,+zca -M no-aliases \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-ZCA %s
# RUN: llvm-mc %s -triple=riscv32 -mattr=+xqcilo,+xqcili,+zcb -M no-aliases \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-ZCB %s
# RUN: llvm-mc %s -triple=riscv32 -mattr=+xqcilo,+xqcili,+zcb -filetype=obj \
# RUN:     | llvm-objdump -d -r --mattr=+xqcilo,+xqcili,+zcb \
# RUN:       --no-print-imm-hex - \
# RUN:     | FileCheck -check-prefix=CHECK-OBJ %s

# Basic expansion — use t1 (x6, not in GPRC) so no Zcb compression occurs.
# +xqcilo implies Zca, so qc.e.li always compresses to c.li with -M no-aliases.

# CHECK:          c.li t1, undefined
# CHECK-NEXT:     lb t1, 0(t1), %qc.access(undefined)
# CHECK-OBJ:      qc.e.li t1, 0
# CHECK-OBJ-NEXT: R_RISCV_VENDOR  QUALCOMM
# CHECK-OBJ-NEXT: R_RISCV_QC_E_32 undefined
# CHECK-OBJ-NEXT: lb  t1, 0(t1)
qc.e.lb t1, undefined

# CHECK:          c.li t1, undefined
# CHECK-NEXT:     lbu t1, 0(t1), %qc.access(undefined)
# CHECK-OBJ:      qc.e.li t1, 0
# CHECK-OBJ-NEXT: R_RISCV_VENDOR  QUALCOMM
# CHECK-OBJ-NEXT: R_RISCV_QC_E_32 undefined
# CHECK-OBJ-NEXT: lbu  t1, 0(t1)
qc.e.lbu t1, undefined

# CHECK:          c.li t1, undefined
# CHECK-NEXT:     lh t1, 0(t1), %qc.access(undefined)
# CHECK-OBJ:      qc.e.li t1, 0
# CHECK-OBJ-NEXT: R_RISCV_VENDOR  QUALCOMM
# CHECK-OBJ-NEXT: R_RISCV_QC_E_32 undefined
# CHECK-OBJ-NEXT: lh  t1, 0(t1)
qc.e.lh t1, undefined

# CHECK:          c.li t1, undefined
# CHECK-NEXT:     lhu t1, 0(t1), %qc.access(undefined)
# CHECK-OBJ:      qc.e.li t1, 0
# CHECK-OBJ-NEXT: R_RISCV_VENDOR  QUALCOMM
# CHECK-OBJ-NEXT: R_RISCV_QC_E_32 undefined
# CHECK-OBJ-NEXT: lhu  t1, 0(t1)
qc.e.lhu t1, undefined

# CHECK:          c.li t1, undefined
# CHECK-NEXT:     lw t1, 0(t1), %qc.access(undefined)
# CHECK-OBJ:      qc.e.li t1, 0
# CHECK-OBJ-NEXT: R_RISCV_VENDOR  QUALCOMM
# CHECK-OBJ-NEXT: R_RISCV_QC_E_32 undefined
# CHECK-OBJ-NEXT: lw  t1, 0(t1)
qc.e.lw t1, undefined

# Stores: address register t1 (non-GPRC), no Zcb compression.

# CHECK:          c.li t1, undefined
# CHECK-NEXT:     sb a0, 0(t1), %qc.access(undefined)
# CHECK-OBJ:      qc.e.li t1, 0
# CHECK-OBJ-NEXT: R_RISCV_VENDOR  QUALCOMM
# CHECK-OBJ-NEXT: R_RISCV_QC_E_32 undefined
# CHECK-OBJ-NEXT: sb  a0, 0(t1)
qc.e.sb a0, undefined, t1

# CHECK:          c.li t1, undefined
# CHECK-NEXT:     sh a0, 0(t1), %qc.access(undefined)
# CHECK-OBJ:      qc.e.li t1, 0
# CHECK-OBJ-NEXT: R_RISCV_VENDOR  QUALCOMM
# CHECK-OBJ-NEXT: R_RISCV_QC_E_32 undefined
# CHECK-OBJ-NEXT: sh  a0, 0(t1)
qc.e.sh a0, undefined, t1

# CHECK:          c.li t1, undefined
# CHECK-NEXT:     sw a0, 0(t1), %qc.access(undefined)
# CHECK-OBJ:      qc.e.li t1, 0
# CHECK-OBJ-NEXT: R_RISCV_VENDOR  QUALCOMM
# CHECK-OBJ-NEXT: R_RISCV_QC_E_32 undefined
# CHECK-OBJ-NEXT: sw  a0, 0(t1)
qc.e.sw a0, undefined, t1

# lw with a GPRC register (a0 = x10): +xqcilo implies Zca, so lw always
# compresses to c.lw in all runs. The objdump shows the c.lw alias as lw.

# CHECK:          c.li a0, undefined
# CHECK-NEXT:     c.lw a0, 0(a0), %qc.access(undefined)
# CHECK-OBJ:      qc.e.li a0, 0
# CHECK-OBJ-NEXT: R_RISCV_VENDOR  QUALCOMM
# CHECK-OBJ-NEXT: R_RISCV_QC_E_32 undefined
# CHECK-OBJ-NEXT: lw  a0, 0(a0)
qc.e.lw a0, undefined

# sw with GPRC registers: +xqcilo implies Zca, so sw always compresses to
# c.sw in all runs. The objdump shows the c.sw alias as sw.

# CHECK:          c.li a1, undefined
# CHECK-NEXT:     c.sw a0, 0(a1), %qc.access(undefined)
# CHECK-OBJ:      qc.e.li a1, 0
# CHECK-OBJ-NEXT: R_RISCV_VENDOR  QUALCOMM
# CHECK-OBJ-NEXT: R_RISCV_QC_E_32 undefined
# CHECK-OBJ-NEXT: sw  a0, 0(a1)
qc.e.sw a0, undefined, a1

# lbu/lh/lhu with GPRC registers: compressed to c.lbu/c.lh/c.lhu only with Zcb.
# The objdump shows the Zcb compressed forms as their unaliased equivalents.

# CHECK:           c.li a0, undefined
# CHECK-BASE-NEXT: lbu a0, 0(a0), %qc.access(undefined)
# CHECK-ZCA-NEXT:  lbu a0, 0(a0), %qc.access(undefined)
# CHECK-ZCB-NEXT:  c.lbu a0, 0(a0), %qc.access(undefined)
# CHECK-OBJ:       qc.e.li a0, 0
# CHECK-OBJ-NEXT:  R_RISCV_VENDOR  QUALCOMM
# CHECK-OBJ-NEXT:  R_RISCV_QC_E_32 undefined
# CHECK-OBJ-NEXT:  lbu  a0, 0(a0)
qc.e.lbu a0, undefined

# CHECK:           c.li a0, undefined
# CHECK-BASE-NEXT: lh a0, 0(a0), %qc.access(undefined)
# CHECK-ZCA-NEXT:  lh a0, 0(a0), %qc.access(undefined)
# CHECK-ZCB-NEXT:  c.lh a0, 0(a0), %qc.access(undefined)
# CHECK-OBJ:       qc.e.li a0, 0
# CHECK-OBJ-NEXT:  R_RISCV_VENDOR  QUALCOMM
# CHECK-OBJ-NEXT:  R_RISCV_QC_E_32 undefined
# CHECK-OBJ-NEXT:  lh  a0, 0(a0)
qc.e.lh a0, undefined

# CHECK:           c.li a0, undefined
# CHECK-BASE-NEXT: lhu a0, 0(a0), %qc.access(undefined)
# CHECK-ZCA-NEXT:  lhu a0, 0(a0), %qc.access(undefined)
# CHECK-ZCB-NEXT:  c.lhu a0, 0(a0), %qc.access(undefined)
# CHECK-OBJ:       qc.e.li a0, 0
# CHECK-OBJ-NEXT:  R_RISCV_VENDOR  QUALCOMM
# CHECK-OBJ-NEXT:  R_RISCV_QC_E_32 undefined
# CHECK-OBJ-NEXT:  lhu  a0, 0(a0)
qc.e.lhu a0, undefined

# sb/sh with GPRC registers: compressed to c.sb/c.sh only with Zcb.
# The objdump shows the Zcb compressed forms as their unaliased equivalents.

# CHECK:           c.li a1, undefined
# CHECK-BASE-NEXT: sb a0, 0(a1), %qc.access(undefined)
# CHECK-ZCA-NEXT:  sb a0, 0(a1), %qc.access(undefined)
# CHECK-ZCB-NEXT:  c.sb a0, 0(a1), %qc.access(undefined)
# CHECK-OBJ:       qc.e.li a1, 0
# CHECK-OBJ-NEXT:  R_RISCV_VENDOR  QUALCOMM
# CHECK-OBJ-NEXT:  R_RISCV_QC_E_32 undefined
# CHECK-OBJ-NEXT:  sb  a0, 0(a1)
qc.e.sb a0, undefined, a1

# CHECK:           c.li a1, undefined
# CHECK-BASE-NEXT: sh a0, 0(a1), %qc.access(undefined)
# CHECK-ZCA-NEXT:  sh a0, 0(a1), %qc.access(undefined)
# CHECK-ZCB-NEXT:  c.sh a0, 0(a1), %qc.access(undefined)
# CHECK-OBJ:       qc.e.li a1, 0
# CHECK-OBJ-NEXT:  R_RISCV_VENDOR  QUALCOMM
# CHECK-OBJ-NEXT:  R_RISCV_QC_E_32 undefined
# CHECK-OBJ-NEXT:  sh  a0, 0(a1)
qc.e.sh a0, undefined, a1
