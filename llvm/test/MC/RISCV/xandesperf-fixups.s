# RUN: llvm-mc -triple riscv32 -mattr=+xandesperf -M no-aliases < %s -show-encoding \
# RUN:     | FileCheck -check-prefix=CHECK-FIXUP %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+xandesperf < %s \
# RUN:     | llvm-objdump --mattr=+xandesperf --no-print-imm-hex -M no-aliases -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INSTR %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+xandesperf %s \
# RUN:     | llvm-readobj -r - | FileCheck %s -check-prefix=CHECK-REL

.LBB0:

.fill 1000

nds.bbc t0, 7, .LBB0
# CHECK-FIXUP: fixup A - offset: 0, value: .LBB0, kind: fixup_riscv_nds_branch_10
# CHECK-INSTR: nds.bbc t0, 7, 0

nds.bbs t0, 7, .LBB1
# CHECK-FIXUP: fixup A - offset: 0, value: .LBB1, kind: fixup_riscv_nds_branch_10
# CHECK-INSTR: nds.bbs t0, 7, 0x7e0

nds.beqc t0, 7, .LBB0
# CHECK-FIXUP: fixup A - offset: 0, value: .LBB0, kind: fixup_riscv_nds_branch_10
# CHECK-INSTR: nds.beqc t0, 7, 0

nds.bnec t0, 7, .LBB1
# CHECK-FIXUP: fixup A - offset: 0, value: .LBB1, kind: fixup_riscv_nds_branch_10
# CHECK-INSTR: nds.bnec t0, 7, 0x7e0

.fill 1000

.LBB1:

# CHECK-REL-NOT: R_RISCV
