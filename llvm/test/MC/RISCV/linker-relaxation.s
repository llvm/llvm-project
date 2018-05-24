# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+relax < %s \
# RUN:     | llvm-readobj -r | FileCheck -check-prefix=RELAX-RELOC %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=-relax < %s \
# RUN:     | llvm-readobj -r | FileCheck -check-prefix=NORELAX-RELOC %s
# RUN: llvm-mc -triple riscv32 -mattr=+relax < %s -show-encoding \
# RUN:     | FileCheck -check-prefix=RELAX-FIXUP %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+relax < %s \
# RUN:     | llvm-readobj -r | FileCheck -check-prefix=RELAX-RELOC %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=-relax < %s \
# RUN:     | llvm-readobj -r | FileCheck -check-prefix=NORELAX-RELOC %s
# RUN: llvm-mc -triple riscv64 -mattr=+relax < %s -show-encoding \
# RUN:     | FileCheck -check-prefix=RELAX-FIXUP %s

.long foo

.L1:
call foo
# NORELAX-RELOC: R_RISCV_CALL foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_CALL foo 0x0
# RELAX-RELOC: R_RISCV_RELAX foo 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: foo, kind: fixup_riscv_relax
# RELAX-FIXUP: fixup B - offset: 0, value: foo, kind:
beq s1, s1, .L1
# RELAX-RELOC: R_RISCV_BRANCH .L1 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: .L1, kind: fixup_riscv_branch
