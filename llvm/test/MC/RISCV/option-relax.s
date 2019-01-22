# RUN: llvm-mc -triple riscv32 < %s \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 < %s \
# RUN:     | llvm-readobj -r | FileCheck -check-prefix=CHECK-RELOC %s

# RUN: llvm-mc -triple riscv64 < %s \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv64 < %s \
# RUN:     | llvm-readobj -r | FileCheck -check-prefix=CHECK-RELOC %s

# Check .option relax causes R_RISCV_RELAX to be emitted, and .option
# norelax suppresses it. Also check that if .option relax was enabled
# at any point and an instruction may have been relaxed, diff & branch
# relocations are emitted to ensure correct codegen. See
# linker-relaxation.s and fixups-expr.s for behaviour of the relax
# attribute.

.L1:
.option norelax
# CHECK-INST: .option norelax

# CHECK-INST: call foo
# CHECK-RELOC: R_RISCV_CALL foo 0x0
# CHECK-RELOC-NOT: R_RISCV_RELAX - 0x0
call foo

# CHECK-RELOC-NEXT: R_RISCV_ADD64
# CHECK-RELOC-NEXT: R_RISCV_SUB64
.dword .L2-.L1
# CHECK-RELOC-NEXT: R_RISCV_JAL
jal zero, .L1
# CHECK-RELOC-NEXT: R_RISCV_BRANCH
beq s1, s1, .L1

.L2:
.option relax
# CHECK-INST: .option relax

# CHECK-INST: call bar
# CHECK-RELOC-NEXT: R_RISCV_CALL bar 0x0
# CHECK-RELOC-NEXT: R_RISCV_RELAX - 0x0
call bar

# CHECK-RELOC-NEXT: R_RISCV_ADD64
# CHECK-RELOC-NEXT: R_RISCV_SUB64
.dword .L2-.L1
# CHECK-RELOC-NEXT: R_RISCV_JAL
jal zero, .L1
# CHECK-RELOC-NEXT: R_RISCV_BRANCH
beq s1, s1, .L1

.option norelax
# CHECK-INST: .option norelax

# CHECK-INST: call baz
# CHECK-RELOC-NEXT: R_RISCV_CALL baz 0x0
# CHECK-RELOC-NOT: R_RISCV_RELAX - 0x0
call baz

# CHECK-RELOC-NEXT: R_RISCV_ADD64
# CHECK-RELOC-NEXT: R_RISCV_SUB64
.dword .L2-.L1
# CHECK-RELOC-NEXT: R_RISCV_JAL
jal zero, .L1
# CHECK-RELOC-NEXT: R_RISCV_BRANCH
beq s1, s1, .L1
