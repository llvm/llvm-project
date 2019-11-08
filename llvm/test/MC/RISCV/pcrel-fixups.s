# RUN: llvm-mc -triple riscv32 -mattr=-relax -filetype obj %s \
# RUN:    | llvm-objdump -M no-aliases -d -r - \
# RUN:    | FileCheck --check-prefix NORELAX %s
# RUN: llvm-mc -triple riscv32 -mattr=+relax -filetype obj %s \
# RUN:    | llvm-objdump -M no-aliases -d -r - \
# RUN:    | FileCheck --check-prefix RELAX %s
# RUN: llvm-mc -triple riscv64 -mattr=-relax -filetype obj %s \
# RUN:    | llvm-objdump -M no-aliases -d -r - \
# RUN:    | FileCheck --check-prefix NORELAX %s
# RUN: llvm-mc -triple riscv64 -mattr=+relax -filetype obj %s \
# RUN:    | llvm-objdump -M no-aliases -d -r - \
# RUN:    | FileCheck --check-prefix RELAX %s

# Fixups for %pcrel_hi / %pcrel_lo can be evaluated within a section,
# regardless of the fragment containing the target address.

function:
.Lpcrel_label1:
	auipc	a0, %pcrel_hi(other_function)
	addi	a1, a0, %pcrel_lo(.Lpcrel_label1)
# NORELAX: auipc	a0, 0
# NORELAX-NOT: R_RISCV
# NORELAX: addi	a1, a0, 16
# NORELAX-NOT: R_RISCV

# RELAX: auipc	a0, 0
# RELAX: R_RISCV_PCREL_HI20	other_function
# RELAX: R_RISCV_RELAX	*ABS*
# RELAX: addi	a1, a0, 0
# RELAX: R_RISCV_PCREL_LO12_I	.Lpcrel_label1
# RELAX: R_RISCV_RELAX	*ABS*

	.p2align	2   # Cause a new fragment be emitted here
.Lpcrel_label2:
	auipc	a0, %pcrel_hi(other_function)
	addi	a1, a0, %pcrel_lo(.Lpcrel_label2)
# NORELAX: auipc	a0, 0
# NORELAX-NOT: R_RISCV
# NORELAX: addi	a1, a0, 8
# NORELAX-NOT: R_RISCV

# RELAX: auipc	a0, 0
# RELAX: R_RISCV_PCREL_HI20	other_function
# RELAX: R_RISCV_RELAX	*ABS*
# RELAX: addi	a1, a0, 0
# RELAX: R_RISCV_PCREL_LO12_I	.Lpcrel_label2
# RELAX: R_RISCV_RELAX	*ABS*

	.type	other_function,@function
other_function:
	ret

