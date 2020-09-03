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
# regardless of the fragment containing the target address, provided symbol
# binding allows it.

function:
.Lpcrel_label1:
	auipc	a0, %pcrel_hi(local_function)
	addi	a1, a0, %pcrel_lo(.Lpcrel_label1)
# NORELAX: auipc	a0, 0
# NORELAX-NOT: R_RISCV
# NORELAX: addi	a1, a0, 16
# NORELAX-NOT: R_RISCV

# RELAX: auipc	a0, 0
# RELAX: R_RISCV_PCREL_HI20	local_function
# RELAX: R_RISCV_RELAX	*ABS*
# RELAX: addi	a1, a0, 0
# RELAX: R_RISCV_PCREL_LO12_I	.Lpcrel_label1
# RELAX: R_RISCV_RELAX	*ABS*

	.p2align	2   # Cause a new fragment be emitted here
.Lpcrel_label2:
	auipc	a0, %pcrel_hi(local_function)
	addi	a1, a0, %pcrel_lo(.Lpcrel_label2)
# NORELAX: auipc	a0, 0
# NORELAX-NOT: R_RISCV
# NORELAX: addi	a1, a0, 8
# NORELAX-NOT: R_RISCV

# RELAX: auipc	a0, 0
# RELAX: R_RISCV_PCREL_HI20	local_function
# RELAX: R_RISCV_RELAX	*ABS*
# RELAX: addi	a1, a0, 0
# RELAX: R_RISCV_PCREL_LO12_I	.Lpcrel_label2
# RELAX: R_RISCV_RELAX	*ABS*

	.type	local_function,@function
local_function:
	ret

# Check we correctly evaluate when fixups are in different fragments

.Lpcrel_label3:
	auipc	a0, %pcrel_hi(local_function)
	.p2align	2   # Cause a new fragment be emitted here
	addi	a1, a0, %pcrel_lo(.Lpcrel_label3)
# NORELAX: auipc	a0, 0
# NORELAX-NOT: R_RISCV
# NORELAX: addi	a1, a0, -4
# NORELAX-NOT: R_RISCV

# RELAX: auipc	a0, 0
# RELAX: R_RISCV_PCREL_HI20	local_function
# RELAX: R_RISCV_RELAX	*ABS*
# RELAX: addi	a1, a0, 0
# RELAX: R_RISCV_PCREL_LO12_I	.Lpcrel_label3
# RELAX: R_RISCV_RELAX	*ABS*

# Check handling of symbol binding.

.Lpcrel_label4:
	auipc	a0, %pcrel_hi(global_function)
	addi	a1, a0, %pcrel_lo(.Lpcrel_label4)
# NORELAX: auipc	a0, 0
# NORELAX: R_RISCV_PCREL_HI20	global_function
# NORELAX: addi	a1, a0, 0
# NORELAX: R_RISCV_PCREL_LO12_I	.Lpcrel_label4

# RELAX: auipc	a0, 0
# RELAX: R_RISCV_PCREL_HI20	global_function
# RELAX: R_RISCV_RELAX	*ABS*
# RELAX: addi	a1, a0, 0
# RELAX: R_RISCV_PCREL_LO12_I	.Lpcrel_label4
# RELAX: R_RISCV_RELAX	*ABS*

.Lpcrel_label5:
	auipc	a0, %pcrel_hi(weak_function)
	addi	a1, a0, %pcrel_lo(.Lpcrel_label5)
# NORELAX: auipc	a0, 0
# NORELAX: R_RISCV_PCREL_HI20	weak_function
# NORELAX: addi	a1, a0, 0
# NORELAX: R_RISCV_PCREL_LO12_I	.Lpcrel_label5

# RELAX: auipc	a0, 0
# RELAX: R_RISCV_PCREL_HI20	weak_function
# RELAX: R_RISCV_RELAX	*ABS*
# RELAX: addi	a1, a0, 0
# RELAX: R_RISCV_PCREL_LO12_I	.Lpcrel_label5
# RELAX: R_RISCV_RELAX	*ABS*

	.global	global_function
	.type	global_function,@function
global_function:
	ret

	.weak	weak_function
	.type	weak_function,@function
weak_function:
	ret
