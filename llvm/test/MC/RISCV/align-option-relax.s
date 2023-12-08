# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=-relax < %s \
# RUN:     | llvm-readobj -r - | FileCheck %s

# Check that .option relax overrides -mno-relax and enables R_RISCV_ALIGN
# relocations.
# CHECK: R_RISCV_ALIGN
	.option relax
	.align 4
