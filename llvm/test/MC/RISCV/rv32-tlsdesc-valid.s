# RUN: llvm-mc -filetype=obj -triple riscv32 < %s | llvm-objdump -d -M no-aliases - | FileCheck %s

start:                                  # @start
# %bb.0:                                # %entry
.Ltlsdesc_hi0:
	auipc	a0, %tlsdesc_hi(unspecified)
	#CHECK: auipc a0, 0x0
	lw	a1, %tlsdesc_load_lo(.Ltlsdesc_hi0)(a0)
	#CHECK: lw a1, 0x0(a0)
	addi	a0, a0, %tlsdesc_add_lo(.Ltlsdesc_hi0)
	#CHECK: addi a0, a0, 0x0
	jalr	t0, 0(a1), %tlsdesc_call(.Ltlsdesc_hi0)
	#CHECK: jalr t0, 0x0(a1)
	add	a0, a0, tp
	#CHECK: add a0, a0, tp
	ret

