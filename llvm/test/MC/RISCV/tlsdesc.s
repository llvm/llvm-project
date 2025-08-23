# RUN: llvm-mc -filetype=obj -triple riscv32 < %s --defsym RV32=1  \
# RUN:   | llvm-objdump -dr -M no-aliases - \
# RUN:   | FileCheck %s --check-prefixes=INST,RV32
# RUN: llvm-mc -filetype=obj -triple riscv64 < %s \
# RUN:   | llvm-objdump -dr -M no-aliases - \
# RUN:   | FileCheck %s --check-prefixes=INST,RV64
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+relax < %s --defsym RV32=1  \
# RUN:   | llvm-objdump -dr -M no-aliases - \
# RUN:   | FileCheck %s --check-prefixes=INST,RV32,RELAX
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+relax < %s \
# RUN:   | llvm-objdump -dr -M no-aliases - \
# RUN:   | FileCheck %s --check-prefixes=INST,RV64,RELAX

# RUN: not llvm-mc -triple riscv32 < %s --defsym RV32=1 --defsym ERR=1 2>&1 \
# RUN:   | FileCheck %s --check-prefixes=ERR
# RUN: not llvm-mc -triple riscv64 < %s --defsym ERR=1 2>&1 \
# RUN:   | FileCheck %s --check-prefixes=ERR

start:                                  # @start
# %bb.0:                                # %entry
.Ltlsdesc_hi0:
	auipc a0, %tlsdesc_hi(a-4)
	# INST: auipc a0, 0x0
	# INST-NEXT: R_RISCV_TLSDESC_HI20 a-0x4
	# RELAX-NEXT: R_RISCV_RELAX
	auipc	a0, %tlsdesc_hi(unspecified)
	# INST-NEXT: auipc a0, 0x0
	# INST-NEXT: R_RISCV_TLSDESC_HI20 unspecified
	# RELAX-NEXT: R_RISCV_RELAX
.ifdef RV32
	lw	a1, %tlsdesc_load_lo(.Ltlsdesc_hi0)(a0)
	# RV32: lw a1, 0x0(a0)
	# RV32-NEXT: R_RISCV_TLSDESC_LOAD_LO12 .Ltlsdesc_hi0
.else
	ld	a1, %tlsdesc_load_lo(.Ltlsdesc_hi0)(a0)
	# RV64: ld a1, 0x0(a0)
	# RV64-NEXT: R_RISCV_TLSDESC_LOAD_LO12 .Ltlsdesc_hi0
.endif
	# RELAX-NEXT: R_RISCV_RELAX
	addi	a0, a0, %tlsdesc_add_lo(.Ltlsdesc_hi0)
	# INST: addi a0, a0, 0x0
	# INST-NEXT: R_RISCV_TLSDESC_ADD_LO12 .Ltlsdesc_hi0
	# RELAX-NEXT: R_RISCV_RELAX
	jalr	t0, 0(a1), %tlsdesc_call(.Ltlsdesc_hi0)
	# INST-NEXT: jalr t0, 0x0(a1)
	# INST-NEXT: R_RISCV_TLSDESC_CALL .Ltlsdesc_hi0
	# RELAX-NEXT: R_RISCV_RELAX
	add	a0, a0, tp
	# INST-NEXT: add a0, a0, tp
	ret

## Check invalid usage
.ifdef ERR
	auipc x1, %tlsdesc_call(foo) # ERR: :[[#@LINE]]:12: error: operand must be a symbol with a %pcrel_hi/%got_pcrel_hi/%tls_ie_pcrel_hi/%tls_gd_pcrel_hi specifier or an integer in the range
	auipc x1, %tlsdesc_call(1234) # ERR: :[[#@LINE]]:12: error: operand must be a symbol with a %pcrel_hi/%got_pcrel_hi/%tls_ie_pcrel_hi/%tls_gd_pcrel_hi specifier or an integer in the range
	auipc a0, %tlsdesc_hi(a+b) # ERR: :[[#@LINE]]:12: error: operand must be a symbol with a %pcrel_hi/%got_pcrel_hi/%tls_ie_pcrel_hi/%tls_gd_pcrel_hi specifier or an integer in the range

	lw   a0, t0, %tlsdesc_load_lo(a_symbol) # ERR: :[[#@LINE]]:15: error: invalid operand for instruction
	lw   a0, t0, %tlsdesc_load_lo(a_symbol)(a4) # ERR: :[[#@LINE]]:15: error: invalid operand for instruction

	addi a0, t0, %tlsdesc_add_lo(a_symbol)(a4) # ERR: :[[#@LINE]]:41: error: invalid operand for instruction
	addi a0, %tlsdesc_add_lo(a_symbol) # ERR: :[[#@LINE]]:11: error: invalid operand for instruction
	addi x1, %tlsdesc_load_lo(a_symbol)(a0) # ERR: :[[#@LINE]]:11: error: invalid operand for instruction

	jalr x5, 0(a1), %tlsdesc_hi(a_symbol) # ERR: :[[#@LINE]]:18: error: operand must be a symbol with %tlsdesc_call specifier
	jalr x1, 0(a1), %tlsdesc_call(a_symbol) # ERR: :[[#@LINE]]:13: error: the output operand must be t0/x5 when using %tlsdesc_call specifier
.endif
