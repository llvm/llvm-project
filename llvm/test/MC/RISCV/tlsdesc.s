# RUN: llvm-mc -filetype=obj -triple riscv32 < %s --defsym RV32=1  | llvm-objdump -dr -M no-aliases - | FileCheck %s --check-prefixes=INST,RV32
# RUN: llvm-mc -filetype=obj -triple riscv64 < %s | llvm-objdump -dr -M no-aliases - | FileCheck %s --check-prefixes=INST,RV64

# RUN: llvm-mc -filetype=obj -triple riscv32 < %s --mattr=+relax --defsym RV32=1  | llvm-objdump -dr -M no-aliases - | FileCheck %s --check-prefixes=INST,RELAX,RV32
# RUN: llvm-mc -filetype=obj -triple riscv64 < %s --mattr=+relax | llvm-objdump -dr -M no-aliases - | FileCheck %s --check-prefixes=INST,RELAX,RV64


# RUN: not llvm-mc -triple riscv32 < %s --defsym RV32=1 --defsym ERR=1 2>&1 | FileCheck %s --check-prefixes=ERR,ERR-RV32
# RUN: not llvm-mc -triple riscv64 < %s --defsym ERR=1 2>&1 | FileCheck %s --check-prefixes=ERR,ERR-RV64

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
	addi	a0, a0, %tlsdesc_add_lo(.Ltlsdesc_hi0)
	# INST: addi a0, a0, 0x0
	# INST-NEXT: R_RISCV_TLSDESC_ADD_LO12 .Ltlsdesc_hi0
	jalr	t0, 0(a1), %tlsdesc_call(.Ltlsdesc_hi0)
	# INST-NEXT: jalr t0, 0x0(a1)
	# INST-NEXT: R_RISCV_TLSDESC_CALL .Ltlsdesc_hi0
	add	a0, a0, tp
	# INST-NEXT: add a0, a0, tp
	ret

## Check invalid usage
.ifdef ERR
	auipc x1, %tlsdesc_call(foo)
# ERR: :[[#@LINE-1]]:12: error: operand must be a symbol with a %pcrel_hi/%got_pcrel_hi/%tls_ie_pcrel_hi/%tls_gd_pcrel_hi specifier or an integer in the range
	auipc x1, %tlsdesc_call(1234)
# ERR: :[[#@LINE-1]]:12: error: operand must be a symbol with a %pcrel_hi/%got_pcrel_hi/%tls_ie_pcrel_hi/%tls_gd_pcrel_hi specifier or an integer in the range
	auipc a0, %tlsdesc_hi(a+b)
# ERR: :[[#@LINE-1]]:12: error: operand must be a symbol with a %pcrel_hi/%got_pcrel_hi/%tls_ie_pcrel_hi/%tls_gd_pcrel_hi specifier or an integer in the range

	lw   a0, t0, %tlsdesc_load_lo(a_symbol)
# ERR: :[[#@LINE-1]]:15: error: unexpected extra operand for instruction
	lw   a0, t0, %tlsdesc_load_lo(a_symbol)(a4)
# ERR-RV32: :[[#@LINE-1]]:2: error: invalid instruction, any one of the following would fix this:
# ERR-RV32: :[[#@LINE-2]]:15: note: unexpected extra operand for instruction
# ERR-RV32: :[[#@LINE-3]]:11: note: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]
# ERR-RV32: :[[#@LINE-4]]:11: note: immediate must be an integer in the range [-2048, 2047]
# ERR-RV64: :[[#@LINE-5]]:2: error: invalid instruction, any one of the following would fix this:
# ERR-RV64: :[[#@LINE-6]]:15: note: unexpected extra operand for instruction
# ERR-RV64: :[[#@LINE-7]]:11: note: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]
# ERR-RV64: :[[#@LINE-8]]:2: note: instruction requires the following: RV32I Base Instruction Set

	addi a0, t0, %tlsdesc_add_lo(a_symbol)(a4)
# ERR: :[[#@LINE-1]]:41: error: unexpected extra operand for instruction
	addi a0, %tlsdesc_add_lo(a_symbol)
# ERR: :[[#@LINE-1]]:11: error: register must be a GPR
	addi x1, %tlsdesc_load_lo(a_symbol)(a0)
# ERR: :[[#@LINE-1]]:11: error: register must be a GPR

	jalr x5, 0(a1), %tlsdesc_hi(a_symbol)
# ERR: :[[#@LINE-1]]:2: error: invalid instruction, any one of the following would fix this:
# ERR: :[[#@LINE-2]]:18: note: unexpected extra operand for instruction
# ERR: :[[#@LINE-3]]:18: note: operand must be a symbol with %tlsdesc_call specifier
	jalr x1, 0(a1), %tlsdesc_call(a_symbol)
# ERR: :[[#@LINE-1]]:13: error: the output operand must be t0/x5 when using %tlsdesc_call specifier
.endif
