# RUN: not llvm-mc -triple riscv32 -filetype obj %s -o /dev/null 2>&1 | FileCheck %s

.Ltmp1:
	.quad	tls

	lui a0, %hi(tls+0-.Ltmp1)
# CHECK: :[[#@LINE-1]]:2: error: expected relocatable expression
	lw a0, %lo(tls+0-.Ltmp1)(t0)
# CHECK: :[[#@LINE-1]]:2: error: expected relocatable expression
	lui a0, %tprel_hi(tls+0-.Ltmp1)
# CHECK: :[[#@LINE-1]]:2: error: expected relocatable expression
	add a0, a0, tp, %tprel_add(tls+0-.Ltmp1)
# CHECK: :[[#@LINE-1]]:2: error: expected relocatable expression
	addi a0, a0, %tprel_lo(tls+0-.Ltmp1)
# CHECK: :[[#@LINE-1]]:2: error: expected relocatable expression
	auipc a0, %tls_ie_pcrel_hi(tls+0-.Ltmp1)
# CHECK: :[[#@LINE-1]]:2: error: expected relocatable expression
	auipc a0, %tls_gd_pcrel_hi(tls+0-.Ltmp1)
# CHECK: :[[#@LINE-1]]:2: error: expected relocatable expression
	auipc a0, %pcrel_hi(tls-.Ltmp1)
# CHECK: :[[#@LINE-1]]:2: error: expected relocatable expression
	auipc a0, %got_pcrel_hi(tls-.Ltmp1)
# CHECK: :[[#@LINE-1]]:2: error: expected relocatable expression
	addi a0, a0, %pcrel_lo(tls-.Ltmp1)
# CHECK: :[[#@LINE-1]]:2: error: expected relocatable expression

#	tail tls+32
#	tail tls-tls
# _ :[[#@LINE-1]]:2: error: expected relocatable expression
