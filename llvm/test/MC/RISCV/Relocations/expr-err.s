# RUN: llvm-mc --verify -triple riscv32 -filetype obj %s -o /dev/null

.Ltmp1:
	.quad	tls

lui a0, %hi(tls+0-.Ltmp1) # expected-error:10 {{expected relocatable expression}}
lw a0, %lo(tls+0-.Ltmp1)(t0) # expected-error:9 {{expected relocatable expression}}
lui a0, %tprel_hi(tls+0-.Ltmp1) # expected-error:10 {{expected relocatable expression}}
add a0, a0, tp, %tprel_add(tls+0-.Ltmp1) # expected-error:18 {{expected relocatable expression}}
addi a0, a0, %tprel_lo(tls+0-.Ltmp1) # expected-error:15 {{expected relocatable expression}}
auipc a0, %tls_ie_pcrel_hi(tls+0-.Ltmp1) # expected-error:12 {{expected relocatable expression}}
auipc a0, %tls_gd_pcrel_hi(tls+0-.Ltmp1) # expected-error:12 {{expected relocatable expression}}
auipc a0, %pcrel_hi(tls-.Ltmp1) # expected-error:12 {{expected relocatable expression}}
auipc a0, %got_pcrel_hi(tls-.Ltmp1) # expected-error:12 {{expected relocatable expression}}
addi a0, a0, %pcrel_lo(tls-.Ltmp1) # expected-error:15 {{expected relocatable expression}}

#	tail tls+32
#	tail tls-tls
# _ :[[#@LINE-1]]:2: error: expected relocatable expression
