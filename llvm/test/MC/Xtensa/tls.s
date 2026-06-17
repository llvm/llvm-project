# RUN: llvm-mc -triple=xtensa --mcpu=esp32 --filetype=obj < %s \
# RUN: | llvm-objdump -r -s --triple=xtensa --mcpu=esp32 - | FileCheck -check-prefix=XTENSA-CHECK-OBJ %s
# RUN: llvm-mc -triple=xtensa --mcpu=esp32 --filetype=asm < %s | FileCheck -check-prefix=XTENSA-CHECK-ASM %s

# XTENSA-CHECK-OBJ-LABEL: RELOCATION RECORDS FOR [.literal]:
# XTENSA-CHECK-OBJ:       OFFSET   TYPE                     VALUE
# XTENSA-CHECK-OBJ-NEXT:  00000000 R_XTENSA_TLS_TPOFF       tls_var
# XTENSA-CHECK-OBJ-LABEL: RELOCATION RECORDS FOR [.literal.get_tls]:
# XTENSA-CHECK-OBJ:       OFFSET   TYPE                     VALUE
# XTENSA-CHECK-OBJ-NEXT:  00000000 R_XTENSA_TLS_TPOFF       tls_var

# XTENSA-CHECK-ASM-LABEL: .literal_position
# XTENSA-CHECK-ASM:       .literal .LCPI0_0, tls_var@TPOFF
# XTENSA-CHECK-ASM-LABEL: get_tls:
# XTENSA-CHECK-ASM-NEXT:  entry	a1, 32
# XTENSA-CHECK-ASM-NEXT:  or	a7, a1, a1
# XTENSA-CHECK-ASM-NEXT:  l32r	a8, .LCPI0_0
# XTENSA-CHECK-ASM-NEXT:  rur	a9, threadptr
# XTENSA-CHECK-ASM-NEXT:  add	a8, a9, a8
# XTENSA-CHECK-ASM-NEXT:  l32i	a2, a8, 0
# XTENSA-CHECK-ASM-NEXT:  .literal .Ltmp0, tls_var@TPOFF
# XTENSA-CHECK-ASM-NEXT:  l32r	a3, .Ltmp0
# XTENSA-CHECK-ASM-NEXT:  retw.n

	.literal_position
	.literal .LCPI0_0, tls_var@TPOFF
	.text
	.section	.text.get_tls,"ax",@progbits
	.global	get_tls                         # -- Begin function get_tls
	.p2align	2
	.type	get_tls,@function
get_tls:
	entry	a1, 32
	or	a7, a1, a1
	l32r	a8, .LCPI0_0
	rur	a9, threadptr
	add	a8, a9, a8
	l32i	a2, a8, 0
	movi a3, tls_var@TPOFF
	retw.n
.Lfunc_end0:
	.size	get_tls, .Lfunc_end0-get_tls
                                        # -- End function
	.type	tls_var,@object                 # @tls_var
	.section	.tdata,"awT",@progbits
	.global	tls_var
	.p2align	2, 0x0
tls_var:
	.long	42                              # 0x2a
	.size	tls_var, 4
