# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=powerpc-unknown-linux-gnu -filetype=obj -o %t/ppc32_reloc.o %s
# RUN: llvm-rtdyld -triple=powerpc-unknown-linux-gnu -verify \
# RUN:   -dummy-extern external=0x12345678 -check=%s %t/ppc32_reloc.o

	.text
	.globl	caller
	.p2align	2
	.type	caller,@function
caller:
# R_PPC_REL24 to a function of the same object is resolved directly.
# rtdyld-check: decode_operand(call_local, 0) = (callee - call_local) >> 2
call_local:
	bl callee

# R_PPC_REL24 to an external symbol goes through a stub that loads the
# address of the symbol into r12.
# rtdyld-check: decode_operand(call_extern, 0) = (stub_addr(ppc32_reloc.o/.text, external) - call_extern) >> 2
# rtdyld-check: (*{4}(stub_addr(ppc32_reloc.o/.text, external) + 0)) [15:0] = external [31:16]
# rtdyld-check: (*{4}(stub_addr(ppc32_reloc.o/.text, external) + 4)) [15:0] = external [15:0]
# rtdyld-check: *{4}(stub_addr(ppc32_reloc.o/.text, external) + 8) = 0x7D8903A6
# rtdyld-check: *{4}(stub_addr(ppc32_reloc.o/.text, external) + 12) = 0x4E800420
call_extern:
	bl external

# The addend of R_PPC_PLTREL24 is not an offset from the symbol, so this call
# uses the same stub.
# rtdyld-check: decode_operand(call_plt, 0) = (stub_addr(ppc32_reloc.o/.text, external) - call_plt) >> 2
call_plt:
	bl external+32768@PLT

# R_PPC_REL16_HA and R_PPC_REL16_LO
# rtdyld-check: decode_operand(rel16_ha, 2) = (object - rel16_ha + 0x8000) [31:16]
rel16_ha:
	addis 3, 3, object-rel16_ha@ha
# rtdyld-check: decode_operand(rel16_lo, 2) = (object - rel16_ha) [15:0]
rel16_lo:
	addi 3, 3, object-rel16_ha@l
	blr
	.size	caller, .-caller

	.globl	callee
	.p2align	2
	.type	callee,@function
callee:
	blr
	.size	callee, .-callee

	.data
	.globl	object
	.p2align	2
object:
	.long	0

# R_PPC_ADDR32
# rtdyld-check: *{4}addr32 = callee
addr32:
	.long	callee

# R_PPC_REL32
# rtdyld-check: *{4}rel32 = (callee - rel32) [31:0]
rel32:
	.long	callee-.
