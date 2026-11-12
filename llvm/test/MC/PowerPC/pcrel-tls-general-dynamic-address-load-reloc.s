# RUN: llvm-mc -triple=powerpc64le-unknown-unknown -filetype=obj %s 2>&1 | \
# RUN: FileCheck %s -check-prefix=MC
# RUN: llvm-mc -triple=powerpc64le-unknown-unknown -filetype=obj %s | \
# RUN: llvm-readobj -r - | FileCheck %s -check-prefix=READOBJ

# This test checks that on Power PC we can correctly convert @got@tlsgd@pcrel
# x@tlsgd and __tls_get_addr@notoc into R_PPC64_GOT_TLSGD_PCREL34, R_PPC64_TLSGD
# and R_PPC64_REL24_NOTOC for general dynamic relocations with address loaded

# MC-NOT:    error: invalid variant

# READOBJ:        0xC R_PPC64_GOT_TLSGD_PCREL34 x 0x0
# READOBJ-NEXT:   0x14 R_PPC64_TLSGD x 0x0
# READOBJ-NEXT:   0x14 R_PPC64_REL24_NOTOC __tls_get_addr 0x0

GeneralDynamicAddrLoad:                 # @GeneralDynamicAddrLoad
	mflr 0
	std 0, 16(1)
	stdu 1, -32(1)
	paddi 3, 0, x@got@tlsgd@pcrel, 1
	bl __tls_get_addr@notoc(x@tlsgd)
	addi 1, 1, 32
	ld 0, 16(1)
	mtlr 0
	blr
