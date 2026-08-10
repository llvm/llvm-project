# RUN: llvm-mc -triple=powerpc64le-unknown-unknown -filetype=obj %s 2>&1 | \
# RUN: FileCheck %s -check-prefix=MC
# RUN: llvm-mc -triple=powerpc64le-unknown-unknown -filetype=obj %s | \
# RUN: llvm-readobj -r - | FileCheck %s -check-prefix=READOBJ

# This test checks that on Power PC we can correctly convert x@got@tprel@pcrel
# and x@tls@pcrel into R_PPC64_GOT_TPREL_PCREL34, and R_PPC64_TLS for initial
# exec relocations with the value loaded.
# Note that with R_PPC64_TLS relocations, to distinguish PC relative
# TLS the relocation has a field value displaced by one byte from the
# beginning of the instruction.

# MC-NOT:    error: invalid variant

# READOBJ:        0x0 R_PPC64_GOT_TPREL_PCREL34 x 0x0
# READOBJ-NEXT:   0x9 R_PPC64_TLS x 0x0

InitialExecLoad:
	pld 3, x@got@tprel@pcrel(0), 1
	lwzx 3, 3, x@tls@pcrel
	blr
