# RUN: llvm-mc -triple=powerpc64le-unknown-unknown -filetype=obj %s 2>&1 | \
# RUN: FileCheck %s -check-prefix=MC
# RUN: llvm-mc -triple=powerpc64le-unknown-unknown -filetype=obj %s | \
# RUN: llvm-readobj -r - | FileCheck %s -check-prefix=READOBJ

# This test checks that on Power PC we can correctly convert x@TPREL
# into R_PPC64_TPREL34 for local exec relocations with address loaded.

# MC-NOT:    error: invalid variant

# READOBJ:        0x0 R_PPC64_TPREL34 x 0x0

LocalExec:
	paddi 3, 13, x@TPREL, 0
	blr
