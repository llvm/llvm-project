# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=hexagon-unknown-elf -filetype=obj -defsym GOT=1 -o %t/got.o %s
# RUN: llvm-mc -triple=hexagon-unknown-elf -filetype=obj -defsym TLS=1 -o %t/tls.o %s
# RUN: llvm-mc -triple=hexagon-unknown-elf -filetype=obj -defsym GOTREL=1 -o %t/gotrel.o %s
# RUN: llvm-mc -triple=hexagon-unknown-elf -filetype=obj -defsym GPREL=1 -o %t/gprel.o %s
# RUN: not llvm-rtdyld -triple=hexagon-unknown-elf -verify -check=%s %t/got.o 2>&1 \
# RUN:   | FileCheck %s --check-prefix=GOT
# RUN: not llvm-rtdyld -triple=hexagon-unknown-elf -verify -check=%s %t/tls.o 2>&1 \
# RUN:   | FileCheck %s --check-prefix=TLS
# RUN: not llvm-rtdyld -triple=hexagon-unknown-elf -verify -check=%s %t/gotrel.o 2>&1 \
# RUN:   | FileCheck %s --check-prefix=GOTREL
# RUN: not llvm-rtdyld -triple=hexagon-unknown-elf -verify -check=%s %t/gprel.o 2>&1 \
# RUN:   | FileCheck %s --check-prefix=GPREL

# RuntimeDyld has no GOT, TLS or GP support. Relocations that depend on them
# must be rejected rather than resolved as a plain S + A.

	.text
	.globl main
main:
.ifdef GOT
	{ r0 = add(r1, ##sym@GOT) }
.endif
.ifdef TLS
	{ r0 = add(r1, ##sym@TPREL) }
.endif
.ifdef GOTREL
	{ r0 = ##sym@GOTREL }
.endif
.ifdef GPREL
	{ r0 = memw(gp+#sym) }
.endif

# rtdyld-check: *{4}(main) = *{4}(main)

# GOT: Unsupported Hexagon relocation R_HEX_GOT_32_6_X
# TLS: Unsupported Hexagon relocation R_HEX_TPREL_32_6_X
# GOTREL: Unsupported Hexagon relocation R_HEX_GOTREL_32_6_X
# GPREL: Unsupported Hexagon relocation R_HEX_GPREL16_2
