# RUN: llvm-mc %s -triple=mipsel-linux-gnu -filetype=obj -o - | \
# RUN:		 llvm-objdump --no-print-imm-hex -d - | FileCheck %s --check-prefix=MIPS32-EL
# RUN: llvm-mc %s -triple=mips-linux-gnu -filetype=obj -o - | \
# RUN:		 llvm-objdump --no-print-imm-hex -d - | FileCheck %s --check-prefix=MIPS32-EB

# Whether it is a macro or an actual instruction, it always has a delay slot.
# Ensure the delay slot is filled correctly.
# MIPS32-EL:		00 00 10 04   bltzal  $zero, 0x4
# MIPS32-EL-NEXT: 	00 00 00 00   nop
# MIPS32-EB:		04 10 00 00   bltzal  $zero, 0x4
# MIPS32-EB-NEXT:	00 00 00 00   nop

nal_test:
	nal
