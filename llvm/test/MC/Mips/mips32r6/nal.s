# RUN: llvm-mc %s -triple=mipsisa32r6el-linux-gnu -filetype=obj -o - | \
# RUN:		 llvm-objdump --no-print-imm-hex -d - | FileCheck %s --check-prefix=MIPS32R6-EL
# RUN: llvm-mc %s -triple=mipsisa32r6-linux-gnu -filetype=obj -o - | \
# RUN: 		 llvm-objdump --no-print-imm-hex -d - | FileCheck %s --check-prefix=MIPS32R6-EB

# Whether it is a macro or an actual instruction, it always has a delay slot.
# Ensure the delay slot is filled correctly.
# Also ensure that NAL does not reside in a forbidden slot.
# MIPS32R6-EL:		00 00 80 f8   bnezc	$4, 0x4
# MIPS32R6-EL-NEXT:	00 00 00 00   nop
# MIPS32R6-EL:		00 00 10 04   nal
# MIPS32R6-EL-NEXT:	00 00 00 00   nop
# MIPS32R6-EB:		f8 80 00 00   bnezc	$4, 0x4
# MIPS32R6-EB-NEXT:	00 00 00 00   nop
# MIPS32R6-EB:		04 10 00 00   nal
# MIPS32R6-EB-NEXT:	00 00 00 00   nop

nal_test:
	# We generate a fobidden solt just for testing.
	bnezc $a0, 0
	nal
