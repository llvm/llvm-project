# RUN: llvm-mc -triple powerpc64-unknown-linux-gnu --filetype=obj -o - %s | \
# RUN:   llvm-objdump -D  -r - | FileCheck --check-prefix=CHECK-BE %s
# RUN: llvm-mc -triple powerpc64le-unknown-linux-gnu --filetype=obj -o - %s | \
# RUN:   llvm-objdump -D  -r - | FileCheck --check-prefix=CHECK-LE %s

# The purpose of this test is to check that when an alignment nop is added
# it is added correctly with resepect to the labels in the .s file.
# The test contains 3 labels at the end (1:, 2:, 3:). The label 2: is on the
# same line as an unaligned 8 byte instruction. The desired behaviour is to have
# the alignment nop inserted after the 1: label but before the 2: label. The
# branch to 1: should jump to 3c: and the branch to 2: should jump to 40:.

	.text
_start:
	b 1f;
	b 2f;
	b 3f;
# CHECK-BE:      0:       48 00 00 3c
# CHECK-BE-NEXT: 4:       48 00 00 3c
# CHECK-BE-NEXT: 8:       48 00 00 40
# CHECK-LE:      0:       3c 00 00 48
# CHECK-LE-NEXT: 4:       3c 00 00 48
# CHECK-LE-NEXT: 8:       40 00 00 48
	trap
	trap
	trap
	trap
	trap
	trap
	trap
	trap
	trap
	trap
	trap
	trap
1:
2:	paddi 1, 2, 8589934576, 0  # 8 Byte Instruction
3:
	blr
# CHECK-BE:      3c:       60 00 00 00     nop
# CHECK-BE-NEXT: 40:       06 01 ff ff
# CHECK-BE-NEXT: 44:       38 22 ff f0
# CHECK-BE-NEXT: 48:       4e 80 00 20
# CHECK-LE:      3c:       00 00 00 60     nop
# CHECK-LE-NEXT: 40:       ff ff 01 06
# CHECK-LE-NEXT: 44:       f0 ff 22 38
# CHECK-LE-NEXT: 48:       20 00 80 4e

