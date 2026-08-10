# RUN: llvm-mc -assemble -mcpu=mips64r6 -arch=mips64el -filetype=obj %s -o tmp.o
# RUN: llvm-objdump -d tmp.o | FileCheck %s --check-prefix=MIPSELR6

# MIPSELR6:      0000000000000000 <aaa>:
# MIPSELR6-NEXT: beqzc	$13, 0x0 <aaa>
# MIPSELR6-NEXT: b	0x0 <aaa>
# MIPSELR6:      0000000000000008 <bbb>:
# MIPSELR6-NEXT: beqzc	$13, 0x8 <bbb>
# MIPSELR6-NEXT: nop    <aaa>
# MIPSELR6:      b	0x8 <bbb>
	.set noreorder
aaa:
	beqzc $t1, aaa
	b aaa
	.set reorder
bbb:
	beqzc $t1, bbb
	b bbb
