# RUN: llvm-mc -triple=mips -mcpu=mips32 -mattr=+micromips -filetype=obj < %s \
# RUN:   | llvm-objdump -d - | FileCheck %s

.set noreorder

# Force us into the second 256 MB region with a non-zero instruction index
.org 256*1024*1024 + 12
# CHECK-LABEL: 1000000c foo:
# CHECK-NEXT: 1000000c: d4 00 00 06                   j       12 <foo>
# CHECK-NEXT: 10000010: f4 00 00 08                   jal     16 <foo+0x4>
# CHECK-NEXT: 10000014: f0 00 00 05                   jalx    20 <foo+0x8>
# CHECK-NEXT: 10000018: 74 00 00 0c                   jals    24 <foo+0xc>
foo:
	j 12
	jal 16
	jalx 20
	jals 24
