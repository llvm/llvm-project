# RUN: llvm-mc -triple=mips -mcpu=mips32 -filetype=obj < %s \
# RUN:   | llvm-objdump -d - | FileCheck %s
# RUN: llvm-mc -triple=mips64 -mcpu=mips64 -filetype=obj < %s \
# RUN:   | llvm-objdump -d - | FileCheck %s

.set noreorder

# Force us into the second 256 MB region with a non-zero instruction index
.org 256*1024*1024 + 12
# CHECK-LABEL: 1000000c foo:
# CHECK-NEXT: 1000000c: 08 00 00 03                   j       12 <foo>
# CHECK-NEXT: 10000010: 0c 00 00 04                   jal     16 <foo+0x4>
# CHECK-NEXT: 10000014: 74 00 00 05                   jalx    20 <foo+0x8>
foo:
	j 12
	jal 16
	jalx 20
