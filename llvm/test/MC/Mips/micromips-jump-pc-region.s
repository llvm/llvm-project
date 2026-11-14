# RUN: llvm-mc -triple=mips -mcpu=mips32 -mattr=+micromips -filetype=obj < %s \
# RUN:   | llvm-objdump -d - | FileCheck %s

.set noreorder
# 256 MB region for Mips target, with non-zero offset
.org 12
foo_mips:

# Force us into the second 128 MB region with a non-zero instruction index
.org 128*1024*1024 + 12
# CHECK-LABEL: 800000c <foo>:
# CHECK-NEXT: 800000c: d4 00 00 06                   j       0xc <foo>
# CHECK-NEXT: 8000010: f4 00 00 08                   jal     0x10 <foo+0x4>
# CHECK-NEXT: 8000014: f0 00 00 05                   jalx    0x14 <foo_mips+0x8>
# CHECK-NEXT: 8000018: 74 00 00 0c                   jals    0x18 <foo+0xc>
# CHECK-LABEL: 800001c <nonbranch>:
# CHECK-NEXT: 800001c: 30 e2 00 0a                   addiu   $7, $2, 0xa{{$}}
# CHECK-NEXT: 8000020: 50 64 00 0c                   ori     $3, $4, 0xc{{$}}
# CHECK-NEXT: 8000024: 00 a6 40 00                   sll     $5, $6, 0x8{{$}}
# CHECK-NEXT: 8000028: ee 8f                         li16    $5, 0xf{{$}}

foo:
	j 12
	jal 16
	jalx 20 # call to 256MB aligned Mips region
	jals 24

nonbranch:
	addiu $7, $2, 10
	ori   $3, $4, 12
	sll   $5, $6, 8
	li16  $5, 15
