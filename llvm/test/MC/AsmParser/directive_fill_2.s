# RUN: llvm-mc -filetype=obj -defsym=I=6 -triple i686-unknown-unknown %s | llvm-objdump --triple=i686-unknown-unknown -s - | FileCheck --check-prefix="CHECK" %s
# RUN: not llvm-mc -filetype=obj -defsym=I=4 -triple i686-unknown-unknown %s -o /dev/null 2>&1 | FileCheck --check-prefix="CHECK-ERR" %s



# CHECK: Contents of section .text
# CHECK-NEXT: 0000 e9810000 00cc9090 90909090 90909090

# Make sure we emit in correct endianness.

# CHECK: Contents of section .data
# CHECK-NEXT: 0000 78563412 78563412 78563412

.text
foo:
jmp bar2
# CHECK-ERR: [[@LINE+1]]:7: error: invalid number of bytes
.fill ((I+foo) - .), 1, 0xcc
bar:
 .space 128, 0x90
bar2:
.byte 0xff

# This fill length is not known at assembler time.

.if (I==6)

.data
.long 0x12345678
.fill ((foo+8)-bar), 4, 0x12345678

.endif
