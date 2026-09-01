# RUN: llvm-mc -triple=sparcv9 -filetype=obj %s | llvm-objdump -dr - | FileCheck %s --implicit-check-not=R_SPARC
# RUN: llvm-mc -triple=sparcv9 --position-independent -filetype=obj %s | llvm-objdump -dr - | FileCheck %s --implicit-check-not=R_SPARC

# CHECK-LABEL: <abs>:
# CHECK-NEXT:  sethi 0x48d15, %l5
# CHECK-NEXT:  or %l5, 0x278, %l5
# CHECK-NEXT:  sethi 0x3fb72e, %o0
# CHECK-NEXT:  xor %o0, 0x298, %o0
# CHECK-NEXT:  sethi 0x21d950, %o1
# CHECK-NEXT:  or %o1, 0x321, %o1
# CHECK-NEXT:  sethi 0x4, %o2
# CHECK-NEXT:  or %o2, 0x234, %o2
# CHECK-NEXT:  sethi 0x48d15, %o3
# CHECK-NEXT:  or %o3, 0x278, %o3

defined = 0xfedcba98

.globl abs
abs:
  sethi %hi(0x12345678), %l5
  or %l5, %lo(0x12345678), %l5
  sethi %hi(defined), %o0
  xor %o0, %lo(defined), %o0
  sethi %hi(forward), %o1
  or %o1, %lo(forward), %o1
  sethi %hi(.Lend-.Lbegin), %o2
  or %o2, %lo(.Lend-.Lbegin), %o2
  set 0x12345678, %o3

forward = 0x87654321

.data
.Lbegin:
  .space 0x1234
.Lend:
