# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s -o - \
# RUN:   | llvm-objdump -d - | FileCheck %s

# CHECK:      <foo>:
# CHECK-NEXT:        0:       0c 00 00 02     jal     0x8 <loc1>
# CHECK-NEXT:        4:       00 00 00 00     nop
#
# CHECK:      <loc1>:
# CHECK-NEXT:        8:       0c 00 00 06     jal     0x18 <loc3>
# CHECK-NEXT:        c:       00 00 00 00     nop
#
# CHECK:      <loc2>:
# CHECK-NEXT:       10:       10 00 ff fd     b	      0x8 <loc1>
# CHECK-NEXT:       14:       00 00 00 00     nop
#
# CHECK:      <loc3>:
# CHECK-NEXT:       18:       10 43 ff fd     beq     $2, $3, 0x10 <loc2>
# CHECK-NEXT:       1c:       00 00 00 00     nop
# CHECK-NEXT:       20:       04 11 ff f9     bal     0x8 <loc1>
# CHECK-NEXT:       24:       00 00 00 00     nop
# CHECK-NEXT:       28:       08 00 00 04     j       0x10 <loc2>

  .text
  .globl foo
  .ent foo
foo:
  jal loc1
loc1:
  jal loc3
loc2:
  b   loc1
loc3:
  beq $2, $3, loc2
  bal loc1
  j   loc2
  .end foo
