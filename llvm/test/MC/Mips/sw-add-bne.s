# RUN: llvm-mc -assemble -mcpu=mips32r6 -arch=mipsel -filetype=obj %s -o tmp.o
# RUN: llvm-objdump -d tmp.o | FileCheck %s --check-prefix=MIPSELR6

# MIPSELR6:      00000000 <xxx>:
# MIPSELR6-NEXT: addiu $2, $2, 0x4 <xxx+0x4>
# MIPSELR6-NEXT: sw $4, -0x4($2)
# MIPSELR6-NEXT: bne $2, $3, 0x0 <xxx>
# MIPSELR6-NEXT: nop <xxx>
xxx:
$BB0_1:                                 # %for.body
        sw      $4, 0($2)
        addiu   $2, $2, 4
        bne     $2, $3, $BB0_1
