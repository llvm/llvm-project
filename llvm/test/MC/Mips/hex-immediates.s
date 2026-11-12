# RUN: llvm-mc -filetype=obj %s -triple=mips-unknown-linux \
# RUN:   | llvm-objdump -d --print-imm-hex - | FileCheck %s

# CHECK: jal   0x20
# CHECK: addiu $sp, $sp, -0x20
# CHECK: sw    $2,  0x10($fp)

jal     32
addiu   $sp, $sp, -32
sw      $2, 16($fp)
lui     $2, 2
