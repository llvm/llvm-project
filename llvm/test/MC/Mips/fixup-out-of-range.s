# RUN: not llvm-mc -triple mips64 -filetype obj %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:

# CHECK: :[[#@LINE+1]]:1: error: fixup value out of range [-32768, 65535]
addiu $t2, $t3, v_32769
addiu $t2, $t3, v_32768
addiu $t2, $t3, v65535
# CHECK: :[[#@LINE+1]]:1: error: fixup value out of range [-32768, 65535]
addiu $t2, $t3, v65536

v_32769 = -32769
v_32768 = -32768
v65535 = 65535
v65536 = 65536
