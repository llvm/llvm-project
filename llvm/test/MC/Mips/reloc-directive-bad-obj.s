# RUN: not llvm-mc -triple mips-unknown-linux %s \
# RUN:     -target-abi=o32 -filetype=obj -o /dev/null 2>&1 | FileCheck %s
.text
nop
.reloc foo, R_MIPS_32, .text  # CHECK: :[[@LINE]]:24: error: unresolved relocation offset
nop
nop
.reloc bar, R_MIPS_32, .text  # CHECK: :[[@LINE]]:24: error: unresolved relocation offset
nop
