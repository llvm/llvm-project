# RUN: llvm-mc -triple i386-linux-gnu -filetype=obj -o %t \
# RUN:   %S/Inputs/non-english-characters-comments.s
# RUN: llvm-readobj %t | FileCheck %s
# CHECK: Format: elf32-i386

# RUN: not llvm-mc -triple i386-linux-gnu -filetype=obj -o %t \
# RUN:   %S/Inputs/non-english-characters-section-name.s 2>&1 | \
# RUN:     FileCheck %s --check-prefix=ERR
# ERR: invalid character in input
