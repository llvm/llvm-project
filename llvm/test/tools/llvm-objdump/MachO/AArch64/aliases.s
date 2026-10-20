# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %s -o %t
# RUN: llvm-objdump --macho -d -M no-aliases %t | FileCheck %s
# RUN: llvm-objdump --macho -d --disassembler-options=no-aliases %t | FileCheck %s

# CHECK: orr w1, wzr, w2

# RUN: llvm-objdump --macho -d %t | FileCheck %s --check-prefix=ALIAS

# ALIAS: mov w1, w2

# RUN: not llvm-objdump --macho -d -M unknown %t 2>&1 | FileCheck %s -DFILE=%t --check-prefix=ERR

# ERR: error: '[[FILE]]': unrecognized disassembler option: unknown

mov w1, w2
