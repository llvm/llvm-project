# Check that we don't accidentally strip addr32 prefix

# RUN: llvm-mc -disassemble %s -triple=x86_64 | FileCheck %s
# CHECK: addr32 callq
0x67 0xe8 0x00 0x00 0x00 0x00
