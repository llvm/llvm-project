# RUN: llvm-mc --disassemble %s -triple=sparcv9 2>&1 | FileCheck %s

0xff 0xdc 0xba 0x98
# CHECK: [[@LINE-1]]:1: warning: invalid instruction encoding

0xff 0xdc 0xba 0x98
# CHECK: [[@LINE-1]]:1: warning: invalid instruction encoding
