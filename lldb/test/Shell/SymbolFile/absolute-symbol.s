# REQUIRES: system-darwin
# RUN: %clang %s -c -o %t.o
# RUN: %lldb -b -o 'target modules lookup -s absolute_symbol' %t.o | FileCheck %s
# CHECK: 1 symbols match 'absolute_symbol'
# CHECK:   Address: 0x0000000012345678 (0x0000000012345678)
# CHECK:   Summary: 0x0000000012345678
.globl absolute_symbol
absolute_symbol = 0x12345678
