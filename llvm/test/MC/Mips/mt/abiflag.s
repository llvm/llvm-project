# RUN: llvm-mc < %s -arch=mips -mcpu=mips32r2 -mattr=+mt -filetype=obj -o - \
# RUN:   | llvm-readobj -A | FileCheck %s

# Test that the usage of the MT ASE is recorded in .MIPS.abiflags

# CHECK: ASEs
# CHECK-NEXT: MT (0x40)

 .text
  nop
