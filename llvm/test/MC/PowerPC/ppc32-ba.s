# RUN: llvm-mc -triple powerpc-unknown-unknown --show-encoding %s | FileCheck %s

# Check that large immediates in 32bit mode are accepted.

# CHECK: ba -33554432 # encoding: [0x4a,0x00,0x00,0x02]
         ba 0xfe000000
