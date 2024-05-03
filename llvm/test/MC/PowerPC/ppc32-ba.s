# RUN: llvm-mc -triple powerpc-unknown-unknown --show-encoding %s | FileCheck %s

# Check that large and/or negative immediates in 32-bit mode are accepted.

# CHECK:      ba 0xfe000000  # encoding: [0x4a,0x00,0x00,0x02]
# CHECK-NEXT: ba 0xfe000000  # encoding: [0x4a,0x00,0x00,0x02]
# CHECK-NEXT: ba 0xfffffc00  # encoding: [0x4b,0xff,0xfc,0x02]
         ba 0xfe000000
         ba (-33554432)
	 ba (-1024)
