# RUN: llvm-mc -triple powerpc64-unknown-unknown --show-encoding %s | FileCheck -check-prefix=SERVER %s
# RUN: llvm-mc -mcpu=a2 -triple powerpc64-unknown-unknown --show-encoding %s | FileCheck -check-prefix=EMBEDDED %s

# SERVER: dcbt 2, 3, 10                   # encoding: [0x7d,0x42,0x1a,0x2c]
          dcbt 2, 3, 10
# SERVER: dcbtst 2, 3, 10                 # encoding: [0x7d,0x42,0x19,0xec]
          dcbtst 2, 3, 10

# EMBEDDED: dcbt 10, 2, 3                 # encoding: [0x7d,0x42,0x1a,0x2c]
            dcbt 10, 2, 3
# EMBEDDED: dcbtst 10, 2, 3               # encoding: [0x7d,0x42,0x19,0xec]
            dcbtst 10, 2, 3

