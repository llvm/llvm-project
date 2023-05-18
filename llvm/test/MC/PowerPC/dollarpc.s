# RUN: llvm-mc -triple powerpc64-unknown-unknown  --show-encoding %s | FileCheck -check-prefix=CHECK-BE %s
# RUN: llvm-mc -triple powerpc64le-unknown-unknown  --show-encoding %s | FileCheck -check-prefix=CHECK-LE %s
# RUN: llvm-mc -triple powerpc-aix-ibm-xcoff  --show-encoding %s 2>&1 | FileCheck -check-prefix=CHECK-BE %s

# CHECK-BE:                                 # encoding: [0x42,0x00,A,0bAAAAAA00]
# CHECK-LE:                                 # encoding: [0bAAAAAA00,A,0x00,0x42]
            bdnz $-8

