# RUN: llvm-mc -triple powerpc64-unknown-unknown --show-encoding %s | FileCheck -check-prefix=CHECK-BE %s
# RUN: llvm-mc -triple powerpc64le-unknown-unknown --show-encoding %s | FileCheck -check-prefix=CHECK-LE %s

# Instructions specific to the e500 / e500mc cores:

# CHECK-BE: rfdi                            # encoding: [0x4c,0x00,0x00,0x4e]
# CHECK-LE: rfdi                            # encoding: [0x4e,0x00,0x00,0x4c]
            rfdi
# CHECK-BE: rfmci                            # encoding: [0x4c,0x00,0x00,0x4c]
# CHECK-LE: rfmci                            # encoding: [0x4c,0x00,0x00,0x4c]
            rfmci
