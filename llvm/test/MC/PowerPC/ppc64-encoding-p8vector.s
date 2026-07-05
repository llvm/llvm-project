# RUN: llvm-mc -triple powerpc64-unknown-unknown --show-encoding %s | FileCheck -check-prefix=CHECK-BE %s
# RUN: llvm-mc -triple powerpc64le-unknown-unknown --show-encoding %s | FileCheck -check-prefix=CHECK-LE %s

# CHECK-BE: vpksdss 2, 3, 4                 # encoding: [0x10,0x43,0x25,0xce]
# CHECK-LE: vpksdss 2, 3, 4                 # encoding: [0xce,0x25,0x43,0x10]
            vpksdss 2, 3, 4

# CHECK-BE: vpksdus 2, 3, 4                 # encoding: [0x10,0x43,0x25,0x4e]
# CHECK-LE: vpksdus 2, 3, 4                 # encoding: [0x4e,0x25,0x43,0x10]
            vpksdus 2, 3, 4

# CHECK-BE: vpkudus 2, 3, 4                 # encoding: [0x10,0x43,0x24,0xce]
# CHECK-LE: vpkudus 2, 3, 4                 # encoding: [0xce,0x24,0x43,0x10]
            vpkudus 2, 3, 4

# CHECK-BE: vpkudum 2, 3, 4                 # encoding: [0x10,0x43,0x24,0x4e]
# CHECK-LE: vpkudum 2, 3, 4                 # encoding: [0x4e,0x24,0x43,0x10]
            vpkudum 2, 3, 4

# CHECK-BE: vupkhsw 2, 3                    # encoding: [0x10,0x40,0x1e,0x4e]
# CHECK-LE: vupkhsw 2, 3                    # encoding: [0x4e,0x1e,0x40,0x10]
            vupkhsw 2, 3

# CHECK-BE: vupklsw 2, 3                    # encoding: [0x10,0x40,0x1e,0xce]
# CHECK-LE: vupklsw 2, 3                    # encoding: [0xce,0x1e,0x40,0x10]
            vupklsw 2, 3
