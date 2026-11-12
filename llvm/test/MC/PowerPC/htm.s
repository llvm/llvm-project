# RUN: llvm-mc -triple powerpc64-unknown-linux-gnu --show-encoding %s | FileCheck -check-prefix=CHECK-BE %s
# RUN: llvm-mc -triple powerpc64le-unknown-linux-gnu --show-encoding %s | FileCheck -check-prefix=CHECK-LE %s

# CHECK-BE: tbegin. 0                      # encoding: [0x7c,0x00,0x05,0x1d]
# CHECK-LE: tbegin. 0                      # encoding: [0x1d,0x05,0x00,0x7c]
            tbegin. 0
# CHECK-BE: tbegin. 1                      # encoding: [0x7c,0x20,0x05,0x1d]
# CHECK-LE: tbegin. 1                      # encoding: [0x1d,0x05,0x20,0x7c]
            tbegin. 1

# CHECK-BE: tend.                          # encoding: [0x7c,0x00,0x05,0x5d]
# CHECK-LE: tend.                          # encoding: [0x5d,0x05,0x00,0x7c]
            tend. 0
# CHECK-BE: tendall.                       # encoding: [0x7e,0x00,0x05,0x5d]
# CHECK-LE: tendall.                       # encoding: [0x5d,0x05,0x00,0x7e]
            tend. 1

# CHECK-BE: tabort. 9                      # encoding: [0x7c,0x09,0x07,0x1d]
# CHECK-LE: tabort. 9                      # encoding: [0x1d,0x07,0x09,0x7c]
            tabort. 9
# CHECK-BE: tabortdc. 0, 9, 9              # encoding: [0x7c,0x09,0x4e,0x5d]
# CHECK-LE: tabortdc. 0, 9, 9              # encoding: [0x5d,0x4e,0x09,0x7c]
            tabortdc. 0, 9, 9
# CHECK-BE: tabortdci. 0, 9, 0             # encoding: [0x7c,0x09,0x06,0xdd]
# CHECK-LE: tabortdci. 0, 9, 0             # encoding: [0xdd,0x06,0x09,0x7c]
            tabortdci. 0, 9, 0
# CHECK-BE: tabortwc. 0, 9, 9              # encoding: [0x7c,0x09,0x4e,0x1d]
# CHECK-LE: tabortwc. 0, 9, 9              # encoding: [0x1d,0x4e,0x09,0x7c]
            tabortwc. 0, 9, 9
# CHECK-BE: tabortwci. 0, 9, 0             # encoding: [0x7c,0x09,0x06,0x9d]
# CHECK-LE: tabortwci. 0, 9, 0             # encoding: [0x9d,0x06,0x09,0x7c]
            tabortwci. 0, 9, 0

# CHECK-BE: tsuspend.                      # encoding: [0x7c,0x00,0x05,0xdd]
# CHECK-LE: tsuspend.                      # encoding: [0xdd,0x05,0x00,0x7c]
            tsr. 0
# CHECK-BE: tresume.                       # encoding: [0x7c,0x20,0x05,0xdd]
# CHECK-LE: tresume.                       # encoding: [0xdd,0x05,0x20,0x7c]
            tsr. 1

# CHECK-BE: tcheck 0                       # encoding: [0x7c,0x00,0x05,0x9c]
# CHECK-LE: tcheck 0                       # encoding: [0x9c,0x05,0x00,0x7c]
            tcheck 0
# CHECK-BE: tcheck 3                       # encoding: [0x7d,0x80,0x05,0x9c]
# CHECK-LE: tcheck 3                       # encoding: [0x9c,0x05,0x80,0x7d]
            tcheck 3

# CHECK-BE: treclaim. 9                    # encoding: [0x7c,0x09,0x07,0x5d]
# CHECK-LE: treclaim. 9                    # encoding: [0x5d,0x07,0x09,0x7c]
            treclaim. 9
# CHECK-BE: trechkpt.                      # encoding: [0x7c,0x00,0x07,0xdd]
# CHECK-LE: trechkpt.                      # encoding: [0xdd,0x07,0x00,0x7c]
            trechkpt.
