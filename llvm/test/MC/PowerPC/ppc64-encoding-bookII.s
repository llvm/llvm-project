
# RUN: llvm-mc -triple powerpc64-unknown-unknown --show-encoding %s | FileCheck -check-prefix=CHECK-BE %s
# RUN: llvm-mc -triple powerpc64le-unknown-unknown --show-encoding %s | FileCheck -check-prefix=CHECK-LE %s

# Cache management instruction

# CHECK-BE: icbi 2, 3                       # encoding: [0x7c,0x02,0x1f,0xac]
# CHECK-LE: icbi 2, 3                       # encoding: [0xac,0x1f,0x02,0x7c]
            icbi 2, 3

# CHECK-BE: icbt 0, 5, 31                   # encoding: [0x7c,0x05,0xf8,0x2c]
# CHECK-LE: icbt 0, 5, 31                   # encoding: [0x2c,0xf8,0x05,0x7c]
            icbt 0, 5, 31

# CHECK-BE: dcbt 2, 3, 10                   # encoding: [0x7d,0x42,0x1a,0x2c]
# CHECK-LE: dcbt 2, 3, 10                   # encoding: [0x2c,0x1a,0x42,0x7d]
            dcbt 2, 3, 10
# CHECK-BE: dcbt 2, 3, 10                   # encoding: [0x7d,0x42,0x1a,0x2c]
# CHECK-LE: dcbt 2, 3, 10                   # encoding: [0x2c,0x1a,0x42,0x7d]
            dcbtct 2, 3, 10
# CHECK-BE: dcbt 2, 3, 10                   # encoding: [0x7d,0x42,0x1a,0x2c]
# CHECK-LE: dcbt 2, 3, 10                   # encoding: [0x2c,0x1a,0x42,0x7d]
            dcbtds 2, 3, 10
# CHECK-BE: dcbt 2, 3                       # encoding: [0x7c,0x02,0x1a,0x2c]
# CHECK-LE: dcbt 2, 3                       # encoding: [0x2c,0x1a,0x02,0x7c]
            dcbt 2, 3
# CHECK-BE: dcbt 2, 3                       # encoding: [0x7c,0x02,0x1a,0x2c]
# CHECK-LE: dcbt 2, 3                       # encoding: [0x2c,0x1a,0x02,0x7c]
            dcbt 2, 3, 0
# CHECK-BE: dcbtt 2, 3                      # encoding: [0x7e,0x02,0x1a,0x2c]
# CHECK-LE: dcbtt 2, 3                      # encoding: [0x2c,0x1a,0x02,0x7e]
            dcbtt 2, 3
# CHECK-BE: dcbtt 2, 3                      # encoding: [0x7e,0x02,0x1a,0x2c]
# CHECK-LE: dcbtt 2, 3                      # encoding: [0x2c,0x1a,0x02,0x7e]
            dcbt 2, 3, 16
# CHECK-BE: dcbtst 2, 3, 10                 # encoding: [0x7d,0x42,0x19,0xec]
# CHECK-LE: dcbtst 2, 3, 10                 # encoding: [0xec,0x19,0x42,0x7d]
            dcbtst 2, 3, 10
# CHECK-BE: dcbtst 2, 3, 10                 # encoding: [0x7d,0x42,0x19,0xec]
# CHECK-LE: dcbtst 2, 3, 10                 # encoding: [0xec,0x19,0x42,0x7d]
            dcbtstct 2, 3, 10
# CHECK-BE: dcbtst 2, 3, 10                 # encoding: [0x7d,0x42,0x19,0xec]
# CHECK-LE: dcbtst 2, 3, 10                 # encoding: [0xec,0x19,0x42,0x7d]
            dcbtstds 2, 3, 10
# CHECK-BE: dcbtst 2, 3                     # encoding: [0x7c,0x02,0x19,0xec]
# CHECK-LE: dcbtst 2, 3                     # encoding: [0xec,0x19,0x02,0x7c]
            dcbtst 2, 3
# CHECK-BE: dcbtst 2, 3                     # encoding: [0x7c,0x02,0x19,0xec]
# CHECK-LE: dcbtst 2, 3                     # encoding: [0xec,0x19,0x02,0x7c]
            dcbtst 2, 3, 0
# CHECK-BE: dcbtstt 2, 3                    # encoding: [0x7e,0x02,0x19,0xec]
# CHECK-LE: dcbtstt 2, 3                    # encoding: [0xec,0x19,0x02,0x7e]
            dcbtstt 2, 3
# CHECK-BE: dcbtstt 2, 3                    # encoding: [0x7e,0x02,0x19,0xec]
# CHECK-LE: dcbtstt 2, 3                    # encoding: [0xec,0x19,0x02,0x7e]
            dcbtst 2, 3, 16
# CHECK-BE: dcbz 2, 3                       # encoding: [0x7c,0x02,0x1f,0xec]
# CHECK-LE: dcbz 2, 3                       # encoding: [0xec,0x1f,0x02,0x7c]
            dcbz 2, 3
# CHECK-BE: dcbst 2, 3                      # encoding: [0x7c,0x02,0x18,0x6c]
# CHECK-LE: dcbst 2, 3                      # encoding: [0x6c,0x18,0x02,0x7c]
            dcbst 2, 3
# CHECK-BE: dcbfl 2, 3                      # encoding: [0x7c,0x22,0x18,0xac]
# CHECK-LE: dcbfl 2, 3                      # encoding: [0xac,0x18,0x22,0x7c]
            dcbf 2, 3, 1
# CHECK-BE: dcbflp 2, 3                     # encoding: [0x7c,0x62,0x18,0xac]
# CHECK-LE: dcbflp 2, 3                     # encoding: [0xac,0x18,0x62,0x7c]
            dcbf 2, 3, 3

# Synchronization instructions

# CHECK-BE: isync                           # encoding: [0x4c,0x00,0x01,0x2c]
# CHECK-LE: isync                           # encoding: [0x2c,0x01,0x00,0x4c]
            isync

# FIXME:    stbcx. 2, 3, 4
# FIXME:    sthcx. 2, 3, 4
# CHECK-BE: stwcx. 2, 3, 4                  # encoding: [0x7c,0x43,0x21,0x2d]
# CHECK-LE: stwcx. 2, 3, 4                  # encoding: [0x2d,0x21,0x43,0x7c]
            stwcx. 2, 3, 4

# CHECK-BE: stdcx. 2, 3, 4                  # encoding: [0x7c,0x43,0x21,0xad]
# CHECK-LE: stdcx. 2, 3, 4                  # encoding: [0xad,0x21,0x43,0x7c]
            stdcx. 2, 3, 4

# CHECK-BE: stwat 2, 3, 28                  # encoding: [0x7c,0x43,0xe5,0x8c]
# CHECK-LE: stwat 2, 3, 28                  # encoding: [0x8c,0xe5,0x43,0x7c]
            stwat 2, 3, 28

# CHECK-BE: stdat 2, 3, 28                  # encoding: [0x7c,0x43,0xe5,0xcc]
# CHECK-LE: stdat 2, 3, 28                  # encoding: [0xcc,0xe5,0x43,0x7c]
            stdat 2, 3, 28

# CHECK-BE: ptesync                         # encoding: [0x7c,0x40,0x04,0xac]
# CHECK-LE: ptesync                         # encoding: [0xac,0x04,0x40,0x7c]
            sync 2
# CHECK-BE: eieio                           # encoding: [0x7c,0x00,0x06,0xac]
# CHECK-LE: eieio                           # encoding: [0xac,0x06,0x00,0x7c]
            eieio
# CHECK-BE: waitimpl                        # encoding: [0x7c,0x40,0x00,0x3c]
# CHECK-LE: waitimpl                        # encoding: [0x3c,0x00,0x40,0x7c]
            wait 2
# CHECK-BE: mbar 1                          # encoding: [0x7c,0x20,0x06,0xac]
# CHECK-LE: mbar 1                          # encoding: [0xac,0x06,0x20,0x7c]
            mbar 1
# CHECK-BE: mbar                            # encoding: [0x7c,0x00,0x06,0xac]
            mbar

# Extended mnemonics

# CHECK-BE: dcbf 2, 3                       # encoding: [0x7c,0x02,0x18,0xac]
# CHECK-LE: dcbf 2, 3                       # encoding: [0xac,0x18,0x02,0x7c]
            dcbf 2, 3
# CHECK-BE: dcbfl 2, 3                      # encoding: [0x7c,0x22,0x18,0xac]
# CHECK-LE: dcbfl 2, 3                      # encoding: [0xac,0x18,0x22,0x7c]
            dcbfl 2, 3
# CHECK-BE: dcbflp 2, 3                     # encoding: [0x7c,0x62,0x18,0xac]
# CHECK-LE: dcbflp 2, 3                     # encoding: [0xac,0x18,0x62,0x7c]
            dcbflp 2, 3

# CHECK-BE: lbarx 2, 3, 4                   # encoding: [0x7c,0x43,0x20,0x68]
# CHECK-LE: lbarx 2, 3, 4                   # encoding: [0x68,0x20,0x43,0x7c]
            lbarx 2, 3, 4

# CHECK-BE: lharx 2, 3, 4                   # encoding: [0x7c,0x43,0x20,0xe8]
# CHECK-LE: lharx 2, 3, 4                   # encoding: [0xe8,0x20,0x43,0x7c]
            lharx 2, 3, 4

# CHECK-BE: lwarx 2, 3, 4                   # encoding: [0x7c,0x43,0x20,0x28]
# CHECK-LE: lwarx 2, 3, 4                   # encoding: [0x28,0x20,0x43,0x7c]
            lwarx 2, 3, 4

# CHECK-BE: ldarx 2, 3, 4                   # encoding: [0x7c,0x43,0x20,0xa8]
# CHECK-LE: ldarx 2, 3, 4                   # encoding: [0xa8,0x20,0x43,0x7c]
            ldarx 2, 3, 4

# CHECK-BE: lbarx 2, 3, 4, 1                # encoding: [0x7c,0x43,0x20,0x69]
# CHECK-LE: lbarx 2, 3, 4, 1                # encoding: [0x69,0x20,0x43,0x7c]
            lbarx 2, 3, 4, 1

# CHECK-BE: lharx 2, 3, 4, 1                # encoding: [0x7c,0x43,0x20,0xe9]
# CHECK-LE: lharx 2, 3, 4, 1                # encoding: [0xe9,0x20,0x43,0x7c]
            lharx 2, 3, 4, 1

# CHECK-BE: lwarx 2, 3, 4, 1                # encoding: [0x7c,0x43,0x20,0x29]
# CHECK-LE: lwarx 2, 3, 4, 1                # encoding: [0x29,0x20,0x43,0x7c]
            lwarx 2, 3, 4, 1

# CHECK-BE: ldarx 2, 3, 4, 1                # encoding: [0x7c,0x43,0x20,0xa9]
# CHECK-LE: ldarx 2, 3, 4, 1                # encoding: [0xa9,0x20,0x43,0x7c]
            ldarx 2, 3, 4, 1

# CHECK-BE: lwat 2, 3, 28                   # encoding: [0x7c,0x43,0xe4,0x8c]
# CHECK-LE: lwat 2, 3, 28                   # encoding: [0x8c,0xe4,0x43,0x7c]
            lwat 2, 3, 28

# CHECK-BE: ldat 2, 3, 28                   # encoding: [0x7c,0x43,0xe4,0xcc]
# CHECK-LE: ldat 2, 3, 28                   # encoding: [0xcc,0xe4,0x43,0x7c]
            ldat 2, 3, 28

# CHECK-BE: sync                            # encoding: [0x7c,0x00,0x04,0xac]
# CHECK-LE: sync                            # encoding: [0xac,0x04,0x00,0x7c]
            sync
# CHECK-BE: sync                            # encoding: [0x7c,0x00,0x04,0xac]
# CHECK-LE: sync                            # encoding: [0xac,0x04,0x00,0x7c]
            msync
# CHECK-BE: lwsync                          # encoding: [0x7c,0x20,0x04,0xac]
# CHECK-LE: lwsync                          # encoding: [0xac,0x04,0x20,0x7c]
            lwsync
# CHECK-BE: ptesync                         # encoding: [0x7c,0x40,0x04,0xac]
# CHECK-LE: ptesync                         # encoding: [0xac,0x04,0x40,0x7c]
            ptesync

# CHECK-BE: wait                            # encoding: [0x7c,0x00,0x00,0x3c]
# CHECK-LE: wait                            # encoding: [0x3c,0x00,0x00,0x7c]
            wait
# CHECK-BE: waitrsv                         # encoding: [0x7c,0x20,0x00,0x3c]
# CHECK-LE: waitrsv                         # encoding: [0x3c,0x00,0x20,0x7c]
            waitrsv
# CHECK-BE: waitimpl                        # encoding: [0x7c,0x40,0x00,0x3c]
# CHECK-LE: waitimpl                        # encoding: [0x3c,0x00,0x40,0x7c]
            waitimpl

# Time base instructions

# CHECK-BE: mftb 2, 123                     # encoding: [0x7c,0x5b,0x1a,0xe6]
# CHECK-LE: mftb 2, 123                     # encoding: [0xe6,0x1a,0x5b,0x7c]
            mftb 2, 123
# CHECK-BE: mftb 2, 268                     # encoding: [0x7c,0x4c,0x42,0xe6]
# CHECK-LE: mftb 2, 268                     # encoding: [0xe6,0x42,0x4c,0x7c]
            mftb 2
# CHECK-BE: mftb 2, 268                     # encoding: [0x7c,0x4c,0x42,0xe6]
# CHECK-LE: mftb 2, 268                     # encoding: [0xe6,0x42,0x4c,0x7c]
            mftbl 2
# CHECK-BE: mftbu 2                         # encoding: [0x7c,0x4d,0x42,0xe6]
# CHECK-LE: mftbu 2                         # encoding: [0xe6,0x42,0x4d,0x7c]
            mftbu 2

# CHECK-BE: mttbl 3                         # encoding: [0x7c,0x7c,0x43,0xa6]
# CHECK-LE: mttbl 3                         # encoding: [0xa6,0x43,0x7c,0x7c]
            mttbl 3
# CHECK-BE: mttbu 3                         # encoding: [0x7c,0x7d,0x43,0xa6]
# CHECK-LE: mttbu 3                         # encoding: [0xa6,0x43,0x7d,0x7c]
            mttbu 3
