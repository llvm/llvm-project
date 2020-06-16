
# RUN: llvm-mc -triple powerpc64-unknown-unknown --show-encoding %s | FileCheck -check-prefix=CHECK-BE %s
# RUN: llvm-mc -triple powerpc64le-unknown-unknown --show-encoding %s | FileCheck -check-prefix=CHECK-LE %s

# Condition register bit symbols

# CHECK-BE: beqlr 0                         # encoding: [0x4d,0x82,0x00,0x20]
# CHECK-LE: beqlr 0                         # encoding: [0x20,0x00,0x82,0x4d]
            beqlr cr0
# CHECK-BE: beqlr 1                         # encoding: [0x4d,0x86,0x00,0x20]
# CHECK-LE: beqlr 1                         # encoding: [0x20,0x00,0x86,0x4d]
            beqlr cr1
# CHECK-BE: beqlr 2                         # encoding: [0x4d,0x8a,0x00,0x20]
# CHECK-LE: beqlr 2                         # encoding: [0x20,0x00,0x8a,0x4d]
            beqlr cr2
# CHECK-BE: beqlr 3                         # encoding: [0x4d,0x8e,0x00,0x20]
# CHECK-LE: beqlr 3                         # encoding: [0x20,0x00,0x8e,0x4d]
            beqlr cr3
# CHECK-BE: beqlr 4                         # encoding: [0x4d,0x92,0x00,0x20]
# CHECK-LE: beqlr 4                         # encoding: [0x20,0x00,0x92,0x4d]
            beqlr cr4
# CHECK-BE: beqlr 5                         # encoding: [0x4d,0x96,0x00,0x20]
# CHECK-LE: beqlr 5                         # encoding: [0x20,0x00,0x96,0x4d]
            beqlr cr5
# CHECK-BE: beqlr 6                         # encoding: [0x4d,0x9a,0x00,0x20]
# CHECK-LE: beqlr 6                         # encoding: [0x20,0x00,0x9a,0x4d]
            beqlr cr6
# CHECK-BE: beqlr 7                         # encoding: [0x4d,0x9e,0x00,0x20]
# CHECK-LE: beqlr 7                         # encoding: [0x20,0x00,0x9e,0x4d]
            beqlr cr7

# CHECK-BE: bclr 12, 0                      # encoding: [0x4d,0x80,0x00,0x20]
# CHECK-LE: bclr 12, 0                      # encoding: [0x20,0x00,0x80,0x4d]
            btlr 4*cr0+lt
# CHECK-BE: bclr 12, 1                      # encoding: [0x4d,0x81,0x00,0x20]
# CHECK-LE: bclr 12, 1                      # encoding: [0x20,0x00,0x81,0x4d]
            btlr 4*cr0+gt
# CHECK-BE: bclr 12, 2                      # encoding: [0x4d,0x82,0x00,0x20]
# CHECK-LE: bclr 12, 2                      # encoding: [0x20,0x00,0x82,0x4d]
            btlr 4*cr0+eq
# CHECK-BE: bclr 12, 3                      # encoding: [0x4d,0x83,0x00,0x20]
# CHECK-LE: bclr 12, 3                      # encoding: [0x20,0x00,0x83,0x4d]
            btlr 4*cr0+so
# CHECK-BE: bclr 12, 3                      # encoding: [0x4d,0x83,0x00,0x20]
# CHECK-LE: bclr 12, 3                      # encoding: [0x20,0x00,0x83,0x4d]
            btlr 4*cr0+un
# CHECK-BE: bclr 12, 4                      # encoding: [0x4d,0x84,0x00,0x20]
# CHECK-LE: bclr 12, 4                      # encoding: [0x20,0x00,0x84,0x4d]
            btlr 4*cr1+lt
# CHECK-BE: bclr 12, 5                      # encoding: [0x4d,0x85,0x00,0x20]
# CHECK-LE: bclr 12, 5                      # encoding: [0x20,0x00,0x85,0x4d]
            btlr 4*cr1+gt
# CHECK-BE: bclr 12, 6                      # encoding: [0x4d,0x86,0x00,0x20]
# CHECK-LE: bclr 12, 6                      # encoding: [0x20,0x00,0x86,0x4d]
            btlr 4*cr1+eq
# CHECK-BE: bclr 12, 7                      # encoding: [0x4d,0x87,0x00,0x20]
# CHECK-LE: bclr 12, 7                      # encoding: [0x20,0x00,0x87,0x4d]
            btlr 4*cr1+so
# CHECK-BE: bclr 12, 7                      # encoding: [0x4d,0x87,0x00,0x20]
# CHECK-LE: bclr 12, 7                      # encoding: [0x20,0x00,0x87,0x4d]
            btlr 4*cr1+un
# CHECK-BE: bclr 12, 8                      # encoding: [0x4d,0x88,0x00,0x20]
# CHECK-LE: bclr 12, 8                      # encoding: [0x20,0x00,0x88,0x4d]
            btlr 4*cr2+lt
# CHECK-BE: bclr 12, 9                      # encoding: [0x4d,0x89,0x00,0x20]
# CHECK-LE: bclr 12, 9                      # encoding: [0x20,0x00,0x89,0x4d]
            btlr 4*cr2+gt
# CHECK-BE: bclr 12, 10                     # encoding: [0x4d,0x8a,0x00,0x20]
# CHECK-LE: bclr 12, 10                     # encoding: [0x20,0x00,0x8a,0x4d]
            btlr 4*cr2+eq
# CHECK-BE: bclr 12, 11                     # encoding: [0x4d,0x8b,0x00,0x20]
# CHECK-LE: bclr 12, 11                     # encoding: [0x20,0x00,0x8b,0x4d]
            btlr 4*cr2+so
# CHECK-BE: bclr 12, 11                     # encoding: [0x4d,0x8b,0x00,0x20]
# CHECK-LE: bclr 12, 11                     # encoding: [0x20,0x00,0x8b,0x4d]
            btlr 4*cr2+un
# CHECK-BE: bclr 12, 12                     # encoding: [0x4d,0x8c,0x00,0x20]
# CHECK-LE: bclr 12, 12                     # encoding: [0x20,0x00,0x8c,0x4d]
            btlr 4*cr3+lt
# CHECK-BE: bclr 12, 13                     # encoding: [0x4d,0x8d,0x00,0x20]
# CHECK-LE: bclr 12, 13                     # encoding: [0x20,0x00,0x8d,0x4d]
            btlr 4*cr3+gt
# CHECK-BE: bclr 12, 14                     # encoding: [0x4d,0x8e,0x00,0x20]
# CHECK-LE: bclr 12, 14                     # encoding: [0x20,0x00,0x8e,0x4d]
            btlr 4*cr3+eq
# CHECK-BE: bclr 12, 15                     # encoding: [0x4d,0x8f,0x00,0x20]
# CHECK-LE: bclr 12, 15                     # encoding: [0x20,0x00,0x8f,0x4d]
            btlr 4*cr3+so
# CHECK-BE: bclr 12, 15                     # encoding: [0x4d,0x8f,0x00,0x20]
# CHECK-LE: bclr 12, 15                     # encoding: [0x20,0x00,0x8f,0x4d]
            btlr 4*cr3+un
# CHECK-BE: bclr 12, 16                     # encoding: [0x4d,0x90,0x00,0x20]
# CHECK-LE: bclr 12, 16                     # encoding: [0x20,0x00,0x90,0x4d]
            btlr 4*cr4+lt
# CHECK-BE: bclr 12, 17                     # encoding: [0x4d,0x91,0x00,0x20]
# CHECK-LE: bclr 12, 17                     # encoding: [0x20,0x00,0x91,0x4d]
            btlr 4*cr4+gt
# CHECK-BE: bclr 12, 18                     # encoding: [0x4d,0x92,0x00,0x20]
# CHECK-LE: bclr 12, 18                     # encoding: [0x20,0x00,0x92,0x4d]
            btlr 4*cr4+eq
# CHECK-BE: bclr 12, 19                     # encoding: [0x4d,0x93,0x00,0x20]
# CHECK-LE: bclr 12, 19                     # encoding: [0x20,0x00,0x93,0x4d]
            btlr 4*cr4+so
# CHECK-BE: bclr 12, 19                     # encoding: [0x4d,0x93,0x00,0x20]
# CHECK-LE: bclr 12, 19                     # encoding: [0x20,0x00,0x93,0x4d]
            btlr 4*cr4+un
# CHECK-BE: bclr 12, 20                     # encoding: [0x4d,0x94,0x00,0x20]
# CHECK-LE: bclr 12, 20                     # encoding: [0x20,0x00,0x94,0x4d]
            btlr 4*cr5+lt
# CHECK-BE: bclr 12, 21                     # encoding: [0x4d,0x95,0x00,0x20]
# CHECK-LE: bclr 12, 21                     # encoding: [0x20,0x00,0x95,0x4d]
            btlr 4*cr5+gt
# CHECK-BE: bclr 12, 22                     # encoding: [0x4d,0x96,0x00,0x20]
# CHECK-LE: bclr 12, 22                     # encoding: [0x20,0x00,0x96,0x4d]
            btlr 4*cr5+eq
# CHECK-BE: bclr 12, 23                     # encoding: [0x4d,0x97,0x00,0x20]
# CHECK-LE: bclr 12, 23                     # encoding: [0x20,0x00,0x97,0x4d]
            btlr 4*cr5+so
# CHECK-BE: bclr 12, 23                     # encoding: [0x4d,0x97,0x00,0x20]
# CHECK-LE: bclr 12, 23                     # encoding: [0x20,0x00,0x97,0x4d]
            btlr 4*cr5+un
# CHECK-BE: bclr 12, 24                     # encoding: [0x4d,0x98,0x00,0x20]
# CHECK-LE: bclr 12, 24                     # encoding: [0x20,0x00,0x98,0x4d]
            btlr 4*cr6+lt
# CHECK-BE: bclr 12, 25                     # encoding: [0x4d,0x99,0x00,0x20]
# CHECK-LE: bclr 12, 25                     # encoding: [0x20,0x00,0x99,0x4d]
            btlr 4*cr6+gt
# CHECK-BE: bclr 12, 26                     # encoding: [0x4d,0x9a,0x00,0x20]
# CHECK-LE: bclr 12, 26                     # encoding: [0x20,0x00,0x9a,0x4d]
            btlr 4*cr6+eq
# CHECK-BE: bclr 12, 27                     # encoding: [0x4d,0x9b,0x00,0x20]
# CHECK-LE: bclr 12, 27                     # encoding: [0x20,0x00,0x9b,0x4d]
            btlr 4*cr6+so
# CHECK-BE: bclr 12, 27                     # encoding: [0x4d,0x9b,0x00,0x20]
# CHECK-LE: bclr 12, 27                     # encoding: [0x20,0x00,0x9b,0x4d]
            btlr 4*cr6+un
# CHECK-BE: bclr 12, 28                     # encoding: [0x4d,0x9c,0x00,0x20]
# CHECK-LE: bclr 12, 28                     # encoding: [0x20,0x00,0x9c,0x4d]
            btlr 4*cr7+lt
# CHECK-BE: bclr 12, 29                     # encoding: [0x4d,0x9d,0x00,0x20]
# CHECK-LE: bclr 12, 29                     # encoding: [0x20,0x00,0x9d,0x4d]
            btlr 4*cr7+gt
# CHECK-BE: bclr 12, 30                     # encoding: [0x4d,0x9e,0x00,0x20]
# CHECK-LE: bclr 12, 30                     # encoding: [0x20,0x00,0x9e,0x4d]
            btlr 4*cr7+eq
# CHECK-BE: bclr 12, 31                     # encoding: [0x4d,0x9f,0x00,0x20]
# CHECK-LE: bclr 12, 31                     # encoding: [0x20,0x00,0x9f,0x4d]
            btlr 4*cr7+so
# CHECK-BE: bclr 12, 31                     # encoding: [0x4d,0x9f,0x00,0x20]
# CHECK-LE: bclr 12, 31                     # encoding: [0x20,0x00,0x9f,0x4d]
            btlr 4*cr7+un

# Branch mnemonics

# CHECK-BE: blr                             # encoding: [0x4e,0x80,0x00,0x20]
# CHECK-LE: blr                             # encoding: [0x20,0x00,0x80,0x4e]
            blr
# CHECK-BE: bctr                            # encoding: [0x4e,0x80,0x04,0x20]
# CHECK-LE: bctr                            # encoding: [0x20,0x04,0x80,0x4e]
            bctr
# CHECK-BE: blrl                            # encoding: [0x4e,0x80,0x00,0x21]
# CHECK-LE: blrl                            # encoding: [0x21,0x00,0x80,0x4e]
            blrl
# CHECK-BE: bctrl                           # encoding: [0x4e,0x80,0x04,0x21]
# CHECK-LE: bctrl                           # encoding: [0x21,0x04,0x80,0x4e]
            bctrl

# CHECK-BE: bt 2, target                    # encoding: [0x41,0x82,A,0bAAAAAA00]
# CHECK-LE: bt 2, target                    # encoding: [0bAAAAAA00,A,0x82,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bt 2, target
# CHECK-BE: bta 2, target                   # encoding: [0x41,0x82,A,0bAAAAAA10]
# CHECK-LE: bta 2, target                   # encoding: [0bAAAAAA10,A,0x82,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bta 2, target
# CHECK-BE: bclr 12, 2                      # encoding: [0x4d,0x82,0x00,0x20]
# CHECK-LE: bclr 12, 2                      # encoding: [0x20,0x00,0x82,0x4d]
            btlr 2
# CHECK-BE: bcctr 12, 2                     # encoding: [0x4d,0x82,0x04,0x20]
# CHECK-LE: bcctr 12, 2                     # encoding: [0x20,0x04,0x82,0x4d]
            btctr 2
# CHECK-BE: btl 2, target                   # encoding: [0x41,0x82,A,0bAAAAAA01]
# CHECK-LE: btl 2, target                   # encoding: [0bAAAAAA01,A,0x82,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            btl 2, target
# CHECK-BE: btla 2, target                  # encoding: [0x41,0x82,A,0bAAAAAA11]
# CHECK-LE: btla 2, target                  # encoding: [0bAAAAAA11,A,0x82,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            btla 2, target
# CHECK-BE: bclrl 12, 2                     # encoding: [0x4d,0x82,0x00,0x21]
# CHECK-LE: bclrl 12, 2                     # encoding: [0x21,0x00,0x82,0x4d]
            btlrl 2
# CHECK-BE: bcctrl 12, 2                    # encoding: [0x4d,0x82,0x04,0x21]
# CHECK-LE: bcctrl 12, 2                    # encoding: [0x21,0x04,0x82,0x4d]
            btctrl 2

# CHECK-BE: bt+ 2, target                   # encoding: [0x41,0xe2,A,0bAAAAAA00]
# CHECK-LE: bt+ 2, target                   # encoding: [0bAAAAAA00,A,0xe2,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bt+ 2, target
# CHECK-BE: bta+ 2, target                  # encoding: [0x41,0xe2,A,0bAAAAAA10]
# CHECK-LE: bta+ 2, target                  # encoding: [0bAAAAAA10,A,0xe2,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bta+ 2, target
# CHECK-BE: bclr 15, 2                      # encoding: [0x4d,0xe2,0x00,0x20]
# CHECK-LE: bclr 15, 2                      # encoding: [0x20,0x00,0xe2,0x4d]
            btlr+ 2
# CHECK-BE: bcctr 15, 2                     # encoding: [0x4d,0xe2,0x04,0x20]
# CHECK-LE: bcctr 15, 2                     # encoding: [0x20,0x04,0xe2,0x4d]
            btctr+ 2
# CHECK-BE: btl+ 2, target                  # encoding: [0x41,0xe2,A,0bAAAAAA01]
# CHECK-LE: btl+ 2, target                  # encoding: [0bAAAAAA01,A,0xe2,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            btl+ 2, target
# CHECK-BE: btla+ 2, target                 # encoding: [0x41,0xe2,A,0bAAAAAA11]
# CHECK-LE: btla+ 2, target                 # encoding: [0bAAAAAA11,A,0xe2,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            btla+ 2, target
# CHECK-BE: bclrl 15, 2                     # encoding: [0x4d,0xe2,0x00,0x21]
# CHECK-LE: bclrl 15, 2                     # encoding: [0x21,0x00,0xe2,0x4d]
            btlrl+ 2
# CHECK-BE: bcctrl 15, 2                    # encoding: [0x4d,0xe2,0x04,0x21]
# CHECK-LE: bcctrl 15, 2                    # encoding: [0x21,0x04,0xe2,0x4d]
            btctrl+ 2

# CHECK-BE: bt- 2, target                   # encoding: [0x41,0xc2,A,0bAAAAAA00]
# CHECK-LE: bt- 2, target                   # encoding: [0bAAAAAA00,A,0xc2,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bt- 2, target
# CHECK-BE: bta- 2, target                  # encoding: [0x41,0xc2,A,0bAAAAAA10]
# CHECK-LE: bta- 2, target                  # encoding: [0bAAAAAA10,A,0xc2,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bta- 2, target
# CHECK-BE: bclr 14, 2                      # encoding: [0x4d,0xc2,0x00,0x20]
# CHECK-LE: bclr 14, 2                      # encoding: [0x20,0x00,0xc2,0x4d]
            btlr- 2
# CHECK-BE: bcctr 14, 2                     # encoding: [0x4d,0xc2,0x04,0x20]
# CHECK-LE: bcctr 14, 2                     # encoding: [0x20,0x04,0xc2,0x4d]
            btctr- 2
# CHECK-BE: btl- 2, target                  # encoding: [0x41,0xc2,A,0bAAAAAA01]
# CHECK-LE: btl- 2, target                  # encoding: [0bAAAAAA01,A,0xc2,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            btl- 2, target
# CHECK-BE: btla- 2, target                 # encoding: [0x41,0xc2,A,0bAAAAAA11]
# CHECK-LE: btla- 2, target                 # encoding: [0bAAAAAA11,A,0xc2,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            btla- 2, target
# CHECK-BE: bclrl 14, 2                     # encoding: [0x4d,0xc2,0x00,0x21]
# CHECK-LE: bclrl 14, 2                     # encoding: [0x21,0x00,0xc2,0x4d]
            btlrl- 2
# CHECK-BE: bcctrl 14, 2                    # encoding: [0x4d,0xc2,0x04,0x21]
# CHECK-LE: bcctrl 14, 2                    # encoding: [0x21,0x04,0xc2,0x4d]
            btctrl- 2

# CHECK-BE: bf 2, target                    # encoding: [0x40,0x82,A,0bAAAAAA00]
# CHECK-LE: bf 2, target                    # encoding: [0bAAAAAA00,A,0x82,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bf 2, target
# CHECK-BE: bfa 2, target                   # encoding: [0x40,0x82,A,0bAAAAAA10]
# CHECK-LE: bfa 2, target                   # encoding: [0bAAAAAA10,A,0x82,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bfa 2, target
# CHECK-BE: bclr 4, 2                       # encoding: [0x4c,0x82,0x00,0x20]
# CHECK-LE: bclr 4, 2                       # encoding: [0x20,0x00,0x82,0x4c]
            bflr 2
# CHECK-BE: bcctr 4, 2                      # encoding: [0x4c,0x82,0x04,0x20]
# CHECK-LE: bcctr 4, 2                      # encoding: [0x20,0x04,0x82,0x4c]
            bfctr 2
# CHECK-BE: bfl 2, target                   # encoding: [0x40,0x82,A,0bAAAAAA01]
# CHECK-LE: bfl 2, target                   # encoding: [0bAAAAAA01,A,0x82,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bfl 2, target
# CHECK-BE: bfla 2, target                  # encoding: [0x40,0x82,A,0bAAAAAA11]
# CHECK-LE: bfla 2, target                  # encoding: [0bAAAAAA11,A,0x82,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bfla 2, target
# CHECK-BE: bclrl 4, 2                      # encoding: [0x4c,0x82,0x00,0x21]
# CHECK-LE: bclrl 4, 2                      # encoding: [0x21,0x00,0x82,0x4c]
            bflrl 2
# CHECK-BE: bcctrl 4, 2                     # encoding: [0x4c,0x82,0x04,0x21]
# CHECK-LE: bcctrl 4, 2                     # encoding: [0x21,0x04,0x82,0x4c]
            bfctrl 2

# CHECK-BE: bf+ 2, target                   # encoding: [0x40,0xe2,A,0bAAAAAA00]
# CHECK-LE: bf+ 2, target                   # encoding: [0bAAAAAA00,A,0xe2,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bf+ 2, target
# CHECK-BE: bfa+ 2, target                  # encoding: [0x40,0xe2,A,0bAAAAAA10]
# CHECK-LE: bfa+ 2, target                  # encoding: [0bAAAAAA10,A,0xe2,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bfa+ 2, target
# CHECK-BE: bclr 7, 2                       # encoding: [0x4c,0xe2,0x00,0x20]
# CHECK-LE: bclr 7, 2                       # encoding: [0x20,0x00,0xe2,0x4c]
            bflr+ 2
# CHECK-BE: bcctr 7, 2                      # encoding: [0x4c,0xe2,0x04,0x20]
# CHECK-LE: bcctr 7, 2                      # encoding: [0x20,0x04,0xe2,0x4c]
            bfctr+ 2
# CHECK-BE: bfl+ 2, target                  # encoding: [0x40,0xe2,A,0bAAAAAA01]
# CHECK-LE: bfl+ 2, target                  # encoding: [0bAAAAAA01,A,0xe2,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bfl+ 2, target
# CHECK-BE: bfla+ 2, target                 # encoding: [0x40,0xe2,A,0bAAAAAA11]
# CHECK-LE: bfla+ 2, target                 # encoding: [0bAAAAAA11,A,0xe2,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bfla+ 2, target
# CHECK-BE: bclrl 7, 2                      # encoding: [0x4c,0xe2,0x00,0x21]
# CHECK-LE: bclrl 7, 2                      # encoding: [0x21,0x00,0xe2,0x4c]
            bflrl+ 2
# CHECK-BE: bcctrl 7, 2                     # encoding: [0x4c,0xe2,0x04,0x21]
# CHECK-LE: bcctrl 7, 2                     # encoding: [0x21,0x04,0xe2,0x4c]
            bfctrl+ 2

# CHECK-BE: bf- 2, target                   # encoding: [0x40,0xc2,A,0bAAAAAA00]
# CHECK-LE: bf- 2, target                   # encoding: [0bAAAAAA00,A,0xc2,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bf- 2, target
# CHECK-BE: bfa- 2, target                  # encoding: [0x40,0xc2,A,0bAAAAAA10]
# CHECK-LE: bfa- 2, target                  # encoding: [0bAAAAAA10,A,0xc2,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bfa- 2, target
# CHECK-BE: bclr 6, 2                       # encoding: [0x4c,0xc2,0x00,0x20]
# CHECK-LE: bclr 6, 2                       # encoding: [0x20,0x00,0xc2,0x4c]
            bflr- 2
# CHECK-BE: bcctr 6, 2                      # encoding: [0x4c,0xc2,0x04,0x20]
# CHECK-LE: bcctr 6, 2                      # encoding: [0x20,0x04,0xc2,0x4c]
            bfctr- 2
# CHECK-BE: bfl- 2, target                  # encoding: [0x40,0xc2,A,0bAAAAAA01]
# CHECK-LE: bfl- 2, target                  # encoding: [0bAAAAAA01,A,0xc2,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bfl- 2, target
# CHECK-BE: bfla- 2, target                 # encoding: [0x40,0xc2,A,0bAAAAAA11]
# CHECK-LE: bfla- 2, target                 # encoding: [0bAAAAAA11,A,0xc2,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bfla- 2, target
# CHECK-BE: bclrl 6, 2                      # encoding: [0x4c,0xc2,0x00,0x21]
# CHECK-LE: bclrl 6, 2                      # encoding: [0x21,0x00,0xc2,0x4c]
            bflrl- 2
# CHECK-BE: bcctrl 6, 2                     # encoding: [0x4c,0xc2,0x04,0x21]
# CHECK-LE: bcctrl 6, 2                     # encoding: [0x21,0x04,0xc2,0x4c]
            bfctrl- 2

# CHECK-BE: bdnz target                     # encoding: [0x42,0x00,A,0bAAAAAA00]
# CHECK-LE: bdnz target                     # encoding: [0bAAAAAA00,A,0x00,0x42]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bdnz target
# CHECK-BE: bdnza target                    # encoding: [0x42,0x00,A,0bAAAAAA10]
# CHECK-LE: bdnza target                    # encoding: [0bAAAAAA10,A,0x00,0x42]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bdnza target
# CHECK-BE: bdnzlr                          # encoding: [0x4e,0x00,0x00,0x20]
# CHECK-LE: bdnzlr                          # encoding: [0x20,0x00,0x00,0x4e]
            bdnzlr
# CHECK-BE: bdnzl target                    # encoding: [0x42,0x00,A,0bAAAAAA01]
# CHECK-LE: bdnzl target                    # encoding: [0bAAAAAA01,A,0x00,0x42]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bdnzl target
# CHECK-BE: bdnzla target                   # encoding: [0x42,0x00,A,0bAAAAAA11]
# CHECK-LE: bdnzla target                   # encoding: [0bAAAAAA11,A,0x00,0x42]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bdnzla target
# CHECK-BE: bdnzlrl                         # encoding: [0x4e,0x00,0x00,0x21]
# CHECK-LE: bdnzlrl                         # encoding: [0x21,0x00,0x00,0x4e]
            bdnzlrl

# CHECK-BE: bdnz+ target                    # encoding: [0x43,0x20,A,0bAAAAAA00]
# CHECK-LE: bdnz+ target                    # encoding: [0bAAAAAA00,A,0x20,0x43]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bdnz+ target
# CHECK-BE: bdnza+ target                   # encoding: [0x43,0x20,A,0bAAAAAA10]
# CHECK-LE: bdnza+ target                   # encoding: [0bAAAAAA10,A,0x20,0x43]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bdnza+ target
# CHECK-BE: bdnzlr+                         # encoding: [0x4f,0x20,0x00,0x20]
# CHECK-LE: bdnzlr+                         # encoding: [0x20,0x00,0x20,0x4f]
            bdnzlr+
# CHECK-BE: bdnzl+ target                   # encoding: [0x43,0x20,A,0bAAAAAA01]
# CHECK-LE: bdnzl+ target                   # encoding: [0bAAAAAA01,A,0x20,0x43]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bdnzl+ target
# CHECK-BE: bdnzla+ target                  # encoding: [0x43,0x20,A,0bAAAAAA11]
# CHECK-LE: bdnzla+ target                  # encoding: [0bAAAAAA11,A,0x20,0x43]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bdnzla+ target
# CHECK-BE: bdnzlrl+                        # encoding: [0x4f,0x20,0x00,0x21]
# CHECK-LE: bdnzlrl+                        # encoding: [0x21,0x00,0x20,0x4f]
            bdnzlrl+

# CHECK-BE: bdnz- target                    # encoding: [0x43,0x00,A,0bAAAAAA00]
# CHECK-LE: bdnz- target                    # encoding: [0bAAAAAA00,A,0x00,0x43]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bdnz- target
# CHECK-BE: bdnza- target                   # encoding: [0x43,0x00,A,0bAAAAAA10]
# CHECK-LE: bdnza- target                   # encoding: [0bAAAAAA10,A,0x00,0x43]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bdnza- target
# CHECK-BE: bdnzlr-                         # encoding: [0x4f,0x00,0x00,0x20]
# CHECK-LE: bdnzlr-                         # encoding: [0x20,0x00,0x00,0x4f]
            bdnzlr-
# CHECK-BE: bdnzl- target                   # encoding: [0x43,0x00,A,0bAAAAAA01]
# CHECK-LE: bdnzl- target                   # encoding: [0bAAAAAA01,A,0x00,0x43]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bdnzl- target
# CHECK-BE: bdnzla- target                  # encoding: [0x43,0x00,A,0bAAAAAA11]
# CHECK-LE: bdnzla- target                  # encoding: [0bAAAAAA11,A,0x00,0x43]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bdnzla- target
# CHECK-BE: bdnzlrl-                        # encoding: [0x4f,0x00,0x00,0x21]
# CHECK-LE: bdnzlrl-                        # encoding: [0x21,0x00,0x00,0x4f]
            bdnzlrl-

# CHECK-BE: bdnzt 2, target                 # encoding: [0x41,0x02,A,0bAAAAAA00]
# CHECK-LE: bdnzt 2, target                 # encoding: [0bAAAAAA00,A,0x02,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bdnzt 2, target
# CHECK-BE: bdnzta 2, target                # encoding: [0x41,0x02,A,0bAAAAAA10]
# CHECK-LE: bdnzta 2, target                # encoding: [0bAAAAAA10,A,0x02,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bdnzta 2, target
# CHECK-BE: bclr 8, 2                       # encoding: [0x4d,0x02,0x00,0x20]
# CHECK-LE: bclr 8, 2                       # encoding: [0x20,0x00,0x02,0x4d]
            bdnztlr 2
# CHECK-BE: bdnztl 2, target                # encoding: [0x41,0x02,A,0bAAAAAA01]
# CHECK-LE: bdnztl 2, target                # encoding: [0bAAAAAA01,A,0x02,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bdnztl 2, target
# CHECK-BE: bdnztla 2, target               # encoding: [0x41,0x02,A,0bAAAAAA11]
# CHECK-LE: bdnztla 2, target               # encoding: [0bAAAAAA11,A,0x02,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bdnztla 2, target
# CHECK-BE: bclrl 8, 2                      # encoding: [0x4d,0x02,0x00,0x21]
# CHECK-LE: bclrl 8, 2                      # encoding: [0x21,0x00,0x02,0x4d]
            bdnztlrl 2

# CHECK-BE: bdnzf 2, target                 # encoding: [0x40,0x02,A,0bAAAAAA00]
# CHECK-LE: bdnzf 2, target                 # encoding: [0bAAAAAA00,A,0x02,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bdnzf 2, target
# CHECK-BE: bdnzfa 2, target                # encoding: [0x40,0x02,A,0bAAAAAA10]
# CHECK-LE: bdnzfa 2, target                # encoding: [0bAAAAAA10,A,0x02,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bdnzfa 2, target
# CHECK-BE: bclr 0, 2                       # encoding: [0x4c,0x02,0x00,0x20]
# CHECK-LE: bclr 0, 2                       # encoding: [0x20,0x00,0x02,0x4c]
            bdnzflr 2
# CHECK-BE: bdnzfl 2, target                # encoding: [0x40,0x02,A,0bAAAAAA01]
# CHECK-LE: bdnzfl 2, target                # encoding: [0bAAAAAA01,A,0x02,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bdnzfl 2, target
# CHECK-BE: bdnzfla 2, target               # encoding: [0x40,0x02,A,0bAAAAAA11]
# CHECK-LE: bdnzfla 2, target               # encoding: [0bAAAAAA11,A,0x02,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bdnzfla 2, target
# CHECK-BE: bclrl 0, 2                      # encoding: [0x4c,0x02,0x00,0x21]
# CHECK-LE: bclrl 0, 2                      # encoding: [0x21,0x00,0x02,0x4c]
            bdnzflrl 2

# CHECK-BE: bdz target                      # encoding: [0x42,0x40,A,0bAAAAAA00]
# CHECK-LE: bdz target                      # encoding: [0bAAAAAA00,A,0x40,0x42]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bdz target
# CHECK-BE: bdza target                     # encoding: [0x42,0x40,A,0bAAAAAA10]
# CHECK-LE: bdza target                     # encoding: [0bAAAAAA10,A,0x40,0x42]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bdza target
# CHECK-BE: bdzlr                           # encoding: [0x4e,0x40,0x00,0x20]
# CHECK-LE: bdzlr                           # encoding: [0x20,0x00,0x40,0x4e]
            bdzlr
# CHECK-BE: bdzl target                     # encoding: [0x42,0x40,A,0bAAAAAA01]
# CHECK-LE: bdzl target                     # encoding: [0bAAAAAA01,A,0x40,0x42]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bdzl target
# CHECK-BE: bdzla target                    # encoding: [0x42,0x40,A,0bAAAAAA11]
# CHECK-LE: bdzla target                    # encoding: [0bAAAAAA11,A,0x40,0x42]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bdzla target
# CHECK-BE: bdzlrl                          # encoding: [0x4e,0x40,0x00,0x21]
# CHECK-LE: bdzlrl                          # encoding: [0x21,0x00,0x40,0x4e]
            bdzlrl

# CHECK-BE: bdz+ target                     # encoding: [0x43,0x60,A,0bAAAAAA00]
# CHECK-LE: bdz+ target                     # encoding: [0bAAAAAA00,A,0x60,0x43]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bdz+ target
# CHECK-BE: bdza+ target                    # encoding: [0x43,0x60,A,0bAAAAAA10]
# CHECK-LE: bdza+ target                    # encoding: [0bAAAAAA10,A,0x60,0x43]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bdza+ target
# CHECK-BE: bdzlr+                          # encoding: [0x4f,0x60,0x00,0x20]
# CHECK-LE: bdzlr+                          # encoding: [0x20,0x00,0x60,0x4f]
            bdzlr+
# CHECK-BE: bdzl+ target                    # encoding: [0x43,0x60,A,0bAAAAAA01]
# CHECK-LE: bdzl+ target                    # encoding: [0bAAAAAA01,A,0x60,0x43]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bdzl+ target
# CHECK-BE: bdzla+ target                   # encoding: [0x43,0x60,A,0bAAAAAA11]
# CHECK-LE: bdzla+ target                   # encoding: [0bAAAAAA11,A,0x60,0x43]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bdzla+ target
# CHECK-BE: bdzlrl+                         # encoding: [0x4f,0x60,0x00,0x21]
# CHECK-LE: bdzlrl+                         # encoding: [0x21,0x00,0x60,0x4f]
            bdzlrl+

# CHECK-BE: bdz- target                     # encoding: [0x43,0x40,A,0bAAAAAA00]
# CHECK-LE: bdz- target                     # encoding: [0bAAAAAA00,A,0x40,0x43]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bdz- target
# CHECK-BE: bdza- target                    # encoding: [0x43,0x40,A,0bAAAAAA10]
# CHECK-LE: bdza- target                    # encoding: [0bAAAAAA10,A,0x40,0x43]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bdza- target
# CHECK-BE: bdzlr-                          # encoding: [0x4f,0x40,0x00,0x20]
# CHECK-LE: bdzlr-                          # encoding: [0x20,0x00,0x40,0x4f]
            bdzlr-
# CHECK-BE: bdzl- target                    # encoding: [0x43,0x40,A,0bAAAAAA01]
# CHECK-LE: bdzl- target                    # encoding: [0bAAAAAA01,A,0x40,0x43]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bdzl- target
# CHECK-BE: bdzla- target                   # encoding: [0x43,0x40,A,0bAAAAAA11]
# CHECK-LE: bdzla- target                   # encoding: [0bAAAAAA11,A,0x40,0x43]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bdzla- target
# CHECK-BE: bdzlrl-                         # encoding: [0x4f,0x40,0x00,0x21]
# CHECK-LE: bdzlrl-                         # encoding: [0x21,0x00,0x40,0x4f]
            bdzlrl-

# CHECK-BE: bdzt 2, target                  # encoding: [0x41,0x42,A,0bAAAAAA00]
# CHECK-LE: bdzt 2, target                  # encoding: [0bAAAAAA00,A,0x42,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bdzt 2, target
# CHECK-BE: bdzta 2, target                 # encoding: [0x41,0x42,A,0bAAAAAA10]
# CHECK-LE: bdzta 2, target                 # encoding: [0bAAAAAA10,A,0x42,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bdzta 2, target
# CHECK-BE: bclr 10, 2                      # encoding: [0x4d,0x42,0x00,0x20]
# CHECK-LE: bclr 10, 2                      # encoding: [0x20,0x00,0x42,0x4d]
            bdztlr 2
# CHECK-BE: bdztl 2, target                 # encoding: [0x41,0x42,A,0bAAAAAA01]
# CHECK-LE: bdztl 2, target                 # encoding: [0bAAAAAA01,A,0x42,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bdztl 2, target
# CHECK-BE: bdztla 2, target                # encoding: [0x41,0x42,A,0bAAAAAA11]
# CHECK-LE: bdztla 2, target                # encoding: [0bAAAAAA11,A,0x42,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bdztla 2, target
# CHECK-BE: bclrl 10, 2                     # encoding: [0x4d,0x42,0x00,0x21]
# CHECK-LE: bclrl 10, 2                     # encoding: [0x21,0x00,0x42,0x4d]
            bdztlrl 2

# CHECK-BE: bdzf 2, target                  # encoding: [0x40,0x42,A,0bAAAAAA00]
# CHECK-LE: bdzf 2, target                  # encoding: [0bAAAAAA00,A,0x42,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bdzf 2, target
# CHECK-BE: bdzfa 2, target                 # encoding: [0x40,0x42,A,0bAAAAAA10]
# CHECK-LE: bdzfa 2, target                 # encoding: [0bAAAAAA10,A,0x42,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bdzfa 2, target
# CHECK-BE: bclr 2, 2                       # encoding: [0x4c,0x42,0x00,0x20]
# CHECK-LE: bclr 2, 2                       # encoding: [0x20,0x00,0x42,0x4c]
            bdzflr 2
# CHECK-BE: bdzfl 2, target                 # encoding: [0x40,0x42,A,0bAAAAAA01]
# CHECK-LE: bdzfl 2, target                 # encoding: [0bAAAAAA01,A,0x42,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bdzfl 2, target
# CHECK-BE: bdzfla 2, target                # encoding: [0x40,0x42,A,0bAAAAAA11]
# CHECK-LE: bdzfla 2, target                # encoding: [0bAAAAAA11,A,0x42,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bdzfla 2, target
# CHECK-BE: bclrl 2, 2                      # encoding: [0x4c,0x42,0x00,0x21]
# CHECK-LE: bclrl 2, 2                      # encoding: [0x21,0x00,0x42,0x4c]
            bdzflrl 2

# CHECK-BE: blt 2, target                   # encoding: [0x41,0x88,A,0bAAAAAA00]
# CHECK-LE: blt 2, target                   # encoding: [0bAAAAAA00,A,0x88,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            blt 2, target
# CHECK-BE: blt 0, target                   # encoding: [0x41,0x80,A,0bAAAAAA00]
# CHECK-LE: blt 0, target                   # encoding: [0bAAAAAA00,A,0x80,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            blt target
# CHECK-BE: blta 2, target                  # encoding: [0x41,0x88,A,0bAAAAAA10]
# CHECK-LE: blta 2, target                  # encoding: [0bAAAAAA10,A,0x88,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            blta 2, target
# CHECK-BE: blta 0, target                  # encoding: [0x41,0x80,A,0bAAAAAA10]
# CHECK-LE: blta 0, target                  # encoding: [0bAAAAAA10,A,0x80,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            blta target
# CHECK-BE: bltlr 2                         # encoding: [0x4d,0x88,0x00,0x20]
# CHECK-LE: bltlr 2                         # encoding: [0x20,0x00,0x88,0x4d]
            bltlr 2
# CHECK-BE: bltlr 0                         # encoding: [0x4d,0x80,0x00,0x20]
# CHECK-LE: bltlr 0                         # encoding: [0x20,0x00,0x80,0x4d]
            bltlr
# CHECK-BE: bltctr 2                        # encoding: [0x4d,0x88,0x04,0x20]
# CHECK-LE: bltctr 2                        # encoding: [0x20,0x04,0x88,0x4d]
            bltctr 2
# CHECK-BE: bltctr 0                        # encoding: [0x4d,0x80,0x04,0x20]
# CHECK-LE: bltctr 0                        # encoding: [0x20,0x04,0x80,0x4d]
            bltctr
# CHECK-BE: bltl 2, target                  # encoding: [0x41,0x88,A,0bAAAAAA01]
# CHECK-LE: bltl 2, target                  # encoding: [0bAAAAAA01,A,0x88,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bltl 2, target
# CHECK-BE: bltl 0, target                  # encoding: [0x41,0x80,A,0bAAAAAA01]
# CHECK-LE: bltl 0, target                  # encoding: [0bAAAAAA01,A,0x80,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bltl target
# CHECK-BE: bltla 2, target                 # encoding: [0x41,0x88,A,0bAAAAAA11]
# CHECK-LE: bltla 2, target                 # encoding: [0bAAAAAA11,A,0x88,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bltla 2, target
# CHECK-BE: bltla 0, target                 # encoding: [0x41,0x80,A,0bAAAAAA11]
# CHECK-LE: bltla 0, target                 # encoding: [0bAAAAAA11,A,0x80,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bltla target
# CHECK-BE: bltlrl 2                        # encoding: [0x4d,0x88,0x00,0x21]
# CHECK-LE: bltlrl 2                        # encoding: [0x21,0x00,0x88,0x4d]
            bltlrl 2
# CHECK-BE: bltlrl 0                        # encoding: [0x4d,0x80,0x00,0x21]
# CHECK-LE: bltlrl 0                        # encoding: [0x21,0x00,0x80,0x4d]
            bltlrl
# CHECK-BE: bltctrl 2                       # encoding: [0x4d,0x88,0x04,0x21]
# CHECK-LE: bltctrl 2                       # encoding: [0x21,0x04,0x88,0x4d]
            bltctrl 2
# CHECK-BE: bltctrl 0                       # encoding: [0x4d,0x80,0x04,0x21]
# CHECK-LE: bltctrl 0                       # encoding: [0x21,0x04,0x80,0x4d]
            bltctrl

# CHECK-BE: blt+ 2, target                  # encoding: [0x41,0xe8,A,0bAAAAAA00]
# CHECK-LE: blt+ 2, target                  # encoding: [0bAAAAAA00,A,0xe8,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            blt+ 2, target
# CHECK-BE: blt+ 0, target                  # encoding: [0x41,0xe0,A,0bAAAAAA00]
# CHECK-LE: blt+ 0, target                  # encoding: [0bAAAAAA00,A,0xe0,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            blt+ target
# CHECK-BE: blta+ 2, target                 # encoding: [0x41,0xe8,A,0bAAAAAA10]
# CHECK-LE: blta+ 2, target                 # encoding: [0bAAAAAA10,A,0xe8,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            blta+ 2, target
# CHECK-BE: blta+ 0, target                 # encoding: [0x41,0xe0,A,0bAAAAAA10]
# CHECK-LE: blta+ 0, target                 # encoding: [0bAAAAAA10,A,0xe0,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            blta+ target
# CHECK-BE: bltlr+ 2                        # encoding: [0x4d,0xe8,0x00,0x20]
# CHECK-LE: bltlr+ 2                        # encoding: [0x20,0x00,0xe8,0x4d]
            bltlr+ 2
# CHECK-BE: bltlr+ 0                        # encoding: [0x4d,0xe0,0x00,0x20]
# CHECK-LE: bltlr+ 0                        # encoding: [0x20,0x00,0xe0,0x4d]
            bltlr+
# CHECK-BE: bltctr+ 2                       # encoding: [0x4d,0xe8,0x04,0x20]
# CHECK-LE: bltctr+ 2                       # encoding: [0x20,0x04,0xe8,0x4d]
            bltctr+ 2
# CHECK-BE: bltctr+ 0                       # encoding: [0x4d,0xe0,0x04,0x20]
# CHECK-LE: bltctr+ 0                       # encoding: [0x20,0x04,0xe0,0x4d]
            bltctr+
# CHECK-BE: bltl+ 2, target                 # encoding: [0x41,0xe8,A,0bAAAAAA01]
# CHECK-LE: bltl+ 2, target                 # encoding: [0bAAAAAA01,A,0xe8,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bltl+ 2, target
# CHECK-BE: bltl+ 0, target                 # encoding: [0x41,0xe0,A,0bAAAAAA01]
# CHECK-LE: bltl+ 0, target                 # encoding: [0bAAAAAA01,A,0xe0,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bltl+ target
# CHECK-BE: bltla+ 2, target                # encoding: [0x41,0xe8,A,0bAAAAAA11]
# CHECK-LE: bltla+ 2, target                # encoding: [0bAAAAAA11,A,0xe8,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bltla+ 2, target
# CHECK-BE: bltla+ 0, target                # encoding: [0x41,0xe0,A,0bAAAAAA11]
# CHECK-LE: bltla+ 0, target                # encoding: [0bAAAAAA11,A,0xe0,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bltla+ target
# CHECK-BE: bltlrl+ 2                       # encoding: [0x4d,0xe8,0x00,0x21]
# CHECK-LE: bltlrl+ 2                       # encoding: [0x21,0x00,0xe8,0x4d]
            bltlrl+ 2
# CHECK-BE: bltlrl+ 0                       # encoding: [0x4d,0xe0,0x00,0x21]
# CHECK-LE: bltlrl+ 0                       # encoding: [0x21,0x00,0xe0,0x4d]
            bltlrl+
# CHECK-BE: bltctrl+ 2                      # encoding: [0x4d,0xe8,0x04,0x21]
# CHECK-LE: bltctrl+ 2                      # encoding: [0x21,0x04,0xe8,0x4d]
            bltctrl+ 2
# CHECK-BE: bltctrl+ 0                      # encoding: [0x4d,0xe0,0x04,0x21]
# CHECK-LE: bltctrl+ 0                      # encoding: [0x21,0x04,0xe0,0x4d]
            bltctrl+

# CHECK-BE: blt- 2, target                  # encoding: [0x41,0xc8,A,0bAAAAAA00]
# CHECK-LE: blt- 2, target                  # encoding: [0bAAAAAA00,A,0xc8,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            blt- 2, target
# CHECK-BE: blt- 0, target                  # encoding: [0x41,0xc0,A,0bAAAAAA00]
# CHECK-LE: blt- 0, target                  # encoding: [0bAAAAAA00,A,0xc0,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            blt- target
# CHECK-BE: blta- 2, target                 # encoding: [0x41,0xc8,A,0bAAAAAA10]
# CHECK-LE: blta- 2, target                 # encoding: [0bAAAAAA10,A,0xc8,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            blta- 2, target
# CHECK-BE: blta- 0, target                 # encoding: [0x41,0xc0,A,0bAAAAAA10]
# CHECK-LE: blta- 0, target                 # encoding: [0bAAAAAA10,A,0xc0,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            blta- target
# CHECK-BE: bltlr- 2                        # encoding: [0x4d,0xc8,0x00,0x20]
# CHECK-LE: bltlr- 2                        # encoding: [0x20,0x00,0xc8,0x4d]
            bltlr- 2
# CHECK-BE: bltlr- 0                        # encoding: [0x4d,0xc0,0x00,0x20]
# CHECK-LE: bltlr- 0                        # encoding: [0x20,0x00,0xc0,0x4d]
            bltlr-
# CHECK-BE: bltctr- 2                       # encoding: [0x4d,0xc8,0x04,0x20]
# CHECK-LE: bltctr- 2                       # encoding: [0x20,0x04,0xc8,0x4d]
            bltctr- 2
# CHECK-BE: bltctr- 0                       # encoding: [0x4d,0xc0,0x04,0x20]
# CHECK-LE: bltctr- 0                       # encoding: [0x20,0x04,0xc0,0x4d]
            bltctr-
# CHECK-BE: bltl- 2, target                 # encoding: [0x41,0xc8,A,0bAAAAAA01]
# CHECK-LE: bltl- 2, target                 # encoding: [0bAAAAAA01,A,0xc8,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bltl- 2, target
# CHECK-BE: bltl- 0, target                 # encoding: [0x41,0xc0,A,0bAAAAAA01]
# CHECK-LE: bltl- 0, target                 # encoding: [0bAAAAAA01,A,0xc0,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bltl- target
# CHECK-BE: bltla- 2, target                # encoding: [0x41,0xc8,A,0bAAAAAA11]
# CHECK-LE: bltla- 2, target                # encoding: [0bAAAAAA11,A,0xc8,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bltla- 2, target
# CHECK-BE: bltla- 0, target                # encoding: [0x41,0xc0,A,0bAAAAAA11]
# CHECK-LE: bltla- 0, target                # encoding: [0bAAAAAA11,A,0xc0,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bltla- target
# CHECK-BE: bltlrl- 2                       # encoding: [0x4d,0xc8,0x00,0x21]
# CHECK-LE: bltlrl- 2                       # encoding: [0x21,0x00,0xc8,0x4d]
            bltlrl- 2
# CHECK-BE: bltlrl- 0                       # encoding: [0x4d,0xc0,0x00,0x21]
# CHECK-LE: bltlrl- 0                       # encoding: [0x21,0x00,0xc0,0x4d]
            bltlrl-
# CHECK-BE: bltctrl- 2                      # encoding: [0x4d,0xc8,0x04,0x21]
# CHECK-LE: bltctrl- 2                      # encoding: [0x21,0x04,0xc8,0x4d]
            bltctrl- 2
# CHECK-BE: bltctrl- 0                      # encoding: [0x4d,0xc0,0x04,0x21]
# CHECK-LE: bltctrl- 0                      # encoding: [0x21,0x04,0xc0,0x4d]
            bltctrl-

# CHECK-BE: ble 2, target                   # encoding: [0x40,0x89,A,0bAAAAAA00]
# CHECK-LE: ble 2, target                   # encoding: [0bAAAAAA00,A,0x89,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            ble 2, target
# CHECK-BE: ble 0, target                   # encoding: [0x40,0x81,A,0bAAAAAA00]
# CHECK-LE: ble 0, target                   # encoding: [0bAAAAAA00,A,0x81,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            ble target
# CHECK-BE: blea 2, target                  # encoding: [0x40,0x89,A,0bAAAAAA10]
# CHECK-LE: blea 2, target                  # encoding: [0bAAAAAA10,A,0x89,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            blea 2, target
# CHECK-BE: blea 0, target                  # encoding: [0x40,0x81,A,0bAAAAAA10]
# CHECK-LE: blea 0, target                  # encoding: [0bAAAAAA10,A,0x81,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            blea target
# CHECK-BE: blelr 2                         # encoding: [0x4c,0x89,0x00,0x20]
# CHECK-LE: blelr 2                         # encoding: [0x20,0x00,0x89,0x4c]
            blelr 2
# CHECK-BE: blelr 0                         # encoding: [0x4c,0x81,0x00,0x20]
# CHECK-LE: blelr 0                         # encoding: [0x20,0x00,0x81,0x4c]
            blelr
# CHECK-BE: blectr 2                        # encoding: [0x4c,0x89,0x04,0x20]
# CHECK-LE: blectr 2                        # encoding: [0x20,0x04,0x89,0x4c]
            blectr 2
# CHECK-BE: blectr 0                        # encoding: [0x4c,0x81,0x04,0x20]
# CHECK-LE: blectr 0                        # encoding: [0x20,0x04,0x81,0x4c]
            blectr
# CHECK-BE: blel 2, target                  # encoding: [0x40,0x89,A,0bAAAAAA01]
# CHECK-LE: blel 2, target                  # encoding: [0bAAAAAA01,A,0x89,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            blel 2, target
# CHECK-BE: blel 0, target                  # encoding: [0x40,0x81,A,0bAAAAAA01]
# CHECK-LE: blel 0, target                  # encoding: [0bAAAAAA01,A,0x81,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            blel target
# CHECK-BE: blela 2, target                 # encoding: [0x40,0x89,A,0bAAAAAA11]
# CHECK-LE: blela 2, target                 # encoding: [0bAAAAAA11,A,0x89,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            blela 2, target
# CHECK-BE: blela 0, target                 # encoding: [0x40,0x81,A,0bAAAAAA11]
# CHECK-LE: blela 0, target                 # encoding: [0bAAAAAA11,A,0x81,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            blela target
# CHECK-BE: blelrl 2                        # encoding: [0x4c,0x89,0x00,0x21]
# CHECK-LE: blelrl 2                        # encoding: [0x21,0x00,0x89,0x4c]
            blelrl 2
# CHECK-BE: blelrl 0                        # encoding: [0x4c,0x81,0x00,0x21]
# CHECK-LE: blelrl 0                        # encoding: [0x21,0x00,0x81,0x4c]
            blelrl
# CHECK-BE: blectrl 2                       # encoding: [0x4c,0x89,0x04,0x21]
# CHECK-LE: blectrl 2                       # encoding: [0x21,0x04,0x89,0x4c]
            blectrl 2
# CHECK-BE: blectrl 0                       # encoding: [0x4c,0x81,0x04,0x21]
# CHECK-LE: blectrl 0                       # encoding: [0x21,0x04,0x81,0x4c]
            blectrl

# CHECK-BE: ble+ 2, target                  # encoding: [0x40,0xe9,A,0bAAAAAA00]
# CHECK-LE: ble+ 2, target                  # encoding: [0bAAAAAA00,A,0xe9,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            ble+ 2, target
# CHECK-BE: ble+ 0, target                  # encoding: [0x40,0xe1,A,0bAAAAAA00]
# CHECK-LE: ble+ 0, target                  # encoding: [0bAAAAAA00,A,0xe1,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            ble+ target
# CHECK-BE: blea+ 2, target                 # encoding: [0x40,0xe9,A,0bAAAAAA10]
# CHECK-LE: blea+ 2, target                 # encoding: [0bAAAAAA10,A,0xe9,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            blea+ 2, target
# CHECK-BE: blea+ 0, target                 # encoding: [0x40,0xe1,A,0bAAAAAA10]
# CHECK-LE: blea+ 0, target                 # encoding: [0bAAAAAA10,A,0xe1,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            blea+ target
# CHECK-BE: blelr+ 2                        # encoding: [0x4c,0xe9,0x00,0x20]
# CHECK-LE: blelr+ 2                        # encoding: [0x20,0x00,0xe9,0x4c]
            blelr+ 2
# CHECK-BE: blelr+ 0                        # encoding: [0x4c,0xe1,0x00,0x20]
# CHECK-LE: blelr+ 0                        # encoding: [0x20,0x00,0xe1,0x4c]
            blelr+
# CHECK-BE: blectr+ 2                       # encoding: [0x4c,0xe9,0x04,0x20]
# CHECK-LE: blectr+ 2                       # encoding: [0x20,0x04,0xe9,0x4c]
            blectr+ 2
# CHECK-BE: blectr+ 0                       # encoding: [0x4c,0xe1,0x04,0x20]
# CHECK-LE: blectr+ 0                       # encoding: [0x20,0x04,0xe1,0x4c]
            blectr+
# CHECK-BE: blel+ 2, target                 # encoding: [0x40,0xe9,A,0bAAAAAA01]
# CHECK-LE: blel+ 2, target                 # encoding: [0bAAAAAA01,A,0xe9,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            blel+ 2, target
# CHECK-BE: blel+ 0, target                 # encoding: [0x40,0xe1,A,0bAAAAAA01]
# CHECK-LE: blel+ 0, target                 # encoding: [0bAAAAAA01,A,0xe1,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            blel+ target
# CHECK-BE: blela+ 2, target                # encoding: [0x40,0xe9,A,0bAAAAAA11]
# CHECK-LE: blela+ 2, target                # encoding: [0bAAAAAA11,A,0xe9,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            blela+ 2, target
# CHECK-BE: blela+ 0, target                # encoding: [0x40,0xe1,A,0bAAAAAA11]
# CHECK-LE: blela+ 0, target                # encoding: [0bAAAAAA11,A,0xe1,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            blela+ target
# CHECK-BE: blelrl+ 2                       # encoding: [0x4c,0xe9,0x00,0x21]
# CHECK-LE: blelrl+ 2                       # encoding: [0x21,0x00,0xe9,0x4c]
            blelrl+ 2
# CHECK-BE: blelrl+ 0                       # encoding: [0x4c,0xe1,0x00,0x21]
# CHECK-LE: blelrl+ 0                       # encoding: [0x21,0x00,0xe1,0x4c]
            blelrl+
# CHECK-BE: blectrl+ 2                      # encoding: [0x4c,0xe9,0x04,0x21]
# CHECK-LE: blectrl+ 2                      # encoding: [0x21,0x04,0xe9,0x4c]
            blectrl+ 2
# CHECK-BE: blectrl+ 0                      # encoding: [0x4c,0xe1,0x04,0x21]
# CHECK-LE: blectrl+ 0                      # encoding: [0x21,0x04,0xe1,0x4c]
            blectrl+

# CHECK-BE: ble- 2, target                  # encoding: [0x40,0xc9,A,0bAAAAAA00]
# CHECK-LE: ble- 2, target                  # encoding: [0bAAAAAA00,A,0xc9,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            ble- 2, target
# CHECK-BE: ble- 0, target                  # encoding: [0x40,0xc1,A,0bAAAAAA00]
# CHECK-LE: ble- 0, target                  # encoding: [0bAAAAAA00,A,0xc1,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            ble- target
# CHECK-BE: blea- 2, target                 # encoding: [0x40,0xc9,A,0bAAAAAA10]
# CHECK-LE: blea- 2, target                 # encoding: [0bAAAAAA10,A,0xc9,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            blea- 2, target
# CHECK-BE: blea- 0, target                 # encoding: [0x40,0xc1,A,0bAAAAAA10]
# CHECK-LE: blea- 0, target                 # encoding: [0bAAAAAA10,A,0xc1,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            blea- target
# CHECK-BE: blelr- 2                        # encoding: [0x4c,0xc9,0x00,0x20]
# CHECK-LE: blelr- 2                        # encoding: [0x20,0x00,0xc9,0x4c]
            blelr- 2
# CHECK-BE: blelr- 0                        # encoding: [0x4c,0xc1,0x00,0x20]
# CHECK-LE: blelr- 0                        # encoding: [0x20,0x00,0xc1,0x4c]
            blelr-
# CHECK-BE: blectr- 2                       # encoding: [0x4c,0xc9,0x04,0x20]
# CHECK-LE: blectr- 2                       # encoding: [0x20,0x04,0xc9,0x4c]
            blectr- 2
# CHECK-BE: blectr- 0                       # encoding: [0x4c,0xc1,0x04,0x20]
# CHECK-LE: blectr- 0                       # encoding: [0x20,0x04,0xc1,0x4c]
            blectr-
# CHECK-BE: blel- 2, target                 # encoding: [0x40,0xc9,A,0bAAAAAA01]
# CHECK-LE: blel- 2, target                 # encoding: [0bAAAAAA01,A,0xc9,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            blel- 2, target
# CHECK-BE: blel- 0, target                 # encoding: [0x40,0xc1,A,0bAAAAAA01]
# CHECK-LE: blel- 0, target                 # encoding: [0bAAAAAA01,A,0xc1,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            blel- target
# CHECK-BE: blela- 2, target                # encoding: [0x40,0xc9,A,0bAAAAAA11]
# CHECK-LE: blela- 2, target                # encoding: [0bAAAAAA11,A,0xc9,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            blela- 2, target
# CHECK-BE: blela- 0, target                # encoding: [0x40,0xc1,A,0bAAAAAA11]
# CHECK-LE: blela- 0, target                # encoding: [0bAAAAAA11,A,0xc1,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            blela- target
# CHECK-BE: blelrl- 2                       # encoding: [0x4c,0xc9,0x00,0x21]
# CHECK-LE: blelrl- 2                       # encoding: [0x21,0x00,0xc9,0x4c]
            blelrl- 2
# CHECK-BE: blelrl- 0                       # encoding: [0x4c,0xc1,0x00,0x21]
# CHECK-LE: blelrl- 0                       # encoding: [0x21,0x00,0xc1,0x4c]
            blelrl-
# CHECK-BE: blectrl- 2                      # encoding: [0x4c,0xc9,0x04,0x21]
# CHECK-LE: blectrl- 2                      # encoding: [0x21,0x04,0xc9,0x4c]
            blectrl- 2
# CHECK-BE: blectrl- 0                      # encoding: [0x4c,0xc1,0x04,0x21]
# CHECK-LE: blectrl- 0                      # encoding: [0x21,0x04,0xc1,0x4c]
            blectrl-

# CHECK-BE: beq 2, target                   # encoding: [0x41,0x8a,A,0bAAAAAA00]
# CHECK-LE: beq 2, target                   # encoding: [0bAAAAAA00,A,0x8a,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            beq 2, target
# CHECK-BE: beq 0, target                   # encoding: [0x41,0x82,A,0bAAAAAA00]
# CHECK-LE: beq 0, target                   # encoding: [0bAAAAAA00,A,0x82,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            beq target
# CHECK-BE: beqa 2, target                  # encoding: [0x41,0x8a,A,0bAAAAAA10]
# CHECK-LE: beqa 2, target                  # encoding: [0bAAAAAA10,A,0x8a,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            beqa 2, target
# CHECK-BE: beqa 0, target                  # encoding: [0x41,0x82,A,0bAAAAAA10]
# CHECK-LE: beqa 0, target                  # encoding: [0bAAAAAA10,A,0x82,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            beqa target
# CHECK-BE: beqlr 2                         # encoding: [0x4d,0x8a,0x00,0x20]
# CHECK-LE: beqlr 2                         # encoding: [0x20,0x00,0x8a,0x4d]
            beqlr 2
# CHECK-BE: beqlr 0                         # encoding: [0x4d,0x82,0x00,0x20]
# CHECK-LE: beqlr 0                         # encoding: [0x20,0x00,0x82,0x4d]
            beqlr
# CHECK-BE: beqctr 2                        # encoding: [0x4d,0x8a,0x04,0x20]
# CHECK-LE: beqctr 2                        # encoding: [0x20,0x04,0x8a,0x4d]
            beqctr 2
# CHECK-BE: beqctr 0                        # encoding: [0x4d,0x82,0x04,0x20]
# CHECK-LE: beqctr 0                        # encoding: [0x20,0x04,0x82,0x4d]
            beqctr
# CHECK-BE: beql 2, target                  # encoding: [0x41,0x8a,A,0bAAAAAA01]
# CHECK-LE: beql 2, target                  # encoding: [0bAAAAAA01,A,0x8a,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            beql 2, target
# CHECK-BE: beql 0, target                  # encoding: [0x41,0x82,A,0bAAAAAA01]
# CHECK-LE: beql 0, target                  # encoding: [0bAAAAAA01,A,0x82,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            beql target
# CHECK-BE: beqla 2, target                 # encoding: [0x41,0x8a,A,0bAAAAAA11]
# CHECK-LE: beqla 2, target                 # encoding: [0bAAAAAA11,A,0x8a,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            beqla 2, target
# CHECK-BE: beqla 0, target                 # encoding: [0x41,0x82,A,0bAAAAAA11]
# CHECK-LE: beqla 0, target                 # encoding: [0bAAAAAA11,A,0x82,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            beqla target
# CHECK-BE: beqlrl 2                        # encoding: [0x4d,0x8a,0x00,0x21]
# CHECK-LE: beqlrl 2                        # encoding: [0x21,0x00,0x8a,0x4d]
            beqlrl 2
# CHECK-BE: beqlrl 0                        # encoding: [0x4d,0x82,0x00,0x21]
# CHECK-LE: beqlrl 0                        # encoding: [0x21,0x00,0x82,0x4d]
            beqlrl
# CHECK-BE: beqctrl 2                       # encoding: [0x4d,0x8a,0x04,0x21]
# CHECK-LE: beqctrl 2                       # encoding: [0x21,0x04,0x8a,0x4d]
            beqctrl 2
# CHECK-BE: beqctrl 0                       # encoding: [0x4d,0x82,0x04,0x21]
# CHECK-LE: beqctrl 0                       # encoding: [0x21,0x04,0x82,0x4d]
            beqctrl

# CHECK-BE: beq+ 2, target                  # encoding: [0x41,0xea,A,0bAAAAAA00]
# CHECK-LE: beq+ 2, target                  # encoding: [0bAAAAAA00,A,0xea,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            beq+ 2, target
# CHECK-BE: beq+ 0, target                  # encoding: [0x41,0xe2,A,0bAAAAAA00]
# CHECK-LE: beq+ 0, target                  # encoding: [0bAAAAAA00,A,0xe2,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            beq+ target
# CHECK-BE: beqa+ 2, target                 # encoding: [0x41,0xea,A,0bAAAAAA10]
# CHECK-LE: beqa+ 2, target                 # encoding: [0bAAAAAA10,A,0xea,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            beqa+ 2, target
# CHECK-BE: beqa+ 0, target                 # encoding: [0x41,0xe2,A,0bAAAAAA10]
# CHECK-LE: beqa+ 0, target                 # encoding: [0bAAAAAA10,A,0xe2,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            beqa+ target
# CHECK-BE: beqlr+ 2                        # encoding: [0x4d,0xea,0x00,0x20]
# CHECK-LE: beqlr+ 2                        # encoding: [0x20,0x00,0xea,0x4d]
            beqlr+ 2
# CHECK-BE: beqlr+ 0                        # encoding: [0x4d,0xe2,0x00,0x20]
# CHECK-LE: beqlr+ 0                        # encoding: [0x20,0x00,0xe2,0x4d]
            beqlr+
# CHECK-BE: beqctr+ 2                       # encoding: [0x4d,0xea,0x04,0x20]
# CHECK-LE: beqctr+ 2                       # encoding: [0x20,0x04,0xea,0x4d]
            beqctr+ 2
# CHECK-BE: beqctr+ 0                       # encoding: [0x4d,0xe2,0x04,0x20]
# CHECK-LE: beqctr+ 0                       # encoding: [0x20,0x04,0xe2,0x4d]
            beqctr+
# CHECK-BE: beql+ 2, target                 # encoding: [0x41,0xea,A,0bAAAAAA01]
# CHECK-LE: beql+ 2, target                 # encoding: [0bAAAAAA01,A,0xea,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            beql+ 2, target
# CHECK-BE: beql+ 0, target                 # encoding: [0x41,0xe2,A,0bAAAAAA01]
# CHECK-LE: beql+ 0, target                 # encoding: [0bAAAAAA01,A,0xe2,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            beql+ target
# CHECK-BE: beqla+ 2, target                # encoding: [0x41,0xea,A,0bAAAAAA11]
# CHECK-LE: beqla+ 2, target                # encoding: [0bAAAAAA11,A,0xea,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            beqla+ 2, target
# CHECK-BE: beqla+ 0, target                # encoding: [0x41,0xe2,A,0bAAAAAA11]
# CHECK-LE: beqla+ 0, target                # encoding: [0bAAAAAA11,A,0xe2,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            beqla+ target
# CHECK-BE: beqlrl+ 2                       # encoding: [0x4d,0xea,0x00,0x21]
# CHECK-LE: beqlrl+ 2                       # encoding: [0x21,0x00,0xea,0x4d]
            beqlrl+ 2
# CHECK-BE: beqlrl+ 0                       # encoding: [0x4d,0xe2,0x00,0x21]
# CHECK-LE: beqlrl+ 0                       # encoding: [0x21,0x00,0xe2,0x4d]
            beqlrl+
# CHECK-BE: beqctrl+ 2                      # encoding: [0x4d,0xea,0x04,0x21]
# CHECK-LE: beqctrl+ 2                      # encoding: [0x21,0x04,0xea,0x4d]
            beqctrl+ 2
# CHECK-BE: beqctrl+ 0                      # encoding: [0x4d,0xe2,0x04,0x21]
# CHECK-LE: beqctrl+ 0                      # encoding: [0x21,0x04,0xe2,0x4d]
            beqctrl+

# CHECK-BE: beq- 2, target                  # encoding: [0x41,0xca,A,0bAAAAAA00]
# CHECK-LE: beq- 2, target                  # encoding: [0bAAAAAA00,A,0xca,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            beq- 2, target
# CHECK-BE: beq- 0, target                  # encoding: [0x41,0xc2,A,0bAAAAAA00]
# CHECK-LE: beq- 0, target                  # encoding: [0bAAAAAA00,A,0xc2,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            beq- target
# CHECK-BE: beqa- 2, target                 # encoding: [0x41,0xca,A,0bAAAAAA10]
# CHECK-LE: beqa- 2, target                 # encoding: [0bAAAAAA10,A,0xca,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            beqa- 2, target
# CHECK-BE: beqa- 0, target                 # encoding: [0x41,0xc2,A,0bAAAAAA10]
# CHECK-LE: beqa- 0, target                 # encoding: [0bAAAAAA10,A,0xc2,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            beqa- target
# CHECK-BE: beqlr- 2                        # encoding: [0x4d,0xca,0x00,0x20]
# CHECK-LE: beqlr- 2                        # encoding: [0x20,0x00,0xca,0x4d]
            beqlr- 2
# CHECK-BE: beqlr- 0                        # encoding: [0x4d,0xc2,0x00,0x20]
# CHECK-LE: beqlr- 0                        # encoding: [0x20,0x00,0xc2,0x4d]
            beqlr-
# CHECK-BE: beqctr- 2                       # encoding: [0x4d,0xca,0x04,0x20]
# CHECK-LE: beqctr- 2                       # encoding: [0x20,0x04,0xca,0x4d]
            beqctr- 2
# CHECK-BE: beqctr- 0                       # encoding: [0x4d,0xc2,0x04,0x20]
# CHECK-LE: beqctr- 0                       # encoding: [0x20,0x04,0xc2,0x4d]
            beqctr-
# CHECK-BE: beql- 2, target                 # encoding: [0x41,0xca,A,0bAAAAAA01]
# CHECK-LE: beql- 2, target                 # encoding: [0bAAAAAA01,A,0xca,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            beql- 2, target
# CHECK-BE: beql- 0, target                 # encoding: [0x41,0xc2,A,0bAAAAAA01]
# CHECK-LE: beql- 0, target                 # encoding: [0bAAAAAA01,A,0xc2,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            beql- target
# CHECK-BE: beqla- 2, target                # encoding: [0x41,0xca,A,0bAAAAAA11]
# CHECK-LE: beqla- 2, target                # encoding: [0bAAAAAA11,A,0xca,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            beqla- 2, target
# CHECK-BE: beqla- 0, target                # encoding: [0x41,0xc2,A,0bAAAAAA11]
# CHECK-LE: beqla- 0, target                # encoding: [0bAAAAAA11,A,0xc2,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            beqla- target
# CHECK-BE: beqlrl- 2                       # encoding: [0x4d,0xca,0x00,0x21]
# CHECK-LE: beqlrl- 2                       # encoding: [0x21,0x00,0xca,0x4d]
            beqlrl- 2
# CHECK-BE: beqlrl- 0                       # encoding: [0x4d,0xc2,0x00,0x21]
# CHECK-LE: beqlrl- 0                       # encoding: [0x21,0x00,0xc2,0x4d]
            beqlrl-
# CHECK-BE: beqctrl- 2                      # encoding: [0x4d,0xca,0x04,0x21]
# CHECK-LE: beqctrl- 2                      # encoding: [0x21,0x04,0xca,0x4d]
            beqctrl- 2
# CHECK-BE: beqctrl- 0                      # encoding: [0x4d,0xc2,0x04,0x21]
# CHECK-LE: beqctrl- 0                      # encoding: [0x21,0x04,0xc2,0x4d]
            beqctrl-

# CHECK-BE: bge 2, target                   # encoding: [0x40,0x88,A,0bAAAAAA00]
# CHECK-LE: bge 2, target                   # encoding: [0bAAAAAA00,A,0x88,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bge 2, target
# CHECK-BE: bge 0, target                   # encoding: [0x40,0x80,A,0bAAAAAA00]
# CHECK-LE: bge 0, target                   # encoding: [0bAAAAAA00,A,0x80,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bge target
# CHECK-BE: bgea 2, target                  # encoding: [0x40,0x88,A,0bAAAAAA10]
# CHECK-LE: bgea 2, target                  # encoding: [0bAAAAAA10,A,0x88,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bgea 2, target
# CHECK-BE: bgea 0, target                  # encoding: [0x40,0x80,A,0bAAAAAA10]
# CHECK-LE: bgea 0, target                  # encoding: [0bAAAAAA10,A,0x80,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bgea target
# CHECK-BE: bgelr 2                         # encoding: [0x4c,0x88,0x00,0x20]
# CHECK-LE: bgelr 2                         # encoding: [0x20,0x00,0x88,0x4c]
            bgelr 2
# CHECK-BE: bgelr 0                         # encoding: [0x4c,0x80,0x00,0x20]
# CHECK-LE: bgelr 0                         # encoding: [0x20,0x00,0x80,0x4c]
            bgelr
# CHECK-BE: bgectr 2                        # encoding: [0x4c,0x88,0x04,0x20]
# CHECK-LE: bgectr 2                        # encoding: [0x20,0x04,0x88,0x4c]
            bgectr 2
# CHECK-BE: bgectr 0                        # encoding: [0x4c,0x80,0x04,0x20]
# CHECK-LE: bgectr 0                        # encoding: [0x20,0x04,0x80,0x4c]
            bgectr
# CHECK-BE: bgel 2, target                  # encoding: [0x40,0x88,A,0bAAAAAA01]
# CHECK-LE: bgel 2, target                  # encoding: [0bAAAAAA01,A,0x88,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bgel 2, target
# CHECK-BE: bgel 0, target                  # encoding: [0x40,0x80,A,0bAAAAAA01]
# CHECK-LE: bgel 0, target                  # encoding: [0bAAAAAA01,A,0x80,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bgel target
# CHECK-BE: bgela 2, target                 # encoding: [0x40,0x88,A,0bAAAAAA11]
# CHECK-LE: bgela 2, target                 # encoding: [0bAAAAAA11,A,0x88,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bgela 2, target
# CHECK-BE: bgela 0, target                 # encoding: [0x40,0x80,A,0bAAAAAA11]
# CHECK-LE: bgela 0, target                 # encoding: [0bAAAAAA11,A,0x80,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bgela target
# CHECK-BE: bgelrl 2                        # encoding: [0x4c,0x88,0x00,0x21]
# CHECK-LE: bgelrl 2                        # encoding: [0x21,0x00,0x88,0x4c]
            bgelrl 2
# CHECK-BE: bgelrl 0                        # encoding: [0x4c,0x80,0x00,0x21]
# CHECK-LE: bgelrl 0                        # encoding: [0x21,0x00,0x80,0x4c]
            bgelrl
# CHECK-BE: bgectrl 2                       # encoding: [0x4c,0x88,0x04,0x21]
# CHECK-LE: bgectrl 2                       # encoding: [0x21,0x04,0x88,0x4c]
            bgectrl 2
# CHECK-BE: bgectrl 0                       # encoding: [0x4c,0x80,0x04,0x21]
# CHECK-LE: bgectrl 0                       # encoding: [0x21,0x04,0x80,0x4c]
            bgectrl

# CHECK-BE: bge+ 2, target                   # encoding: [0x40,0xe8,A,0bAAAAAA00]
# CHECK-LE: bge+ 2, target                   # encoding: [0bAAAAAA00,A,0xe8,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bge+ 2, target
# CHECK-BE: bge+ 0, target                   # encoding: [0x40,0xe0,A,0bAAAAAA00]
# CHECK-LE: bge+ 0, target                   # encoding: [0bAAAAAA00,A,0xe0,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bge+ target
# CHECK-BE: bgea+ 2, target                  # encoding: [0x40,0xe8,A,0bAAAAAA10]
# CHECK-LE: bgea+ 2, target                  # encoding: [0bAAAAAA10,A,0xe8,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bgea+ 2, target
# CHECK-BE: bgea+ 0, target                  # encoding: [0x40,0xe0,A,0bAAAAAA10]
# CHECK-LE: bgea+ 0, target                  # encoding: [0bAAAAAA10,A,0xe0,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bgea+ target
# CHECK-BE: bgelr+ 2                         # encoding: [0x4c,0xe8,0x00,0x20]
# CHECK-LE: bgelr+ 2                         # encoding: [0x20,0x00,0xe8,0x4c]
            bgelr+ 2
# CHECK-BE: bgelr+ 0                         # encoding: [0x4c,0xe0,0x00,0x20]
# CHECK-LE: bgelr+ 0                         # encoding: [0x20,0x00,0xe0,0x4c]
            bgelr+
# CHECK-BE: bgectr+ 2                        # encoding: [0x4c,0xe8,0x04,0x20]
# CHECK-LE: bgectr+ 2                        # encoding: [0x20,0x04,0xe8,0x4c]
            bgectr+ 2
# CHECK-BE: bgectr+ 0                        # encoding: [0x4c,0xe0,0x04,0x20]
# CHECK-LE: bgectr+ 0                        # encoding: [0x20,0x04,0xe0,0x4c]
            bgectr+
# CHECK-BE: bgel+ 2, target                  # encoding: [0x40,0xe8,A,0bAAAAAA01]
# CHECK-LE: bgel+ 2, target                  # encoding: [0bAAAAAA01,A,0xe8,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bgel+ 2, target
# CHECK-BE: bgel+ 0, target                  # encoding: [0x40,0xe0,A,0bAAAAAA01]
# CHECK-LE: bgel+ 0, target                  # encoding: [0bAAAAAA01,A,0xe0,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bgel+ target
# CHECK-BE: bgela+ 2, target                 # encoding: [0x40,0xe8,A,0bAAAAAA11]
# CHECK-LE: bgela+ 2, target                 # encoding: [0bAAAAAA11,A,0xe8,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bgela+ 2, target
# CHECK-BE: bgela+ 0, target                 # encoding: [0x40,0xe0,A,0bAAAAAA11]
# CHECK-LE: bgela+ 0, target                 # encoding: [0bAAAAAA11,A,0xe0,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bgela+ target
# CHECK-BE: bgelrl+ 2                        # encoding: [0x4c,0xe8,0x00,0x21]
# CHECK-LE: bgelrl+ 2                        # encoding: [0x21,0x00,0xe8,0x4c]
            bgelrl+ 2
# CHECK-BE: bgelrl+ 0                        # encoding: [0x4c,0xe0,0x00,0x21]
# CHECK-LE: bgelrl+ 0                        # encoding: [0x21,0x00,0xe0,0x4c]
            bgelrl+
# CHECK-BE: bgectrl+ 2                       # encoding: [0x4c,0xe8,0x04,0x21]
# CHECK-LE: bgectrl+ 2                       # encoding: [0x21,0x04,0xe8,0x4c]
            bgectrl+ 2
# CHECK-BE: bgectrl+ 0                       # encoding: [0x4c,0xe0,0x04,0x21]
# CHECK-LE: bgectrl+ 0                       # encoding: [0x21,0x04,0xe0,0x4c]
            bgectrl+

# CHECK-BE: bge- 2, target                   # encoding: [0x40,0xc8,A,0bAAAAAA00]
# CHECK-LE: bge- 2, target                   # encoding: [0bAAAAAA00,A,0xc8,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bge- 2, target
# CHECK-BE: bge- 0, target                   # encoding: [0x40,0xc0,A,0bAAAAAA00]
# CHECK-LE: bge- 0, target                   # encoding: [0bAAAAAA00,A,0xc0,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bge- target
# CHECK-BE: bgea- 2, target                  # encoding: [0x40,0xc8,A,0bAAAAAA10]
# CHECK-LE: bgea- 2, target                  # encoding: [0bAAAAAA10,A,0xc8,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bgea- 2, target
# CHECK-BE: bgea- 0, target                  # encoding: [0x40,0xc0,A,0bAAAAAA10]
# CHECK-LE: bgea- 0, target                  # encoding: [0bAAAAAA10,A,0xc0,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bgea- target
# CHECK-BE: bgelr- 2                         # encoding: [0x4c,0xc8,0x00,0x20]
# CHECK-LE: bgelr- 2                         # encoding: [0x20,0x00,0xc8,0x4c]
            bgelr- 2
# CHECK-BE: bgelr- 0                         # encoding: [0x4c,0xc0,0x00,0x20]
# CHECK-LE: bgelr- 0                         # encoding: [0x20,0x00,0xc0,0x4c]
            bgelr-
# CHECK-BE: bgectr- 2                        # encoding: [0x4c,0xc8,0x04,0x20]
# CHECK-LE: bgectr- 2                        # encoding: [0x20,0x04,0xc8,0x4c]
            bgectr- 2
# CHECK-BE: bgectr- 0                        # encoding: [0x4c,0xc0,0x04,0x20]
# CHECK-LE: bgectr- 0                        # encoding: [0x20,0x04,0xc0,0x4c]
            bgectr-
# CHECK-BE: bgel- 2, target                  # encoding: [0x40,0xc8,A,0bAAAAAA01]
# CHECK-LE: bgel- 2, target                  # encoding: [0bAAAAAA01,A,0xc8,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bgel- 2, target
# CHECK-BE: bgel- 0, target                  # encoding: [0x40,0xc0,A,0bAAAAAA01]
# CHECK-LE: bgel- 0, target                  # encoding: [0bAAAAAA01,A,0xc0,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bgel- target
# CHECK-BE: bgela- 2, target                 # encoding: [0x40,0xc8,A,0bAAAAAA11]
# CHECK-LE: bgela- 2, target                 # encoding: [0bAAAAAA11,A,0xc8,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bgela- 2, target
# CHECK-BE: bgela- 0, target                 # encoding: [0x40,0xc0,A,0bAAAAAA11]
# CHECK-LE: bgela- 0, target                 # encoding: [0bAAAAAA11,A,0xc0,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bgela- target
# CHECK-BE: bgelrl- 2                        # encoding: [0x4c,0xc8,0x00,0x21]
# CHECK-LE: bgelrl- 2                        # encoding: [0x21,0x00,0xc8,0x4c]
            bgelrl- 2
# CHECK-BE: bgelrl- 0                        # encoding: [0x4c,0xc0,0x00,0x21]
# CHECK-LE: bgelrl- 0                        # encoding: [0x21,0x00,0xc0,0x4c]
            bgelrl-
# CHECK-BE: bgectrl- 2                       # encoding: [0x4c,0xc8,0x04,0x21]
# CHECK-LE: bgectrl- 2                       # encoding: [0x21,0x04,0xc8,0x4c]
            bgectrl- 2
# CHECK-BE: bgectrl- 0                       # encoding: [0x4c,0xc0,0x04,0x21]
# CHECK-LE: bgectrl- 0                       # encoding: [0x21,0x04,0xc0,0x4c]
            bgectrl-

# CHECK-BE: bgt 2, target                   # encoding: [0x41,0x89,A,0bAAAAAA00]
# CHECK-LE: bgt 2, target                   # encoding: [0bAAAAAA00,A,0x89,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bgt 2, target
# CHECK-BE: bgt 0, target                   # encoding: [0x41,0x81,A,0bAAAAAA00]
# CHECK-LE: bgt 0, target                   # encoding: [0bAAAAAA00,A,0x81,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bgt target
# CHECK-BE: bgta 2, target                  # encoding: [0x41,0x89,A,0bAAAAAA10]
# CHECK-LE: bgta 2, target                  # encoding: [0bAAAAAA10,A,0x89,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bgta 2, target
# CHECK-BE: bgta 0, target                  # encoding: [0x41,0x81,A,0bAAAAAA10]
# CHECK-LE: bgta 0, target                  # encoding: [0bAAAAAA10,A,0x81,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bgta target
# CHECK-BE: bgtlr 2                         # encoding: [0x4d,0x89,0x00,0x20]
# CHECK-LE: bgtlr 2                         # encoding: [0x20,0x00,0x89,0x4d]
            bgtlr 2
# CHECK-BE: bgtlr 0                         # encoding: [0x4d,0x81,0x00,0x20]
# CHECK-LE: bgtlr 0                         # encoding: [0x20,0x00,0x81,0x4d]
            bgtlr
# CHECK-BE: bgtctr 2                        # encoding: [0x4d,0x89,0x04,0x20]
# CHECK-LE: bgtctr 2                        # encoding: [0x20,0x04,0x89,0x4d]
            bgtctr 2
# CHECK-BE: bgtctr 0                        # encoding: [0x4d,0x81,0x04,0x20]
# CHECK-LE: bgtctr 0                        # encoding: [0x20,0x04,0x81,0x4d]
            bgtctr
# CHECK-BE: bgtl 2, target                  # encoding: [0x41,0x89,A,0bAAAAAA01]
# CHECK-LE: bgtl 2, target                  # encoding: [0bAAAAAA01,A,0x89,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bgtl 2, target
# CHECK-BE: bgtl 0, target                  # encoding: [0x41,0x81,A,0bAAAAAA01]
# CHECK-LE: bgtl 0, target                  # encoding: [0bAAAAAA01,A,0x81,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bgtl target
# CHECK-BE: bgtla 2, target                 # encoding: [0x41,0x89,A,0bAAAAAA11]
# CHECK-LE: bgtla 2, target                 # encoding: [0bAAAAAA11,A,0x89,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bgtla 2, target
# CHECK-BE: bgtla 0, target                 # encoding: [0x41,0x81,A,0bAAAAAA11]
# CHECK-LE: bgtla 0, target                 # encoding: [0bAAAAAA11,A,0x81,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bgtla target
# CHECK-BE: bgtlrl 2                        # encoding: [0x4d,0x89,0x00,0x21]
# CHECK-LE: bgtlrl 2                        # encoding: [0x21,0x00,0x89,0x4d]
            bgtlrl 2
# CHECK-BE: bgtlrl 0                        # encoding: [0x4d,0x81,0x00,0x21]
# CHECK-LE: bgtlrl 0                        # encoding: [0x21,0x00,0x81,0x4d]
            bgtlrl
# CHECK-BE: bgtctrl 2                       # encoding: [0x4d,0x89,0x04,0x21]
# CHECK-LE: bgtctrl 2                       # encoding: [0x21,0x04,0x89,0x4d]
            bgtctrl 2
# CHECK-BE: bgtctrl 0                       # encoding: [0x4d,0x81,0x04,0x21]
# CHECK-LE: bgtctrl 0                       # encoding: [0x21,0x04,0x81,0x4d]
            bgtctrl

# CHECK-BE: bgt+ 2, target                  # encoding: [0x41,0xe9,A,0bAAAAAA00]
# CHECK-LE: bgt+ 2, target                  # encoding: [0bAAAAAA00,A,0xe9,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bgt+ 2, target
# CHECK-BE: bgt+ 0, target                  # encoding: [0x41,0xe1,A,0bAAAAAA00]
# CHECK-LE: bgt+ 0, target                  # encoding: [0bAAAAAA00,A,0xe1,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bgt+ target
# CHECK-BE: bgta+ 2, target                 # encoding: [0x41,0xe9,A,0bAAAAAA10]
# CHECK-LE: bgta+ 2, target                 # encoding: [0bAAAAAA10,A,0xe9,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bgta+ 2, target
# CHECK-BE: bgta+ 0, target                 # encoding: [0x41,0xe1,A,0bAAAAAA10]
# CHECK-LE: bgta+ 0, target                 # encoding: [0bAAAAAA10,A,0xe1,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bgta+ target
# CHECK-BE: bgtlr+ 2                        # encoding: [0x4d,0xe9,0x00,0x20]
# CHECK-LE: bgtlr+ 2                        # encoding: [0x20,0x00,0xe9,0x4d]
            bgtlr+ 2
# CHECK-BE: bgtlr+ 0                        # encoding: [0x4d,0xe1,0x00,0x20]
# CHECK-LE: bgtlr+ 0                        # encoding: [0x20,0x00,0xe1,0x4d]
            bgtlr+
# CHECK-BE: bgtctr+ 2                       # encoding: [0x4d,0xe9,0x04,0x20]
# CHECK-LE: bgtctr+ 2                       # encoding: [0x20,0x04,0xe9,0x4d]
            bgtctr+ 2
# CHECK-BE: bgtctr+ 0                       # encoding: [0x4d,0xe1,0x04,0x20]
# CHECK-LE: bgtctr+ 0                       # encoding: [0x20,0x04,0xe1,0x4d]
            bgtctr+
# CHECK-BE: bgtl+ 2, target                 # encoding: [0x41,0xe9,A,0bAAAAAA01]
# CHECK-LE: bgtl+ 2, target                 # encoding: [0bAAAAAA01,A,0xe9,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bgtl+ 2, target
# CHECK-BE: bgtl+ 0, target                 # encoding: [0x41,0xe1,A,0bAAAAAA01]
# CHECK-LE: bgtl+ 0, target                 # encoding: [0bAAAAAA01,A,0xe1,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bgtl+ target
# CHECK-BE: bgtla+ 2, target                # encoding: [0x41,0xe9,A,0bAAAAAA11]
# CHECK-LE: bgtla+ 2, target                # encoding: [0bAAAAAA11,A,0xe9,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bgtla+ 2, target
# CHECK-BE: bgtla+ 0, target                # encoding: [0x41,0xe1,A,0bAAAAAA11]
# CHECK-LE: bgtla+ 0, target                # encoding: [0bAAAAAA11,A,0xe1,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bgtla+ target
# CHECK-BE: bgtlrl+ 2                       # encoding: [0x4d,0xe9,0x00,0x21]
# CHECK-LE: bgtlrl+ 2                       # encoding: [0x21,0x00,0xe9,0x4d]
            bgtlrl+ 2
# CHECK-BE: bgtlrl+ 0                       # encoding: [0x4d,0xe1,0x00,0x21]
# CHECK-LE: bgtlrl+ 0                       # encoding: [0x21,0x00,0xe1,0x4d]
            bgtlrl+
# CHECK-BE: bgtctrl+ 2                      # encoding: [0x4d,0xe9,0x04,0x21]
# CHECK-LE: bgtctrl+ 2                      # encoding: [0x21,0x04,0xe9,0x4d]
            bgtctrl+ 2
# CHECK-BE: bgtctrl+ 0                      # encoding: [0x4d,0xe1,0x04,0x21]
# CHECK-LE: bgtctrl+ 0                      # encoding: [0x21,0x04,0xe1,0x4d]
            bgtctrl+

# CHECK-BE: bgt- 2, target                  # encoding: [0x41,0xc9,A,0bAAAAAA00]
# CHECK-LE: bgt- 2, target                  # encoding: [0bAAAAAA00,A,0xc9,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bgt- 2, target
# CHECK-BE: bgt- 0, target                  # encoding: [0x41,0xc1,A,0bAAAAAA00]
# CHECK-LE: bgt- 0, target                  # encoding: [0bAAAAAA00,A,0xc1,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bgt- target
# CHECK-BE: bgta- 2, target                 # encoding: [0x41,0xc9,A,0bAAAAAA10]
# CHECK-LE: bgta- 2, target                 # encoding: [0bAAAAAA10,A,0xc9,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bgta- 2, target
# CHECK-BE: bgta- 0, target                 # encoding: [0x41,0xc1,A,0bAAAAAA10]
# CHECK-LE: bgta- 0, target                 # encoding: [0bAAAAAA10,A,0xc1,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bgta- target
# CHECK-BE: bgtlr- 2                        # encoding: [0x4d,0xc9,0x00,0x20]
# CHECK-LE: bgtlr- 2                        # encoding: [0x20,0x00,0xc9,0x4d]
            bgtlr- 2
# CHECK-BE: bgtlr- 0                        # encoding: [0x4d,0xc1,0x00,0x20]
# CHECK-LE: bgtlr- 0                        # encoding: [0x20,0x00,0xc1,0x4d]
            bgtlr-
# CHECK-BE: bgtctr- 2                       # encoding: [0x4d,0xc9,0x04,0x20]
# CHECK-LE: bgtctr- 2                       # encoding: [0x20,0x04,0xc9,0x4d]
            bgtctr- 2
# CHECK-BE: bgtctr- 0                       # encoding: [0x4d,0xc1,0x04,0x20]
# CHECK-LE: bgtctr- 0                       # encoding: [0x20,0x04,0xc1,0x4d]
            bgtctr-
# CHECK-BE: bgtl- 2, target                 # encoding: [0x41,0xc9,A,0bAAAAAA01]
# CHECK-LE: bgtl- 2, target                 # encoding: [0bAAAAAA01,A,0xc9,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bgtl- 2, target
# CHECK-BE: bgtl- 0, target                 # encoding: [0x41,0xc1,A,0bAAAAAA01]
# CHECK-LE: bgtl- 0, target                 # encoding: [0bAAAAAA01,A,0xc1,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bgtl- target
# CHECK-BE: bgtla- 2, target                # encoding: [0x41,0xc9,A,0bAAAAAA11]
# CHECK-LE: bgtla- 2, target                # encoding: [0bAAAAAA11,A,0xc9,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bgtla- 2, target
# CHECK-BE: bgtla- 0, target                # encoding: [0x41,0xc1,A,0bAAAAAA11]
# CHECK-LE: bgtla- 0, target                # encoding: [0bAAAAAA11,A,0xc1,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bgtla- target
# CHECK-BE: bgtlrl- 2                       # encoding: [0x4d,0xc9,0x00,0x21]
# CHECK-LE: bgtlrl- 2                       # encoding: [0x21,0x00,0xc9,0x4d]
            bgtlrl- 2
# CHECK-BE: bgtlrl- 0                       # encoding: [0x4d,0xc1,0x00,0x21]
# CHECK-LE: bgtlrl- 0                       # encoding: [0x21,0x00,0xc1,0x4d]
            bgtlrl-
# CHECK-BE: bgtctrl- 2                      # encoding: [0x4d,0xc9,0x04,0x21]
# CHECK-LE: bgtctrl- 2                      # encoding: [0x21,0x04,0xc9,0x4d]
            bgtctrl- 2
# CHECK-BE: bgtctrl- 0                      # encoding: [0x4d,0xc1,0x04,0x21]
# CHECK-LE: bgtctrl- 0                      # encoding: [0x21,0x04,0xc1,0x4d]
            bgtctrl-

# CHECK-BE: bge 2, target                   # encoding: [0x40,0x88,A,0bAAAAAA00]
# CHECK-LE: bge 2, target                   # encoding: [0bAAAAAA00,A,0x88,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnl 2, target
# CHECK-BE: bge 0, target                   # encoding: [0x40,0x80,A,0bAAAAAA00]
# CHECK-LE: bge 0, target                   # encoding: [0bAAAAAA00,A,0x80,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnl target
# CHECK-BE: bgea 2, target                  # encoding: [0x40,0x88,A,0bAAAAAA10]
# CHECK-LE: bgea 2, target                  # encoding: [0bAAAAAA10,A,0x88,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnla 2, target
# CHECK-BE: bgea 0, target                  # encoding: [0x40,0x80,A,0bAAAAAA10]
# CHECK-LE: bgea 0, target                  # encoding: [0bAAAAAA10,A,0x80,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnla target
# CHECK-BE: bgelr 2                         # encoding: [0x4c,0x88,0x00,0x20]
# CHECK-LE: bgelr 2                         # encoding: [0x20,0x00,0x88,0x4c]
            bnllr 2
# CHECK-BE: bgelr 0                         # encoding: [0x4c,0x80,0x00,0x20]
# CHECK-LE: bgelr 0                         # encoding: [0x20,0x00,0x80,0x4c]
            bnllr
# CHECK-BE: bgectr 2                        # encoding: [0x4c,0x88,0x04,0x20]
# CHECK-LE: bgectr 2                        # encoding: [0x20,0x04,0x88,0x4c]
            bnlctr 2
# CHECK-BE: bgectr 0                        # encoding: [0x4c,0x80,0x04,0x20]
# CHECK-LE: bgectr 0                        # encoding: [0x20,0x04,0x80,0x4c]
            bnlctr
# CHECK-BE: bgel 2, target                  # encoding: [0x40,0x88,A,0bAAAAAA01]
# CHECK-LE: bgel 2, target                  # encoding: [0bAAAAAA01,A,0x88,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnll 2, target
# CHECK-BE: bgel 0, target                  # encoding: [0x40,0x80,A,0bAAAAAA01]
# CHECK-LE: bgel 0, target                  # encoding: [0bAAAAAA01,A,0x80,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnll target
# CHECK-BE: bgela 2, target                  # encoding: [0x40,0x88,A,0bAAAAAA11]
# CHECK-LE: bgela 2, target                  # encoding: [0bAAAAAA11,A,0x88,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnlla 2, target
# CHECK-BE: bgela 0, target                  # encoding: [0x40,0x80,A,0bAAAAAA11]
# CHECK-LE: bgela 0, target                  # encoding: [0bAAAAAA11,A,0x80,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnlla target
# CHECK-BE: bgelrl 2                        # encoding: [0x4c,0x88,0x00,0x21]
# CHECK-LE: bgelrl 2                        # encoding: [0x21,0x00,0x88,0x4c]
            bnllrl 2
# CHECK-BE: bgelrl 0                        # encoding: [0x4c,0x80,0x00,0x21]
# CHECK-LE: bgelrl 0                        # encoding: [0x21,0x00,0x80,0x4c]
            bnllrl
# CHECK-BE: bgectrl 2                       # encoding: [0x4c,0x88,0x04,0x21]
# CHECK-LE: bgectrl 2                       # encoding: [0x21,0x04,0x88,0x4c]
            bnlctrl 2
# CHECK-BE: bgectrl 0                       # encoding: [0x4c,0x80,0x04,0x21]
# CHECK-LE: bgectrl 0                       # encoding: [0x21,0x04,0x80,0x4c]
            bnlctrl

# CHECK-BE: bge+ 2, target                  # encoding: [0x40,0xe8,A,0bAAAAAA00]
# CHECK-LE: bge+ 2, target                  # encoding: [0bAAAAAA00,A,0xe8,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnl+ 2, target
# CHECK-BE: bge+ 0, target                  # encoding: [0x40,0xe0,A,0bAAAAAA00]
# CHECK-LE: bge+ 0, target                  # encoding: [0bAAAAAA00,A,0xe0,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnl+ target
# CHECK-BE: bgea+ 2, target                 # encoding: [0x40,0xe8,A,0bAAAAAA10]
# CHECK-LE: bgea+ 2, target                 # encoding: [0bAAAAAA10,A,0xe8,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnla+ 2, target
# CHECK-BE: bgea+ 0, target                 # encoding: [0x40,0xe0,A,0bAAAAAA10]
# CHECK-LE: bgea+ 0, target                 # encoding: [0bAAAAAA10,A,0xe0,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnla+ target
# CHECK-BE: bgelr+ 2                        # encoding: [0x4c,0xe8,0x00,0x20]
# CHECK-LE: bgelr+ 2                        # encoding: [0x20,0x00,0xe8,0x4c]
            bnllr+ 2
# CHECK-BE: bgelr+ 0                        # encoding: [0x4c,0xe0,0x00,0x20]
# CHECK-LE: bgelr+ 0                        # encoding: [0x20,0x00,0xe0,0x4c]
            bnllr+
# CHECK-BE: bgectr+ 2                       # encoding: [0x4c,0xe8,0x04,0x20]
# CHECK-LE: bgectr+ 2                       # encoding: [0x20,0x04,0xe8,0x4c]
            bnlctr+ 2
# CHECK-BE: bgectr+ 0                       # encoding: [0x4c,0xe0,0x04,0x20]
# CHECK-LE: bgectr+ 0                       # encoding: [0x20,0x04,0xe0,0x4c]
            bnlctr+
# CHECK-BE: bgel+ 2, target                 # encoding: [0x40,0xe8,A,0bAAAAAA01]
# CHECK-LE: bgel+ 2, target                 # encoding: [0bAAAAAA01,A,0xe8,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnll+ 2, target
# CHECK-BE: bgel+ 0, target                 # encoding: [0x40,0xe0,A,0bAAAAAA01]
# CHECK-LE: bgel+ 0, target                 # encoding: [0bAAAAAA01,A,0xe0,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnll+ target
# CHECK-BE: bgela+ 2, target                # encoding: [0x40,0xe8,A,0bAAAAAA11]
# CHECK-LE: bgela+ 2, target                # encoding: [0bAAAAAA11,A,0xe8,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnlla+ 2, target
# CHECK-BE: bgela+ 0, target                # encoding: [0x40,0xe0,A,0bAAAAAA11]
# CHECK-LE: bgela+ 0, target                # encoding: [0bAAAAAA11,A,0xe0,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnlla+ target
# CHECK-BE: bgelrl+ 2                       # encoding: [0x4c,0xe8,0x00,0x21]
# CHECK-LE: bgelrl+ 2                       # encoding: [0x21,0x00,0xe8,0x4c]
            bnllrl+ 2
# CHECK-BE: bgelrl+ 0                       # encoding: [0x4c,0xe0,0x00,0x21]
# CHECK-LE: bgelrl+ 0                       # encoding: [0x21,0x00,0xe0,0x4c]
            bnllrl+
# CHECK-BE: bgectrl+ 2                      # encoding: [0x4c,0xe8,0x04,0x21]
# CHECK-LE: bgectrl+ 2                      # encoding: [0x21,0x04,0xe8,0x4c]
            bnlctrl+ 2
# CHECK-BE: bgectrl+ 0                      # encoding: [0x4c,0xe0,0x04,0x21]
# CHECK-LE: bgectrl+ 0                      # encoding: [0x21,0x04,0xe0,0x4c]
            bnlctrl+

# CHECK-BE: bge- 2, target                  # encoding: [0x40,0xc8,A,0bAAAAAA00]
# CHECK-LE: bge- 2, target                  # encoding: [0bAAAAAA00,A,0xc8,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnl- 2, target
# CHECK-BE: bge- 0, target                  # encoding: [0x40,0xc0,A,0bAAAAAA00]
# CHECK-LE: bge- 0, target                  # encoding: [0bAAAAAA00,A,0xc0,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnl- target
# CHECK-BE: bgea- 2, target                 # encoding: [0x40,0xc8,A,0bAAAAAA10]
# CHECK-LE: bgea- 2, target                 # encoding: [0bAAAAAA10,A,0xc8,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnla- 2, target
# CHECK-BE: bgea- 0, target                 # encoding: [0x40,0xc0,A,0bAAAAAA10]
# CHECK-LE: bgea- 0, target                 # encoding: [0bAAAAAA10,A,0xc0,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnla- target
# CHECK-BE: bgelr- 2                        # encoding: [0x4c,0xc8,0x00,0x20]
# CHECK-LE: bgelr- 2                        # encoding: [0x20,0x00,0xc8,0x4c]
            bnllr- 2
# CHECK-BE: bgelr- 0                        # encoding: [0x4c,0xc0,0x00,0x20]
# CHECK-LE: bgelr- 0                        # encoding: [0x20,0x00,0xc0,0x4c]
            bnllr-
# CHECK-BE: bgectr- 2                       # encoding: [0x4c,0xc8,0x04,0x20]
# CHECK-LE: bgectr- 2                       # encoding: [0x20,0x04,0xc8,0x4c]
            bnlctr- 2
# CHECK-BE: bgectr- 0                       # encoding: [0x4c,0xc0,0x04,0x20]
# CHECK-LE: bgectr- 0                       # encoding: [0x20,0x04,0xc0,0x4c]
            bnlctr-
# CHECK-BE: bgel- 2, target                 # encoding: [0x40,0xc8,A,0bAAAAAA01]
# CHECK-LE: bgel- 2, target                 # encoding: [0bAAAAAA01,A,0xc8,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnll- 2, target
# CHECK-BE: bgel- 0, target                 # encoding: [0x40,0xc0,A,0bAAAAAA01]
# CHECK-LE: bgel- 0, target                 # encoding: [0bAAAAAA01,A,0xc0,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnll- target
# CHECK-BE: bgela- 2, target                # encoding: [0x40,0xc8,A,0bAAAAAA11]
# CHECK-LE: bgela- 2, target                # encoding: [0bAAAAAA11,A,0xc8,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnlla- 2, target
# CHECK-BE: bgela- 0, target                # encoding: [0x40,0xc0,A,0bAAAAAA11]
# CHECK-LE: bgela- 0, target                # encoding: [0bAAAAAA11,A,0xc0,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnlla- target
# CHECK-BE: bgelrl- 2                       # encoding: [0x4c,0xc8,0x00,0x21]
# CHECK-LE: bgelrl- 2                       # encoding: [0x21,0x00,0xc8,0x4c]
            bnllrl- 2
# CHECK-BE: bgelrl- 0                       # encoding: [0x4c,0xc0,0x00,0x21]
# CHECK-LE: bgelrl- 0                       # encoding: [0x21,0x00,0xc0,0x4c]
            bnllrl-
# CHECK-BE: bgectrl- 2                      # encoding: [0x4c,0xc8,0x04,0x21]
# CHECK-LE: bgectrl- 2                      # encoding: [0x21,0x04,0xc8,0x4c]
            bnlctrl- 2
# CHECK-BE: bgectrl- 0                      # encoding: [0x4c,0xc0,0x04,0x21]
# CHECK-LE: bgectrl- 0                      # encoding: [0x21,0x04,0xc0,0x4c]
            bnlctrl-

# CHECK-BE: bne 2, target                   # encoding: [0x40,0x8a,A,0bAAAAAA00]
# CHECK-LE: bne 2, target                   # encoding: [0bAAAAAA00,A,0x8a,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bne 2, target
# CHECK-BE: bne 0, target                   # encoding: [0x40,0x82,A,0bAAAAAA00]
# CHECK-LE: bne 0, target                   # encoding: [0bAAAAAA00,A,0x82,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bne target
# CHECK-BE: bnea 2, target                  # encoding: [0x40,0x8a,A,0bAAAAAA10]
# CHECK-LE: bnea 2, target                  # encoding: [0bAAAAAA10,A,0x8a,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnea 2, target
# CHECK-BE: bnea 0, target                  # encoding: [0x40,0x82,A,0bAAAAAA10]
# CHECK-LE: bnea 0, target                  # encoding: [0bAAAAAA10,A,0x82,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnea target
# CHECK-BE: bnelr 2                         # encoding: [0x4c,0x8a,0x00,0x20]
# CHECK-LE: bnelr 2                         # encoding: [0x20,0x00,0x8a,0x4c]
            bnelr 2
# CHECK-BE: bnelr 0                         # encoding: [0x4c,0x82,0x00,0x20]
# CHECK-LE: bnelr 0                         # encoding: [0x20,0x00,0x82,0x4c]
            bnelr
# CHECK-BE: bnectr 2                        # encoding: [0x4c,0x8a,0x04,0x20]
# CHECK-LE: bnectr 2                        # encoding: [0x20,0x04,0x8a,0x4c]
            bnectr 2
# CHECK-BE: bnectr 0                        # encoding: [0x4c,0x82,0x04,0x20]
# CHECK-LE: bnectr 0                        # encoding: [0x20,0x04,0x82,0x4c]
            bnectr
# CHECK-BE: bnel 2, target                  # encoding: [0x40,0x8a,A,0bAAAAAA01]
# CHECK-LE: bnel 2, target                  # encoding: [0bAAAAAA01,A,0x8a,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnel 2, target
# CHECK-BE: bnel 0, target                  # encoding: [0x40,0x82,A,0bAAAAAA01]
# CHECK-LE: bnel 0, target                  # encoding: [0bAAAAAA01,A,0x82,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnel target
# CHECK-BE: bnela 2, target                 # encoding: [0x40,0x8a,A,0bAAAAAA11]
# CHECK-LE: bnela 2, target                 # encoding: [0bAAAAAA11,A,0x8a,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnela 2, target
# CHECK-BE: bnela 0, target                 # encoding: [0x40,0x82,A,0bAAAAAA11]
# CHECK-LE: bnela 0, target                 # encoding: [0bAAAAAA11,A,0x82,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnela target
# CHECK-BE: bnelrl 2                        # encoding: [0x4c,0x8a,0x00,0x21]
# CHECK-LE: bnelrl 2                        # encoding: [0x21,0x00,0x8a,0x4c]
            bnelrl 2
# CHECK-BE: bnelrl 0                        # encoding: [0x4c,0x82,0x00,0x21]
# CHECK-LE: bnelrl 0                        # encoding: [0x21,0x00,0x82,0x4c]
            bnelrl
# CHECK-BE: bnectrl 2                       # encoding: [0x4c,0x8a,0x04,0x21]
# CHECK-LE: bnectrl 2                       # encoding: [0x21,0x04,0x8a,0x4c]
            bnectrl 2
# CHECK-BE: bnectrl 0                       # encoding: [0x4c,0x82,0x04,0x21]
# CHECK-LE: bnectrl 0                       # encoding: [0x21,0x04,0x82,0x4c]
            bnectrl

# CHECK-BE: bne+ 2, target                  # encoding: [0x40,0xea,A,0bAAAAAA00]
# CHECK-LE: bne+ 2, target                  # encoding: [0bAAAAAA00,A,0xea,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bne+ 2, target
# CHECK-BE: bne+ 0, target                  # encoding: [0x40,0xe2,A,0bAAAAAA00]
# CHECK-LE: bne+ 0, target                  # encoding: [0bAAAAAA00,A,0xe2,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bne+ target
# CHECK-BE: bnea+ 2, target                 # encoding: [0x40,0xea,A,0bAAAAAA10]
# CHECK-LE: bnea+ 2, target                 # encoding: [0bAAAAAA10,A,0xea,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnea+ 2, target
# CHECK-BE: bnea+ 0, target                 # encoding: [0x40,0xe2,A,0bAAAAAA10]
# CHECK-LE: bnea+ 0, target                 # encoding: [0bAAAAAA10,A,0xe2,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnea+ target
# CHECK-BE: bnelr+ 2                        # encoding: [0x4c,0xea,0x00,0x20]
# CHECK-LE: bnelr+ 2                        # encoding: [0x20,0x00,0xea,0x4c]
            bnelr+ 2
# CHECK-BE: bnelr+ 0                        # encoding: [0x4c,0xe2,0x00,0x20]
# CHECK-LE: bnelr+ 0                        # encoding: [0x20,0x00,0xe2,0x4c]
            bnelr+
# CHECK-BE: bnectr+ 2                       # encoding: [0x4c,0xea,0x04,0x20]
# CHECK-LE: bnectr+ 2                       # encoding: [0x20,0x04,0xea,0x4c]
            bnectr+ 2
# CHECK-BE: bnectr+ 0                       # encoding: [0x4c,0xe2,0x04,0x20]
# CHECK-LE: bnectr+ 0                       # encoding: [0x20,0x04,0xe2,0x4c]
            bnectr+
# CHECK-BE: bnel+ 2, target                 # encoding: [0x40,0xea,A,0bAAAAAA01]
# CHECK-LE: bnel+ 2, target                 # encoding: [0bAAAAAA01,A,0xea,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnel+ 2, target
# CHECK-BE: bnel+ 0, target                 # encoding: [0x40,0xe2,A,0bAAAAAA01]
# CHECK-LE: bnel+ 0, target                 # encoding: [0bAAAAAA01,A,0xe2,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnel+ target
# CHECK-BE: bnela+ 2, target                # encoding: [0x40,0xea,A,0bAAAAAA11]
# CHECK-LE: bnela+ 2, target                # encoding: [0bAAAAAA11,A,0xea,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnela+ 2, target
# CHECK-BE: bnela+ 0, target                # encoding: [0x40,0xe2,A,0bAAAAAA11]
# CHECK-LE: bnela+ 0, target                # encoding: [0bAAAAAA11,A,0xe2,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnela+ target
# CHECK-BE: bnelrl+ 2                       # encoding: [0x4c,0xea,0x00,0x21]
# CHECK-LE: bnelrl+ 2                       # encoding: [0x21,0x00,0xea,0x4c]
            bnelrl+ 2
# CHECK-BE: bnelrl+ 0                       # encoding: [0x4c,0xe2,0x00,0x21]
# CHECK-LE: bnelrl+ 0                       # encoding: [0x21,0x00,0xe2,0x4c]
            bnelrl+
# CHECK-BE: bnectrl+ 2                      # encoding: [0x4c,0xea,0x04,0x21]
# CHECK-LE: bnectrl+ 2                      # encoding: [0x21,0x04,0xea,0x4c]
            bnectrl+ 2
# CHECK-BE: bnectrl+ 0                      # encoding: [0x4c,0xe2,0x04,0x21]
# CHECK-LE: bnectrl+ 0                      # encoding: [0x21,0x04,0xe2,0x4c]
            bnectrl+

# CHECK-BE: bne- 2, target                  # encoding: [0x40,0xca,A,0bAAAAAA00]
# CHECK-LE: bne- 2, target                  # encoding: [0bAAAAAA00,A,0xca,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bne- 2, target
# CHECK-BE: bne- 0, target                  # encoding: [0x40,0xc2,A,0bAAAAAA00]
# CHECK-LE: bne- 0, target                  # encoding: [0bAAAAAA00,A,0xc2,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bne- target
# CHECK-BE: bnea- 2, target                 # encoding: [0x40,0xca,A,0bAAAAAA10]
# CHECK-LE: bnea- 2, target                 # encoding: [0bAAAAAA10,A,0xca,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnea- 2, target
# CHECK-BE: bnea- 0, target                 # encoding: [0x40,0xc2,A,0bAAAAAA10]
# CHECK-LE: bnea- 0, target                 # encoding: [0bAAAAAA10,A,0xc2,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnea- target
# CHECK-BE: bnelr- 2                        # encoding: [0x4c,0xca,0x00,0x20]
# CHECK-LE: bnelr- 2                        # encoding: [0x20,0x00,0xca,0x4c]
            bnelr- 2
# CHECK-BE: bnelr- 0                        # encoding: [0x4c,0xc2,0x00,0x20]
# CHECK-LE: bnelr- 0                        # encoding: [0x20,0x00,0xc2,0x4c]
            bnelr-
# CHECK-BE: bnectr- 2                       # encoding: [0x4c,0xca,0x04,0x20]
# CHECK-LE: bnectr- 2                       # encoding: [0x20,0x04,0xca,0x4c]
            bnectr- 2
# CHECK-BE: bnectr- 0                       # encoding: [0x4c,0xc2,0x04,0x20]
# CHECK-LE: bnectr- 0                       # encoding: [0x20,0x04,0xc2,0x4c]
            bnectr-
# CHECK-BE: bnel- 2, target                 # encoding: [0x40,0xca,A,0bAAAAAA01]
# CHECK-LE: bnel- 2, target                 # encoding: [0bAAAAAA01,A,0xca,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnel- 2, target
# CHECK-BE: bnel- 0, target                 # encoding: [0x40,0xc2,A,0bAAAAAA01]
# CHECK-LE: bnel- 0, target                 # encoding: [0bAAAAAA01,A,0xc2,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnel- target
# CHECK-BE: bnela- 2, target                # encoding: [0x40,0xca,A,0bAAAAAA11]
# CHECK-LE: bnela- 2, target                # encoding: [0bAAAAAA11,A,0xca,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnela- 2, target
# CHECK-BE: bnela- 0, target                # encoding: [0x40,0xc2,A,0bAAAAAA11]
# CHECK-LE: bnela- 0, target                # encoding: [0bAAAAAA11,A,0xc2,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnela- target
# CHECK-BE: bnelrl- 2                       # encoding: [0x4c,0xca,0x00,0x21]
# CHECK-LE: bnelrl- 2                       # encoding: [0x21,0x00,0xca,0x4c]
            bnelrl- 2
# CHECK-BE: bnelrl- 0                       # encoding: [0x4c,0xc2,0x00,0x21]
# CHECK-LE: bnelrl- 0                       # encoding: [0x21,0x00,0xc2,0x4c]
            bnelrl-
# CHECK-BE: bnectrl- 2                      # encoding: [0x4c,0xca,0x04,0x21]
# CHECK-LE: bnectrl- 2                      # encoding: [0x21,0x04,0xca,0x4c]
            bnectrl- 2
# CHECK-BE: bnectrl- 0                      # encoding: [0x4c,0xc2,0x04,0x21]
# CHECK-LE: bnectrl- 0                      # encoding: [0x21,0x04,0xc2,0x4c]
            bnectrl-

# CHECK-BE: ble 2, target                   # encoding: [0x40,0x89,A,0bAAAAAA00]
# CHECK-LE: ble 2, target                   # encoding: [0bAAAAAA00,A,0x89,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bng 2, target
# CHECK-BE: ble 0, target                   # encoding: [0x40,0x81,A,0bAAAAAA00]
# CHECK-LE: ble 0, target                   # encoding: [0bAAAAAA00,A,0x81,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bng target
# CHECK-BE: blea 2, target                  # encoding: [0x40,0x89,A,0bAAAAAA10]
# CHECK-LE: blea 2, target                  # encoding: [0bAAAAAA10,A,0x89,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnga 2, target
# CHECK-BE: blea 0, target                  # encoding: [0x40,0x81,A,0bAAAAAA10]
# CHECK-LE: blea 0, target                  # encoding: [0bAAAAAA10,A,0x81,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnga target
# CHECK-BE: blelr 2                         # encoding: [0x4c,0x89,0x00,0x20]
# CHECK-LE: blelr 2                         # encoding: [0x20,0x00,0x89,0x4c]
            bnglr 2
# CHECK-BE: blelr 0                         # encoding: [0x4c,0x81,0x00,0x20]
# CHECK-LE: blelr 0                         # encoding: [0x20,0x00,0x81,0x4c]
            bnglr
# CHECK-BE: blectr 2                        # encoding: [0x4c,0x89,0x04,0x20]
# CHECK-LE: blectr 2                        # encoding: [0x20,0x04,0x89,0x4c]
            bngctr 2
# CHECK-BE: blectr 0                        # encoding: [0x4c,0x81,0x04,0x20]
# CHECK-LE: blectr 0                        # encoding: [0x20,0x04,0x81,0x4c]
            bngctr
# CHECK-BE: blel 2, target                  # encoding: [0x40,0x89,A,0bAAAAAA01]
# CHECK-LE: blel 2, target                  # encoding: [0bAAAAAA01,A,0x89,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bngl 2, target
# CHECK-BE: blel 0, target                  # encoding: [0x40,0x81,A,0bAAAAAA01]
# CHECK-LE: blel 0, target                  # encoding: [0bAAAAAA01,A,0x81,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bngl target
# CHECK-BE: blela 2, target                 # encoding: [0x40,0x89,A,0bAAAAAA11]
# CHECK-LE: blela 2, target                 # encoding: [0bAAAAAA11,A,0x89,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bngla 2, target
# CHECK-BE: blela 0, target                 # encoding: [0x40,0x81,A,0bAAAAAA11]
# CHECK-LE: blela 0, target                 # encoding: [0bAAAAAA11,A,0x81,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bngla target
# CHECK-BE: blelrl 2                        # encoding: [0x4c,0x89,0x00,0x21]
# CHECK-LE: blelrl 2                        # encoding: [0x21,0x00,0x89,0x4c]
            bnglrl 2
# CHECK-BE: blelrl 0                        # encoding: [0x4c,0x81,0x00,0x21]
# CHECK-LE: blelrl 0                        # encoding: [0x21,0x00,0x81,0x4c]
            bnglrl
# CHECK-BE: blectrl 2                       # encoding: [0x4c,0x89,0x04,0x21]
# CHECK-LE: blectrl 2                       # encoding: [0x21,0x04,0x89,0x4c]
            bngctrl 2
# CHECK-BE: blectrl 0                       # encoding: [0x4c,0x81,0x04,0x21]
# CHECK-LE: blectrl 0                       # encoding: [0x21,0x04,0x81,0x4c]
            bngctrl

# CHECK-BE: ble+ 2, target                   # encoding: [0x40,0xe9,A,0bAAAAAA00]
# CHECK-LE: ble+ 2, target                   # encoding: [0bAAAAAA00,A,0xe9,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bng+ 2, target
# CHECK-BE: ble+ 0, target                   # encoding: [0x40,0xe1,A,0bAAAAAA00]
# CHECK-LE: ble+ 0, target                   # encoding: [0bAAAAAA00,A,0xe1,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bng+ target
# CHECK-BE: blea+ 2, target                  # encoding: [0x40,0xe9,A,0bAAAAAA10]
# CHECK-LE: blea+ 2, target                  # encoding: [0bAAAAAA10,A,0xe9,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnga+ 2, target
# CHECK-BE: blea+ 0, target                  # encoding: [0x40,0xe1,A,0bAAAAAA10]
# CHECK-LE: blea+ 0, target                  # encoding: [0bAAAAAA10,A,0xe1,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnga+ target
# CHECK-BE: blelr+ 2                         # encoding: [0x4c,0xe9,0x00,0x20]
# CHECK-LE: blelr+ 2                         # encoding: [0x20,0x00,0xe9,0x4c]
            bnglr+ 2
# CHECK-BE: blelr+ 0                         # encoding: [0x4c,0xe1,0x00,0x20]
# CHECK-LE: blelr+ 0                         # encoding: [0x20,0x00,0xe1,0x4c]
            bnglr+
# CHECK-BE: blectr+ 2                        # encoding: [0x4c,0xe9,0x04,0x20]
# CHECK-LE: blectr+ 2                        # encoding: [0x20,0x04,0xe9,0x4c]
            bngctr+ 2
# CHECK-BE: blectr+ 0                        # encoding: [0x4c,0xe1,0x04,0x20]
# CHECK-LE: blectr+ 0                        # encoding: [0x20,0x04,0xe1,0x4c]
            bngctr+
# CHECK-BE: blel+ 2, target                  # encoding: [0x40,0xe9,A,0bAAAAAA01]
# CHECK-LE: blel+ 2, target                  # encoding: [0bAAAAAA01,A,0xe9,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bngl+ 2, target
# CHECK-BE: blel+ 0, target                  # encoding: [0x40,0xe1,A,0bAAAAAA01]
# CHECK-LE: blel+ 0, target                  # encoding: [0bAAAAAA01,A,0xe1,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bngl+ target
# CHECK-BE: blela+ 2, target                 # encoding: [0x40,0xe9,A,0bAAAAAA11]
# CHECK-LE: blela+ 2, target                 # encoding: [0bAAAAAA11,A,0xe9,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bngla+ 2, target
# CHECK-BE: blela+ 0, target                 # encoding: [0x40,0xe1,A,0bAAAAAA11]
# CHECK-LE: blela+ 0, target                 # encoding: [0bAAAAAA11,A,0xe1,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bngla+ target
# CHECK-BE: blelrl+ 2                        # encoding: [0x4c,0xe9,0x00,0x21]
# CHECK-LE: blelrl+ 2                        # encoding: [0x21,0x00,0xe9,0x4c]
            bnglrl+ 2
# CHECK-BE: blelrl+ 0                        # encoding: [0x4c,0xe1,0x00,0x21]
# CHECK-LE: blelrl+ 0                        # encoding: [0x21,0x00,0xe1,0x4c]
            bnglrl+
# CHECK-BE: blectrl+ 2                       # encoding: [0x4c,0xe9,0x04,0x21]
# CHECK-LE: blectrl+ 2                       # encoding: [0x21,0x04,0xe9,0x4c]
            bngctrl+ 2
# CHECK-BE: blectrl+ 0                       # encoding: [0x4c,0xe1,0x04,0x21]
# CHECK-LE: blectrl+ 0                       # encoding: [0x21,0x04,0xe1,0x4c]
            bngctrl+

# CHECK-BE: ble- 2, target                   # encoding: [0x40,0xc9,A,0bAAAAAA00]
# CHECK-LE: ble- 2, target                   # encoding: [0bAAAAAA00,A,0xc9,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bng- 2, target
# CHECK-BE: ble- 0, target                   # encoding: [0x40,0xc1,A,0bAAAAAA00]
# CHECK-LE: ble- 0, target                   # encoding: [0bAAAAAA00,A,0xc1,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bng- target
# CHECK-BE: blea- 2, target                  # encoding: [0x40,0xc9,A,0bAAAAAA10]
# CHECK-LE: blea- 2, target                  # encoding: [0bAAAAAA10,A,0xc9,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnga- 2, target
# CHECK-BE: blea- 0, target                  # encoding: [0x40,0xc1,A,0bAAAAAA10]
# CHECK-LE: blea- 0, target                  # encoding: [0bAAAAAA10,A,0xc1,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnga- target
# CHECK-BE: blelr- 2                         # encoding: [0x4c,0xc9,0x00,0x20]
# CHECK-LE: blelr- 2                         # encoding: [0x20,0x00,0xc9,0x4c]
            bnglr- 2
# CHECK-BE: blelr- 0                         # encoding: [0x4c,0xc1,0x00,0x20]
# CHECK-LE: blelr- 0                         # encoding: [0x20,0x00,0xc1,0x4c]
            bnglr-
# CHECK-BE: blectr- 2                        # encoding: [0x4c,0xc9,0x04,0x20]
# CHECK-LE: blectr- 2                        # encoding: [0x20,0x04,0xc9,0x4c]
            bngctr- 2
# CHECK-BE: blectr- 0                        # encoding: [0x4c,0xc1,0x04,0x20]
# CHECK-LE: blectr- 0                        # encoding: [0x20,0x04,0xc1,0x4c]
            bngctr-
# CHECK-BE: blel- 2, target                  # encoding: [0x40,0xc9,A,0bAAAAAA01]
# CHECK-LE: blel- 2, target                  # encoding: [0bAAAAAA01,A,0xc9,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bngl- 2, target
# CHECK-BE: blel- 0, target                  # encoding: [0x40,0xc1,A,0bAAAAAA01]
# CHECK-LE: blel- 0, target                  # encoding: [0bAAAAAA01,A,0xc1,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bngl- target
# CHECK-BE: blela- 2, target                 # encoding: [0x40,0xc9,A,0bAAAAAA11]
# CHECK-LE: blela- 2, target                 # encoding: [0bAAAAAA11,A,0xc9,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bngla- 2, target
# CHECK-BE: blela- 0, target                 # encoding: [0x40,0xc1,A,0bAAAAAA11]
# CHECK-LE: blela- 0, target                 # encoding: [0bAAAAAA11,A,0xc1,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bngla- target
# CHECK-BE: blelrl- 2                        # encoding: [0x4c,0xc9,0x00,0x21]
# CHECK-LE: blelrl- 2                        # encoding: [0x21,0x00,0xc9,0x4c]
            bnglrl- 2
# CHECK-BE: blelrl- 0                        # encoding: [0x4c,0xc1,0x00,0x21]
# CHECK-LE: blelrl- 0                        # encoding: [0x21,0x00,0xc1,0x4c]
            bnglrl-
# CHECK-BE: blectrl- 2                       # encoding: [0x4c,0xc9,0x04,0x21]
# CHECK-LE: blectrl- 2                       # encoding: [0x21,0x04,0xc9,0x4c]
            bngctrl- 2
# CHECK-BE: blectrl- 0                       # encoding: [0x4c,0xc1,0x04,0x21]
# CHECK-LE: blectrl- 0                       # encoding: [0x21,0x04,0xc1,0x4c]
            bngctrl-

# CHECK-BE: bun 2, target                   # encoding: [0x41,0x8b,A,0bAAAAAA00]
# CHECK-LE: bun 2, target                   # encoding: [0bAAAAAA00,A,0x8b,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bso 2, target
# CHECK-BE: bun 0, target                   # encoding: [0x41,0x83,A,0bAAAAAA00]
# CHECK-LE: bun 0, target                   # encoding: [0bAAAAAA00,A,0x83,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bso target
# CHECK-BE: buna 2, target                  # encoding: [0x41,0x8b,A,0bAAAAAA10]
# CHECK-LE: buna 2, target                  # encoding: [0bAAAAAA10,A,0x8b,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bsoa 2, target
# CHECK-BE: buna 0, target                  # encoding: [0x41,0x83,A,0bAAAAAA10]
# CHECK-LE: buna 0, target                  # encoding: [0bAAAAAA10,A,0x83,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bsoa target
# CHECK-BE: bunlr 2                         # encoding: [0x4d,0x8b,0x00,0x20]
# CHECK-LE: bunlr 2                         # encoding: [0x20,0x00,0x8b,0x4d]
            bsolr 2
# CHECK-BE: bunlr 0                         # encoding: [0x4d,0x83,0x00,0x20]
# CHECK-LE: bunlr 0                         # encoding: [0x20,0x00,0x83,0x4d]
            bsolr
# CHECK-BE: bunctr 2                        # encoding: [0x4d,0x8b,0x04,0x20]
# CHECK-LE: bunctr 2                        # encoding: [0x20,0x04,0x8b,0x4d]
            bsoctr 2
# CHECK-BE: bunctr 0                        # encoding: [0x4d,0x83,0x04,0x20]
# CHECK-LE: bunctr 0                        # encoding: [0x20,0x04,0x83,0x4d]
            bsoctr
# CHECK-BE: bunl 2, target                  # encoding: [0x41,0x8b,A,0bAAAAAA01]
# CHECK-LE: bunl 2, target                  # encoding: [0bAAAAAA01,A,0x8b,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bsol 2, target
# CHECK-BE: bunl 0, target                  # encoding: [0x41,0x83,A,0bAAAAAA01]
# CHECK-LE: bunl 0, target                  # encoding: [0bAAAAAA01,A,0x83,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bsol target
# CHECK-BE: bunla 2, target                 # encoding: [0x41,0x8b,A,0bAAAAAA11]
# CHECK-LE: bunla 2, target                 # encoding: [0bAAAAAA11,A,0x8b,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bsola 2, target
# CHECK-BE: bunla 0, target                 # encoding: [0x41,0x83,A,0bAAAAAA11]
# CHECK-LE: bunla 0, target                 # encoding: [0bAAAAAA11,A,0x83,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bsola target
# CHECK-BE: bunlrl 2                        # encoding: [0x4d,0x8b,0x00,0x21]
# CHECK-LE: bunlrl 2                        # encoding: [0x21,0x00,0x8b,0x4d]
            bsolrl 2
# CHECK-BE: bunlrl 0                        # encoding: [0x4d,0x83,0x00,0x21]
# CHECK-LE: bunlrl 0                        # encoding: [0x21,0x00,0x83,0x4d]
            bsolrl
# CHECK-BE: bunctrl 2                       # encoding: [0x4d,0x8b,0x04,0x21]
# CHECK-LE: bunctrl 2                       # encoding: [0x21,0x04,0x8b,0x4d]
            bsoctrl 2
# CHECK-BE: bunctrl 0                       # encoding: [0x4d,0x83,0x04,0x21]
# CHECK-LE: bunctrl 0                       # encoding: [0x21,0x04,0x83,0x4d]
            bsoctrl

# CHECK-BE: bun+ 2, target                  # encoding: [0x41,0xeb,A,0bAAAAAA00]
# CHECK-LE: bun+ 2, target                  # encoding: [0bAAAAAA00,A,0xeb,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bso+ 2, target
# CHECK-BE: bun+ 0, target                  # encoding: [0x41,0xe3,A,0bAAAAAA00]
# CHECK-LE: bun+ 0, target                  # encoding: [0bAAAAAA00,A,0xe3,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bso+ target
# CHECK-BE: buna+ 2, target                 # encoding: [0x41,0xeb,A,0bAAAAAA10]
# CHECK-LE: buna+ 2, target                 # encoding: [0bAAAAAA10,A,0xeb,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bsoa+ 2, target
# CHECK-BE: buna+ 0, target                 # encoding: [0x41,0xe3,A,0bAAAAAA10]
# CHECK-LE: buna+ 0, target                 # encoding: [0bAAAAAA10,A,0xe3,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bsoa+ target
# CHECK-BE: bunlr+ 2                        # encoding: [0x4d,0xeb,0x00,0x20]
# CHECK-LE: bunlr+ 2                        # encoding: [0x20,0x00,0xeb,0x4d]
            bsolr+ 2
# CHECK-BE: bunlr+ 0                        # encoding: [0x4d,0xe3,0x00,0x20]
# CHECK-LE: bunlr+ 0                        # encoding: [0x20,0x00,0xe3,0x4d]
            bsolr+
# CHECK-BE: bunctr+ 2                       # encoding: [0x4d,0xeb,0x04,0x20]
# CHECK-LE: bunctr+ 2                       # encoding: [0x20,0x04,0xeb,0x4d]
            bsoctr+ 2
# CHECK-BE: bunctr+ 0                       # encoding: [0x4d,0xe3,0x04,0x20]
# CHECK-LE: bunctr+ 0                       # encoding: [0x20,0x04,0xe3,0x4d]
            bsoctr+
# CHECK-BE: bunl+ 2, target                 # encoding: [0x41,0xeb,A,0bAAAAAA01]
# CHECK-LE: bunl+ 2, target                 # encoding: [0bAAAAAA01,A,0xeb,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bsol+ 2, target
# CHECK-BE: bunl+ 0, target                 # encoding: [0x41,0xe3,A,0bAAAAAA01]
# CHECK-LE: bunl+ 0, target                 # encoding: [0bAAAAAA01,A,0xe3,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bsol+ target
# CHECK-BE: bunla+ 2, target                # encoding: [0x41,0xeb,A,0bAAAAAA11]
# CHECK-LE: bunla+ 2, target                # encoding: [0bAAAAAA11,A,0xeb,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bsola+ 2, target
# CHECK-BE: bunla+ 0, target                # encoding: [0x41,0xe3,A,0bAAAAAA11]
# CHECK-LE: bunla+ 0, target                # encoding: [0bAAAAAA11,A,0xe3,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bsola+ target
# CHECK-BE: bunlrl+ 2                       # encoding: [0x4d,0xeb,0x00,0x21]
# CHECK-LE: bunlrl+ 2                       # encoding: [0x21,0x00,0xeb,0x4d]
            bsolrl+ 2
# CHECK-BE: bunlrl+ 0                       # encoding: [0x4d,0xe3,0x00,0x21]
# CHECK-LE: bunlrl+ 0                       # encoding: [0x21,0x00,0xe3,0x4d]
            bsolrl+
# CHECK-BE: bunctrl+ 2                      # encoding: [0x4d,0xeb,0x04,0x21]
# CHECK-LE: bunctrl+ 2                      # encoding: [0x21,0x04,0xeb,0x4d]
            bsoctrl+ 2
# CHECK-BE: bunctrl+ 0                      # encoding: [0x4d,0xe3,0x04,0x21]
# CHECK-LE: bunctrl+ 0                      # encoding: [0x21,0x04,0xe3,0x4d]
            bsoctrl+

# CHECK-BE: bun- 2, target                  # encoding: [0x41,0xcb,A,0bAAAAAA00]
# CHECK-LE: bun- 2, target                  # encoding: [0bAAAAAA00,A,0xcb,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bso- 2, target
# CHECK-BE: bun- 0, target                  # encoding: [0x41,0xc3,A,0bAAAAAA00]
# CHECK-LE: bun- 0, target                  # encoding: [0bAAAAAA00,A,0xc3,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bso- target
# CHECK-BE: buna- 2, target                 # encoding: [0x41,0xcb,A,0bAAAAAA10]
# CHECK-LE: buna- 2, target                 # encoding: [0bAAAAAA10,A,0xcb,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bsoa- 2, target
# CHECK-BE: buna- 0, target                 # encoding: [0x41,0xc3,A,0bAAAAAA10]
# CHECK-LE: buna- 0, target                 # encoding: [0bAAAAAA10,A,0xc3,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bsoa- target
# CHECK-BE: bunlr- 2                        # encoding: [0x4d,0xcb,0x00,0x20]
# CHECK-LE: bunlr- 2                        # encoding: [0x20,0x00,0xcb,0x4d]
            bsolr- 2
# CHECK-BE: bunlr- 0                        # encoding: [0x4d,0xc3,0x00,0x20]
# CHECK-LE: bunlr- 0                        # encoding: [0x20,0x00,0xc3,0x4d]
            bsolr-
# CHECK-BE: bunctr- 2                       # encoding: [0x4d,0xcb,0x04,0x20]
# CHECK-LE: bunctr- 2                       # encoding: [0x20,0x04,0xcb,0x4d]
            bsoctr- 2
# CHECK-BE: bunctr- 0                       # encoding: [0x4d,0xc3,0x04,0x20]
# CHECK-LE: bunctr- 0                       # encoding: [0x20,0x04,0xc3,0x4d]
            bsoctr-
# CHECK-BE: bunl- 2, target                 # encoding: [0x41,0xcb,A,0bAAAAAA01]
# CHECK-LE: bunl- 2, target                 # encoding: [0bAAAAAA01,A,0xcb,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bsol- 2, target
# CHECK-BE: bunl- 0, target                 # encoding: [0x41,0xc3,A,0bAAAAAA01]
# CHECK-LE: bunl- 0, target                 # encoding: [0bAAAAAA01,A,0xc3,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bsol- target
# CHECK-BE: bunla- 2, target                # encoding: [0x41,0xcb,A,0bAAAAAA11]
# CHECK-LE: bunla- 2, target                # encoding: [0bAAAAAA11,A,0xcb,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bsola- 2, target
# CHECK-BE: bunla- 0, target                # encoding: [0x41,0xc3,A,0bAAAAAA11]
# CHECK-LE: bunla- 0, target                # encoding: [0bAAAAAA11,A,0xc3,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bsola- target
# CHECK-BE: bunlrl- 2                       # encoding: [0x4d,0xcb,0x00,0x21]
# CHECK-LE: bunlrl- 2                       # encoding: [0x21,0x00,0xcb,0x4d]
            bsolrl- 2
# CHECK-BE: bunlrl- 0                       # encoding: [0x4d,0xc3,0x00,0x21]
# CHECK-LE: bunlrl- 0                       # encoding: [0x21,0x00,0xc3,0x4d]
            bsolrl-
# CHECK-BE: bunctrl- 2                      # encoding: [0x4d,0xcb,0x04,0x21]
# CHECK-LE: bunctrl- 2                      # encoding: [0x21,0x04,0xcb,0x4d]
            bsoctrl- 2
# CHECK-BE: bunctrl- 0                      # encoding: [0x4d,0xc3,0x04,0x21]
# CHECK-LE: bunctrl- 0                      # encoding: [0x21,0x04,0xc3,0x4d]
            bsoctrl-

# CHECK-BE: bnu 2, target                   # encoding: [0x40,0x8b,A,0bAAAAAA00]
# CHECK-LE: bnu 2, target                   # encoding: [0bAAAAAA00,A,0x8b,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bns 2, target
# CHECK-BE: bnu 0, target                   # encoding: [0x40,0x83,A,0bAAAAAA00]
# CHECK-LE: bnu 0, target                   # encoding: [0bAAAAAA00,A,0x83,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bns target
# CHECK-BE: bnua 2, target                  # encoding: [0x40,0x8b,A,0bAAAAAA10]
# CHECK-LE: bnua 2, target                  # encoding: [0bAAAAAA10,A,0x8b,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnsa 2, target
# CHECK-BE: bnua 0, target                  # encoding: [0x40,0x83,A,0bAAAAAA10]
# CHECK-LE: bnua 0, target                  # encoding: [0bAAAAAA10,A,0x83,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnsa target
# CHECK-BE: bnulr 2                         # encoding: [0x4c,0x8b,0x00,0x20]
# CHECK-LE: bnulr 2                         # encoding: [0x20,0x00,0x8b,0x4c]
            bnslr 2
# CHECK-BE: bnulr 0                         # encoding: [0x4c,0x83,0x00,0x20]
# CHECK-LE: bnulr 0                         # encoding: [0x20,0x00,0x83,0x4c]
            bnslr
# CHECK-BE: bnuctr 2                        # encoding: [0x4c,0x8b,0x04,0x20]
# CHECK-LE: bnuctr 2                        # encoding: [0x20,0x04,0x8b,0x4c]
            bnsctr 2
# CHECK-BE: bnuctr 0                        # encoding: [0x4c,0x83,0x04,0x20]
# CHECK-LE: bnuctr 0                        # encoding: [0x20,0x04,0x83,0x4c]
            bnsctr
# CHECK-BE: bnul 2, target                  # encoding: [0x40,0x8b,A,0bAAAAAA01]
# CHECK-LE: bnul 2, target                  # encoding: [0bAAAAAA01,A,0x8b,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnsl 2, target
# CHECK-BE: bnul 0, target                  # encoding: [0x40,0x83,A,0bAAAAAA01]
# CHECK-LE: bnul 0, target                  # encoding: [0bAAAAAA01,A,0x83,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnsl target
# CHECK-BE: bnula 2, target                 # encoding: [0x40,0x8b,A,0bAAAAAA11]
# CHECK-LE: bnula 2, target                 # encoding: [0bAAAAAA11,A,0x8b,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnsla 2, target
# CHECK-BE: bnula 0, target                 # encoding: [0x40,0x83,A,0bAAAAAA11]
# CHECK-LE: bnula 0, target                 # encoding: [0bAAAAAA11,A,0x83,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnsla target
# CHECK-BE: bnulrl 2                        # encoding: [0x4c,0x8b,0x00,0x21]
# CHECK-LE: bnulrl 2                        # encoding: [0x21,0x00,0x8b,0x4c]
            bnslrl 2
# CHECK-BE: bnulrl 0                        # encoding: [0x4c,0x83,0x00,0x21]
# CHECK-LE: bnulrl 0                        # encoding: [0x21,0x00,0x83,0x4c]
            bnslrl
# CHECK-BE: bnuctrl 2                       # encoding: [0x4c,0x8b,0x04,0x21]
# CHECK-LE: bnuctrl 2                       # encoding: [0x21,0x04,0x8b,0x4c]
            bnsctrl 2
# CHECK-BE: bnuctrl 0                       # encoding: [0x4c,0x83,0x04,0x21]
# CHECK-LE: bnuctrl 0                       # encoding: [0x21,0x04,0x83,0x4c]
            bnsctrl

# CHECK-BE: bnu+ 2, target                  # encoding: [0x40,0xeb,A,0bAAAAAA00]
# CHECK-LE: bnu+ 2, target                  # encoding: [0bAAAAAA00,A,0xeb,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bns+ 2, target
# CHECK-BE: bnu+ 0, target                  # encoding: [0x40,0xe3,A,0bAAAAAA00]
# CHECK-LE: bnu+ 0, target                  # encoding: [0bAAAAAA00,A,0xe3,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bns+ target
# CHECK-BE: bnua+ 2, target                 # encoding: [0x40,0xeb,A,0bAAAAAA10]
# CHECK-LE: bnua+ 2, target                 # encoding: [0bAAAAAA10,A,0xeb,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnsa+ 2, target
# CHECK-BE: bnua+ 0, target                 # encoding: [0x40,0xe3,A,0bAAAAAA10]
# CHECK-LE: bnua+ 0, target                 # encoding: [0bAAAAAA10,A,0xe3,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnsa+ target
# CHECK-BE: bnulr+ 2                        # encoding: [0x4c,0xeb,0x00,0x20]
# CHECK-LE: bnulr+ 2                        # encoding: [0x20,0x00,0xeb,0x4c]
            bnslr+ 2
# CHECK-BE: bnulr+ 0                        # encoding: [0x4c,0xe3,0x00,0x20]
# CHECK-LE: bnulr+ 0                        # encoding: [0x20,0x00,0xe3,0x4c]
            bnslr+
# CHECK-BE: bnuctr+ 2                       # encoding: [0x4c,0xeb,0x04,0x20]
# CHECK-LE: bnuctr+ 2                       # encoding: [0x20,0x04,0xeb,0x4c]
            bnsctr+ 2
# CHECK-BE: bnuctr+ 0                       # encoding: [0x4c,0xe3,0x04,0x20]
# CHECK-LE: bnuctr+ 0                       # encoding: [0x20,0x04,0xe3,0x4c]
            bnsctr+
# CHECK-BE: bnul+ 2, target                 # encoding: [0x40,0xeb,A,0bAAAAAA01]
# CHECK-LE: bnul+ 2, target                 # encoding: [0bAAAAAA01,A,0xeb,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnsl+ 2, target
# CHECK-BE: bnul+ 0, target                 # encoding: [0x40,0xe3,A,0bAAAAAA01]
# CHECK-LE: bnul+ 0, target                 # encoding: [0bAAAAAA01,A,0xe3,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnsl+ target
# CHECK-BE: bnula+ 2, target                # encoding: [0x40,0xeb,A,0bAAAAAA11]
# CHECK-LE: bnula+ 2, target                # encoding: [0bAAAAAA11,A,0xeb,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnsla+ 2, target
# CHECK-BE: bnula+ 0, target                # encoding: [0x40,0xe3,A,0bAAAAAA11]
# CHECK-LE: bnula+ 0, target                # encoding: [0bAAAAAA11,A,0xe3,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnsla+ target
# CHECK-BE: bnulrl+ 2                       # encoding: [0x4c,0xeb,0x00,0x21]
# CHECK-LE: bnulrl+ 2                       # encoding: [0x21,0x00,0xeb,0x4c]
            bnslrl+ 2
# CHECK-BE: bnulrl+ 0                       # encoding: [0x4c,0xe3,0x00,0x21]
# CHECK-LE: bnulrl+ 0                       # encoding: [0x21,0x00,0xe3,0x4c]
            bnslrl+
# CHECK-BE: bnuctrl+ 2                      # encoding: [0x4c,0xeb,0x04,0x21]
# CHECK-LE: bnuctrl+ 2                      # encoding: [0x21,0x04,0xeb,0x4c]
            bnsctrl+ 2
# CHECK-BE: bnuctrl+ 0                      # encoding: [0x4c,0xe3,0x04,0x21]
# CHECK-LE: bnuctrl+ 0                      # encoding: [0x21,0x04,0xe3,0x4c]
            bnsctrl+

# CHECK-BE: bnu- 2, target                  # encoding: [0x40,0xcb,A,0bAAAAAA00]
# CHECK-LE: bnu- 2, target                  # encoding: [0bAAAAAA00,A,0xcb,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bns- 2, target
# CHECK-BE: bnu- 0, target                  # encoding: [0x40,0xc3,A,0bAAAAAA00]
# CHECK-LE: bnu- 0, target                  # encoding: [0bAAAAAA00,A,0xc3,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bns- target
# CHECK-BE: bnua- 2, target                 # encoding: [0x40,0xcb,A,0bAAAAAA10]
# CHECK-LE: bnua- 2, target                 # encoding: [0bAAAAAA10,A,0xcb,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnsa- 2, target
# CHECK-BE: bnua- 0, target                 # encoding: [0x40,0xc3,A,0bAAAAAA10]
# CHECK-LE: bnua- 0, target                 # encoding: [0bAAAAAA10,A,0xc3,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnsa- target
# CHECK-BE: bnulr- 2                        # encoding: [0x4c,0xcb,0x00,0x20]
# CHECK-LE: bnulr- 2                        # encoding: [0x20,0x00,0xcb,0x4c]
            bnslr- 2
# CHECK-BE: bnulr- 0                        # encoding: [0x4c,0xc3,0x00,0x20]
# CHECK-LE: bnulr- 0                        # encoding: [0x20,0x00,0xc3,0x4c]
            bnslr-
# CHECK-BE: bnuctr- 2                       # encoding: [0x4c,0xcb,0x04,0x20]
# CHECK-LE: bnuctr- 2                       # encoding: [0x20,0x04,0xcb,0x4c]
            bnsctr- 2
# CHECK-BE: bnuctr- 0                       # encoding: [0x4c,0xc3,0x04,0x20]
# CHECK-LE: bnuctr- 0                       # encoding: [0x20,0x04,0xc3,0x4c]
            bnsctr-
# CHECK-BE: bnul- 2, target                 # encoding: [0x40,0xcb,A,0bAAAAAA01]
# CHECK-LE: bnul- 2, target                 # encoding: [0bAAAAAA01,A,0xcb,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnsl- 2, target
# CHECK-BE: bnul- 0, target                 # encoding: [0x40,0xc3,A,0bAAAAAA01]
# CHECK-LE: bnul- 0, target                 # encoding: [0bAAAAAA01,A,0xc3,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnsl- target
# CHECK-BE: bnula- 2, target                # encoding: [0x40,0xcb,A,0bAAAAAA11]
# CHECK-LE: bnula- 2, target                # encoding: [0bAAAAAA11,A,0xcb,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnsla- 2, target
# CHECK-BE: bnula- 0, target                # encoding: [0x40,0xc3,A,0bAAAAAA11]
# CHECK-LE: bnula- 0, target                # encoding: [0bAAAAAA11,A,0xc3,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnsla- target
# CHECK-BE: bnulrl- 2                       # encoding: [0x4c,0xcb,0x00,0x21]
# CHECK-LE: bnulrl- 2                       # encoding: [0x21,0x00,0xcb,0x4c]
            bnslrl- 2
# CHECK-BE: bnulrl- 0                       # encoding: [0x4c,0xc3,0x00,0x21]
# CHECK-LE: bnulrl- 0                       # encoding: [0x21,0x00,0xc3,0x4c]
            bnslrl-
# CHECK-BE: bnuctrl- 2                      # encoding: [0x4c,0xcb,0x04,0x21]
# CHECK-LE: bnuctrl- 2                      # encoding: [0x21,0x04,0xcb,0x4c]
            bnsctrl- 2
# CHECK-BE: bnuctrl- 0                      # encoding: [0x4c,0xc3,0x04,0x21]
# CHECK-LE: bnuctrl- 0                      # encoding: [0x21,0x04,0xc3,0x4c]
            bnsctrl-

# CHECK-BE: bun 2, target                   # encoding: [0x41,0x8b,A,0bAAAAAA00]
# CHECK-LE: bun 2, target                   # encoding: [0bAAAAAA00,A,0x8b,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bun 2, target
# CHECK-BE: bun 0, target                   # encoding: [0x41,0x83,A,0bAAAAAA00]
# CHECK-LE: bun 0, target                   # encoding: [0bAAAAAA00,A,0x83,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bun target
# CHECK-BE: buna 2, target                  # encoding: [0x41,0x8b,A,0bAAAAAA10]
# CHECK-LE: buna 2, target                  # encoding: [0bAAAAAA10,A,0x8b,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            buna 2, target
# CHECK-BE: buna 0, target                  # encoding: [0x41,0x83,A,0bAAAAAA10]
# CHECK-LE: buna 0, target                  # encoding: [0bAAAAAA10,A,0x83,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            buna target
# CHECK-BE: bunlr 2                         # encoding: [0x4d,0x8b,0x00,0x20]
# CHECK-LE: bunlr 2                         # encoding: [0x20,0x00,0x8b,0x4d]
            bunlr 2
# CHECK-BE: bunlr 0                         # encoding: [0x4d,0x83,0x00,0x20]
# CHECK-LE: bunlr 0                         # encoding: [0x20,0x00,0x83,0x4d]
            bunlr
# CHECK-BE: bunctr 2                        # encoding: [0x4d,0x8b,0x04,0x20]
# CHECK-LE: bunctr 2                        # encoding: [0x20,0x04,0x8b,0x4d]
            bunctr 2
# CHECK-BE: bunctr 0                        # encoding: [0x4d,0x83,0x04,0x20]
# CHECK-LE: bunctr 0                        # encoding: [0x20,0x04,0x83,0x4d]
            bunctr
# CHECK-BE: bunl 2, target                  # encoding: [0x41,0x8b,A,0bAAAAAA01]
# CHECK-LE: bunl 2, target                  # encoding: [0bAAAAAA01,A,0x8b,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bunl 2, target
# CHECK-BE: bunl 0, target                  # encoding: [0x41,0x83,A,0bAAAAAA01]
# CHECK-LE: bunl 0, target                  # encoding: [0bAAAAAA01,A,0x83,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bunl target
# CHECK-BE: bunla 2, target                 # encoding: [0x41,0x8b,A,0bAAAAAA11]
# CHECK-LE: bunla 2, target                 # encoding: [0bAAAAAA11,A,0x8b,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bunla 2, target
# CHECK-BE: bunla 0, target                 # encoding: [0x41,0x83,A,0bAAAAAA11]
# CHECK-LE: bunla 0, target                 # encoding: [0bAAAAAA11,A,0x83,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bunla target
# CHECK-BE: bunlrl 2                        # encoding: [0x4d,0x8b,0x00,0x21]
# CHECK-LE: bunlrl 2                        # encoding: [0x21,0x00,0x8b,0x4d]
            bunlrl 2
# CHECK-BE: bunlrl 0                        # encoding: [0x4d,0x83,0x00,0x21]
# CHECK-LE: bunlrl 0                        # encoding: [0x21,0x00,0x83,0x4d]
            bunlrl
# CHECK-BE: bunctrl 2                       # encoding: [0x4d,0x8b,0x04,0x21]
# CHECK-LE: bunctrl 2                       # encoding: [0x21,0x04,0x8b,0x4d]
            bunctrl 2
# CHECK-BE: bunctrl 0                       # encoding: [0x4d,0x83,0x04,0x21]
# CHECK-LE: bunctrl 0                       # encoding: [0x21,0x04,0x83,0x4d]
            bunctrl

# CHECK-BE: bun+ 2, target                  # encoding: [0x41,0xeb,A,0bAAAAAA00]
# CHECK-LE: bun+ 2, target                  # encoding: [0bAAAAAA00,A,0xeb,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bun+ 2, target
# CHECK-BE: bun+ 0, target                  # encoding: [0x41,0xe3,A,0bAAAAAA00]
# CHECK-LE: bun+ 0, target                  # encoding: [0bAAAAAA00,A,0xe3,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bun+ target
# CHECK-BE: buna+ 2, target                 # encoding: [0x41,0xeb,A,0bAAAAAA10]
# CHECK-LE: buna+ 2, target                 # encoding: [0bAAAAAA10,A,0xeb,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            buna+ 2, target
# CHECK-BE: buna+ 0, target                 # encoding: [0x41,0xe3,A,0bAAAAAA10]
# CHECK-LE: buna+ 0, target                 # encoding: [0bAAAAAA10,A,0xe3,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            buna+ target
# CHECK-BE: bunlr+ 2                        # encoding: [0x4d,0xeb,0x00,0x20]
# CHECK-LE: bunlr+ 2                        # encoding: [0x20,0x00,0xeb,0x4d]
            bunlr+ 2
# CHECK-BE: bunlr+ 0                        # encoding: [0x4d,0xe3,0x00,0x20]
# CHECK-LE: bunlr+ 0                        # encoding: [0x20,0x00,0xe3,0x4d]
            bunlr+
# CHECK-BE: bunctr+ 2                       # encoding: [0x4d,0xeb,0x04,0x20]
# CHECK-LE: bunctr+ 2                       # encoding: [0x20,0x04,0xeb,0x4d]
            bunctr+ 2
# CHECK-BE: bunctr+ 0                       # encoding: [0x4d,0xe3,0x04,0x20]
# CHECK-LE: bunctr+ 0                       # encoding: [0x20,0x04,0xe3,0x4d]
            bunctr+
# CHECK-BE: bunl+ 2, target                 # encoding: [0x41,0xeb,A,0bAAAAAA01]
# CHECK-LE: bunl+ 2, target                 # encoding: [0bAAAAAA01,A,0xeb,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bunl+ 2, target
# CHECK-BE: bunl+ 0, target                 # encoding: [0x41,0xe3,A,0bAAAAAA01]
# CHECK-LE: bunl+ 0, target                 # encoding: [0bAAAAAA01,A,0xe3,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bunl+ target
# CHECK-BE: bunla+ 2, target                # encoding: [0x41,0xeb,A,0bAAAAAA11]
# CHECK-LE: bunla+ 2, target                # encoding: [0bAAAAAA11,A,0xeb,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bunla+ 2, target
# CHECK-BE: bunla+ 0, target                # encoding: [0x41,0xe3,A,0bAAAAAA11]
# CHECK-LE: bunla+ 0, target                # encoding: [0bAAAAAA11,A,0xe3,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bunla+ target
# CHECK-BE: bunlrl+ 2                       # encoding: [0x4d,0xeb,0x00,0x21]
# CHECK-LE: bunlrl+ 2                       # encoding: [0x21,0x00,0xeb,0x4d]
            bunlrl+ 2
# CHECK-BE: bunlrl+ 0                       # encoding: [0x4d,0xe3,0x00,0x21]
# CHECK-LE: bunlrl+ 0                       # encoding: [0x21,0x00,0xe3,0x4d]
            bunlrl+
# CHECK-BE: bunctrl+ 2                      # encoding: [0x4d,0xeb,0x04,0x21]
# CHECK-LE: bunctrl+ 2                      # encoding: [0x21,0x04,0xeb,0x4d]
            bunctrl+ 2
# CHECK-BE: bunctrl+ 0                      # encoding: [0x4d,0xe3,0x04,0x21]
# CHECK-LE: bunctrl+ 0                      # encoding: [0x21,0x04,0xe3,0x4d]
            bunctrl+

# CHECK-BE: bun- 2, target                  # encoding: [0x41,0xcb,A,0bAAAAAA00]
# CHECK-LE: bun- 2, target                  # encoding: [0bAAAAAA00,A,0xcb,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bun- 2, target
# CHECK-BE: bun- 0, target                  # encoding: [0x41,0xc3,A,0bAAAAAA00]
# CHECK-LE: bun- 0, target                  # encoding: [0bAAAAAA00,A,0xc3,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bun- target
# CHECK-BE: buna- 2, target                 # encoding: [0x41,0xcb,A,0bAAAAAA10]
# CHECK-LE: buna- 2, target                 # encoding: [0bAAAAAA10,A,0xcb,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            buna- 2, target
# CHECK-BE: buna- 0, target                 # encoding: [0x41,0xc3,A,0bAAAAAA10]
# CHECK-LE: buna- 0, target                 # encoding: [0bAAAAAA10,A,0xc3,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            buna- target
# CHECK-BE: bunlr- 2                        # encoding: [0x4d,0xcb,0x00,0x20]
# CHECK-LE: bunlr- 2                        # encoding: [0x20,0x00,0xcb,0x4d]
            bunlr- 2
# CHECK-BE: bunlr- 0                        # encoding: [0x4d,0xc3,0x00,0x20]
# CHECK-LE: bunlr- 0                        # encoding: [0x20,0x00,0xc3,0x4d]
            bunlr-
# CHECK-BE: bunctr- 2                       # encoding: [0x4d,0xcb,0x04,0x20]
# CHECK-LE: bunctr- 2                       # encoding: [0x20,0x04,0xcb,0x4d]
            bunctr- 2
# CHECK-BE: bunctr- 0                       # encoding: [0x4d,0xc3,0x04,0x20]
# CHECK-LE: bunctr- 0                       # encoding: [0x20,0x04,0xc3,0x4d]
            bunctr-
# CHECK-BE: bunl- 2, target                 # encoding: [0x41,0xcb,A,0bAAAAAA01]
# CHECK-LE: bunl- 2, target                 # encoding: [0bAAAAAA01,A,0xcb,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bunl- 2, target
# CHECK-BE: bunl- 0, target                 # encoding: [0x41,0xc3,A,0bAAAAAA01]
# CHECK-LE: bunl- 0, target                 # encoding: [0bAAAAAA01,A,0xc3,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bunl- target
# CHECK-BE: bunla- 2, target                # encoding: [0x41,0xcb,A,0bAAAAAA11]
# CHECK-LE: bunla- 2, target                # encoding: [0bAAAAAA11,A,0xcb,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bunla- 2, target
# CHECK-BE: bunla- 0, target                # encoding: [0x41,0xc3,A,0bAAAAAA11]
# CHECK-LE: bunla- 0, target                # encoding: [0bAAAAAA11,A,0xc3,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bunla- target
# CHECK-BE: bunlrl- 2                       # encoding: [0x4d,0xcb,0x00,0x21]
# CHECK-LE: bunlrl- 2                       # encoding: [0x21,0x00,0xcb,0x4d]
            bunlrl- 2
# CHECK-BE: bunlrl- 0                       # encoding: [0x4d,0xc3,0x00,0x21]
# CHECK-LE: bunlrl- 0                       # encoding: [0x21,0x00,0xc3,0x4d]
            bunlrl-
# CHECK-BE: bunctrl- 2                      # encoding: [0x4d,0xcb,0x04,0x21]
# CHECK-LE: bunctrl- 2                      # encoding: [0x21,0x04,0xcb,0x4d]
            bunctrl- 2
# CHECK-BE: bunctrl- 0                      # encoding: [0x4d,0xc3,0x04,0x21]
# CHECK-LE: bunctrl- 0                      # encoding: [0x21,0x04,0xc3,0x4d]
            bunctrl-

# CHECK-BE: bnu 2, target                   # encoding: [0x40,0x8b,A,0bAAAAAA00]
# CHECK-LE: bnu 2, target                   # encoding: [0bAAAAAA00,A,0x8b,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnu 2, target
# CHECK-BE: bnu 0, target                   # encoding: [0x40,0x83,A,0bAAAAAA00]
# CHECK-LE: bnu 0, target                   # encoding: [0bAAAAAA00,A,0x83,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnu target
# CHECK-BE: bnua 2, target                  # encoding: [0x40,0x8b,A,0bAAAAAA10]
# CHECK-LE: bnua 2, target                  # encoding: [0bAAAAAA10,A,0x8b,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnua 2, target
# CHECK-BE: bnua 0, target                  # encoding: [0x40,0x83,A,0bAAAAAA10]
# CHECK-LE: bnua 0, target                  # encoding: [0bAAAAAA10,A,0x83,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnua target
# CHECK-BE: bnulr 2                         # encoding: [0x4c,0x8b,0x00,0x20]
# CHECK-LE: bnulr 2                         # encoding: [0x20,0x00,0x8b,0x4c]
            bnulr 2
# CHECK-BE: bnulr 0                         # encoding: [0x4c,0x83,0x00,0x20]
# CHECK-LE: bnulr 0                         # encoding: [0x20,0x00,0x83,0x4c]
            bnulr
# CHECK-BE: bnuctr 2                        # encoding: [0x4c,0x8b,0x04,0x20]
# CHECK-LE: bnuctr 2                        # encoding: [0x20,0x04,0x8b,0x4c]
            bnuctr 2
# CHECK-BE: bnuctr 0                        # encoding: [0x4c,0x83,0x04,0x20]
# CHECK-LE: bnuctr 0                        # encoding: [0x20,0x04,0x83,0x4c]
            bnuctr
# CHECK-BE: bnul 2, target                  # encoding: [0x40,0x8b,A,0bAAAAAA01]
# CHECK-LE: bnul 2, target                  # encoding: [0bAAAAAA01,A,0x8b,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnul 2, target
# CHECK-BE: bnul 0, target                  # encoding: [0x40,0x83,A,0bAAAAAA01]
# CHECK-LE: bnul 0, target                  # encoding: [0bAAAAAA01,A,0x83,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnul target
# CHECK-BE: bnula 2, target                 # encoding: [0x40,0x8b,A,0bAAAAAA11]
# CHECK-LE: bnula 2, target                 # encoding: [0bAAAAAA11,A,0x8b,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnula 2, target
# CHECK-BE: bnula 0, target                 # encoding: [0x40,0x83,A,0bAAAAAA11]
# CHECK-LE: bnula 0, target                 # encoding: [0bAAAAAA11,A,0x83,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnula target
# CHECK-BE: bnulrl 2                        # encoding: [0x4c,0x8b,0x00,0x21]
# CHECK-LE: bnulrl 2                        # encoding: [0x21,0x00,0x8b,0x4c]
            bnulrl 2
# CHECK-BE: bnulrl 0                        # encoding: [0x4c,0x83,0x00,0x21]
# CHECK-LE: bnulrl 0                        # encoding: [0x21,0x00,0x83,0x4c]
            bnulrl
# CHECK-BE: bnuctrl 2                       # encoding: [0x4c,0x8b,0x04,0x21]
# CHECK-LE: bnuctrl 2                       # encoding: [0x21,0x04,0x8b,0x4c]
            bnuctrl 2
# CHECK-BE: bnuctrl 0                       # encoding: [0x4c,0x83,0x04,0x21]
# CHECK-LE: bnuctrl 0                       # encoding: [0x21,0x04,0x83,0x4c]
            bnuctrl

# CHECK-BE: bnu+ 2, target                  # encoding: [0x40,0xeb,A,0bAAAAAA00]
# CHECK-LE: bnu+ 2, target                  # encoding: [0bAAAAAA00,A,0xeb,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnu+ 2, target
# CHECK-BE: bnu+ 0, target                  # encoding: [0x40,0xe3,A,0bAAAAAA00]
# CHECK-LE: bnu+ 0, target                  # encoding: [0bAAAAAA00,A,0xe3,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnu+ target
# CHECK-BE: bnua+ 2, target                 # encoding: [0x40,0xeb,A,0bAAAAAA10]
# CHECK-LE: bnua+ 2, target                 # encoding: [0bAAAAAA10,A,0xeb,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnua+ 2, target
# CHECK-BE: bnua+ 0, target                 # encoding: [0x40,0xe3,A,0bAAAAAA10]
# CHECK-LE: bnua+ 0, target                 # encoding: [0bAAAAAA10,A,0xe3,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnua+ target
# CHECK-BE: bnulr+ 2                        # encoding: [0x4c,0xeb,0x00,0x20]
# CHECK-LE: bnulr+ 2                        # encoding: [0x20,0x00,0xeb,0x4c]
            bnulr+ 2
# CHECK-BE: bnulr+ 0                        # encoding: [0x4c,0xe3,0x00,0x20]
# CHECK-LE: bnulr+ 0                        # encoding: [0x20,0x00,0xe3,0x4c]
            bnulr+
# CHECK-BE: bnuctr+ 2                       # encoding: [0x4c,0xeb,0x04,0x20]
# CHECK-LE: bnuctr+ 2                       # encoding: [0x20,0x04,0xeb,0x4c]
            bnuctr+ 2
# CHECK-BE: bnuctr+ 0                       # encoding: [0x4c,0xe3,0x04,0x20]
# CHECK-LE: bnuctr+ 0                       # encoding: [0x20,0x04,0xe3,0x4c]
            bnuctr+
# CHECK-BE: bnul+ 2, target                 # encoding: [0x40,0xeb,A,0bAAAAAA01]
# CHECK-LE: bnul+ 2, target                 # encoding: [0bAAAAAA01,A,0xeb,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnul+ 2, target
# CHECK-BE: bnul+ 0, target                 # encoding: [0x40,0xe3,A,0bAAAAAA01]
# CHECK-LE: bnul+ 0, target                 # encoding: [0bAAAAAA01,A,0xe3,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnul+ target
# CHECK-BE: bnula+ 2, target                # encoding: [0x40,0xeb,A,0bAAAAAA11]
# CHECK-LE: bnula+ 2, target                # encoding: [0bAAAAAA11,A,0xeb,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnula+ 2, target
# CHECK-BE: bnula+ 0, target                # encoding: [0x40,0xe3,A,0bAAAAAA11]
# CHECK-LE: bnula+ 0, target                # encoding: [0bAAAAAA11,A,0xe3,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnula+ target
# CHECK-BE: bnulrl+ 2                       # encoding: [0x4c,0xeb,0x00,0x21]
# CHECK-LE: bnulrl+ 2                       # encoding: [0x21,0x00,0xeb,0x4c]
            bnulrl+ 2
# CHECK-BE: bnulrl+ 0                       # encoding: [0x4c,0xe3,0x00,0x21]
# CHECK-LE: bnulrl+ 0                       # encoding: [0x21,0x00,0xe3,0x4c]
            bnulrl+
# CHECK-BE: bnuctrl+ 2                      # encoding: [0x4c,0xeb,0x04,0x21]
# CHECK-LE: bnuctrl+ 2                      # encoding: [0x21,0x04,0xeb,0x4c]
            bnuctrl+ 2
# CHECK-BE: bnuctrl+ 0                      # encoding: [0x4c,0xe3,0x04,0x21]
# CHECK-LE: bnuctrl+ 0                      # encoding: [0x21,0x04,0xe3,0x4c]
            bnuctrl+

# CHECK-BE: bnu- 2, target                  # encoding: [0x40,0xcb,A,0bAAAAAA00]
# CHECK-LE: bnu- 2, target                  # encoding: [0bAAAAAA00,A,0xcb,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnu- 2, target
# CHECK-BE: bnu- 0, target                  # encoding: [0x40,0xc3,A,0bAAAAAA00]
# CHECK-LE: bnu- 0, target                  # encoding: [0bAAAAAA00,A,0xc3,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnu- target
# CHECK-BE: bnua- 2, target                 # encoding: [0x40,0xcb,A,0bAAAAAA10]
# CHECK-LE: bnua- 2, target                 # encoding: [0bAAAAAA10,A,0xcb,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnua- 2, target
# CHECK-BE: bnua- 0, target                 # encoding: [0x40,0xc3,A,0bAAAAAA10]
# CHECK-LE: bnua- 0, target                 # encoding: [0bAAAAAA10,A,0xc3,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnua- target
# CHECK-BE: bnulr- 2                        # encoding: [0x4c,0xcb,0x00,0x20]
# CHECK-LE: bnulr- 2                        # encoding: [0x20,0x00,0xcb,0x4c]
            bnulr- 2
# CHECK-BE: bnulr- 0                        # encoding: [0x4c,0xc3,0x00,0x20]
# CHECK-LE: bnulr- 0                        # encoding: [0x20,0x00,0xc3,0x4c]
            bnulr-
# CHECK-BE: bnuctr- 2                       # encoding: [0x4c,0xcb,0x04,0x20]
# CHECK-LE: bnuctr- 2                       # encoding: [0x20,0x04,0xcb,0x4c]
            bnuctr- 2
# CHECK-BE: bnuctr- 0                       # encoding: [0x4c,0xc3,0x04,0x20]
# CHECK-LE: bnuctr- 0                       # encoding: [0x20,0x04,0xc3,0x4c]
            bnuctr-
# CHECK-BE: bnul- 2, target                 # encoding: [0x40,0xcb,A,0bAAAAAA01]
# CHECK-LE: bnul- 2, target                 # encoding: [0bAAAAAA01,A,0xcb,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnul- 2, target
# CHECK-BE: bnul- 0, target                 # encoding: [0x40,0xc3,A,0bAAAAAA01]
# CHECK-LE: bnul- 0, target                 # encoding: [0bAAAAAA01,A,0xc3,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bnul- target
# CHECK-BE: bnula- 2, target                # encoding: [0x40,0xcb,A,0bAAAAAA11]
# CHECK-LE: bnula- 2, target                # encoding: [0bAAAAAA11,A,0xcb,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnula- 2, target
# CHECK-BE: bnula- 0, target                # encoding: [0x40,0xc3,A,0bAAAAAA11]
# CHECK-LE: bnula- 0, target                # encoding: [0bAAAAAA11,A,0xc3,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bnula- target
# CHECK-BE: bnulrl- 2                       # encoding: [0x4c,0xcb,0x00,0x21]
# CHECK-LE: bnulrl- 2                       # encoding: [0x21,0x00,0xcb,0x4c]
            bnulrl- 2
# CHECK-BE: bnulrl- 0                       # encoding: [0x4c,0xc3,0x00,0x21]
# CHECK-LE: bnulrl- 0                       # encoding: [0x21,0x00,0xc3,0x4c]
            bnulrl-
# CHECK-BE: bnuctrl- 2                      # encoding: [0x4c,0xcb,0x04,0x21]
# CHECK-LE: bnuctrl- 2                      # encoding: [0x21,0x04,0xcb,0x4c]
            bnuctrl- 2
# CHECK-BE: bnuctrl- 0                      # encoding: [0x4c,0xc3,0x04,0x21]
# CHECK-LE: bnuctrl- 0                      # encoding: [0x21,0x04,0xc3,0x4c]
            bnuctrl-

# Condition register logical mnemonics

# CHECK-BE: crset 2                         # encoding: [0x4c,0x42,0x12,0x42]
# CHECK-LE: crset 2                         # encoding: [0x42,0x12,0x42,0x4c]
            crset 2
# CHECK-BE: crclr 2                         # encoding: [0x4c,0x42,0x11,0x82]
# CHECK-LE: crclr 2                         # encoding: [0x82,0x11,0x42,0x4c]
            crclr 2
# CHECK-BE: crmove 2, 3                     # encoding: [0x4c,0x43,0x1b,0x82]
# CHECK-LE: crmove 2, 3                     # encoding: [0x82,0x1b,0x43,0x4c]
            crmove 2, 3
# CHECK-BE: crnot 2, 3                      # encoding: [0x4c,0x43,0x18,0x42]
# CHECK-LE: crnot 2, 3                      # encoding: [0x42,0x18,0x43,0x4c]
            crnot 2, 3

# Subtract mnemonics

# CHECK-BE: addi 2, 3, -128                 # encoding: [0x38,0x43,0xff,0x80]
# CHECK-LE: addi 2, 3, -128                 # encoding: [0x80,0xff,0x43,0x38]
            subi 2, 3, 128
# CHECK-BE: addis 2, 3, -128                # encoding: [0x3c,0x43,0xff,0x80]
# CHECK-LE: addis 2, 3, -128                # encoding: [0x80,0xff,0x43,0x3c]
            subis 2, 3, 128
# CHECK-BE: addic 2, 3, -128                # encoding: [0x30,0x43,0xff,0x80]
# CHECK-LE: addic 2, 3, -128                # encoding: [0x80,0xff,0x43,0x30]
            subic 2, 3, 128
# CHECK-BE: addic. 2, 3, -128               # encoding: [0x34,0x43,0xff,0x80]
# CHECK-LE: addic. 2, 3, -128               # encoding: [0x80,0xff,0x43,0x34]
            subic. 2, 3, 128

# CHECK-BE: sub 2, 3, 4                     # encoding: [0x7c,0x44,0x18,0x50]
# CHECK-LE: sub 2, 3, 4                     # encoding: [0x50,0x18,0x44,0x7c]
            sub 2, 3, 4
# CHECK-BE: sub. 2, 3, 4                    # encoding: [0x7c,0x44,0x18,0x51]
# CHECK-LE: sub. 2, 3, 4                    # encoding: [0x51,0x18,0x44,0x7c]
            sub. 2, 3, 4
# CHECK-BE: subc 2, 3, 4                    # encoding: [0x7c,0x44,0x18,0x10]
# CHECK-LE: subc 2, 3, 4                    # encoding: [0x10,0x18,0x44,0x7c]
            subc 2, 3, 4
# CHECK-BE: subc. 2, 3, 4                   # encoding: [0x7c,0x44,0x18,0x11]
# CHECK-LE: subc. 2, 3, 4                   # encoding: [0x11,0x18,0x44,0x7c]
            subc. 2, 3, 4

# Compare mnemonics

# CHECK-BE: cmpdi 2, 3, 128                 # encoding: [0x2d,0x23,0x00,0x80]
# CHECK-LE: cmpdi 2, 3, 128                 # encoding: [0x80,0x00,0x23,0x2d]
            cmpdi 2, 3, 128
# CHECK-BE: cmpdi 3, 128                    # encoding: [0x2c,0x23,0x00,0x80]
# CHECK-LE: cmpdi 3, 128                    # encoding: [0x80,0x00,0x23,0x2c]
            cmpdi 3, 128
# CHECK-BE: cmpd 2, 3, 4                    # encoding: [0x7d,0x23,0x20,0x00]
# CHECK-LE: cmpd 2, 3, 4                    # encoding: [0x00,0x20,0x23,0x7d]
            cmpd 2, 3, 4
# CHECK-BE: cmpd 3, 4                       # encoding: [0x7c,0x23,0x20,0x00]
# CHECK-LE: cmpd 3, 4                       # encoding: [0x00,0x20,0x23,0x7c]
            cmpd 3, 4
# CHECK-BE: cmpldi 2, 3, 128                # encoding: [0x29,0x23,0x00,0x80]
# CHECK-LE: cmpldi 2, 3, 128                # encoding: [0x80,0x00,0x23,0x29]
            cmpldi 2, 3, 128
# CHECK-BE: cmpldi 3, 128                   # encoding: [0x28,0x23,0x00,0x80]
# CHECK-LE: cmpldi 3, 128                   # encoding: [0x80,0x00,0x23,0x28]
            cmpldi 3, 128
# CHECK-BE: cmpld 2, 3, 4                   # encoding: [0x7d,0x23,0x20,0x40]
# CHECK-LE: cmpld 2, 3, 4                   # encoding: [0x40,0x20,0x23,0x7d]
            cmpld 2, 3, 4
# CHECK-BE: cmpld 3, 4                      # encoding: [0x7c,0x23,0x20,0x40]
# CHECK-LE: cmpld 3, 4                      # encoding: [0x40,0x20,0x23,0x7c]
            cmpld 3, 4

# CHECK-BE: cmpwi 2, 3, 128                 # encoding: [0x2d,0x03,0x00,0x80]
# CHECK-LE: cmpwi 2, 3, 128                 # encoding: [0x80,0x00,0x03,0x2d]
            cmpwi 2, 3, 128
# CHECK-BE: cmpwi 3, 128                    # encoding: [0x2c,0x03,0x00,0x80]
# CHECK-LE: cmpwi 3, 128                    # encoding: [0x80,0x00,0x03,0x2c]
            cmpwi 3, 128
# CHECK-BE: cmpw 2, 3, 4                    # encoding: [0x7d,0x03,0x20,0x00]
# CHECK-LE: cmpw 2, 3, 4                    # encoding: [0x00,0x20,0x03,0x7d]
            cmpw 2, 3, 4
# CHECK-BE: cmpw 3, 4                       # encoding: [0x7c,0x03,0x20,0x00]
# CHECK-LE: cmpw 3, 4                       # encoding: [0x00,0x20,0x03,0x7c]
            cmpw 3, 4
# CHECK-BE: cmplwi 2, 3, 128                # encoding: [0x29,0x03,0x00,0x80]
# CHECK-LE: cmplwi 2, 3, 128                # encoding: [0x80,0x00,0x03,0x29]
            cmplwi 2, 3, 128
# CHECK-BE: cmplwi 3, 128                   # encoding: [0x28,0x03,0x00,0x80]
# CHECK-LE: cmplwi 3, 128                   # encoding: [0x80,0x00,0x03,0x28]
            cmplwi 3, 128
# CHECK-BE: cmplw 2, 3, 4                   # encoding: [0x7d,0x03,0x20,0x40]
# CHECK-LE: cmplw 2, 3, 4                   # encoding: [0x40,0x20,0x03,0x7d]
            cmplw 2, 3, 4
# CHECK-BE: cmplw 3, 4                      # encoding: [0x7c,0x03,0x20,0x40]
# CHECK-LE: cmplw 3, 4                      # encoding: [0x40,0x20,0x03,0x7c]
            cmplw 3, 4

# Trap mnemonics

# CHECK-BE: twlti 3, 4                      # encoding: [0x0e,0x03,0x00,0x04]
# CHECK-LE: twlti 3, 4                      # encoding: [0x04,0x00,0x03,0x0e]
            twlti 3, 4
# CHECK-BE: twlt 3, 4                       # encoding: [0x7e,0x03,0x20,0x08]
# CHECK-LE: twlt 3, 4                       # encoding: [0x08,0x20,0x03,0x7e]
            twlt 3, 4
# CHECK-BE: tdlti 3, 4                      # encoding: [0x0a,0x03,0x00,0x04]
# CHECK-LE: tdlti 3, 4                      # encoding: [0x04,0x00,0x03,0x0a]
            tdlti 3, 4
# CHECK-BE: tdlt 3, 4                       # encoding: [0x7e,0x03,0x20,0x88]
# CHECK-LE: tdlt 3, 4                       # encoding: [0x88,0x20,0x03,0x7e]
            tdlt 3, 4

# CHECK-BE: twi 20, 3, 4                    # encoding: [0x0e,0x83,0x00,0x04]
# CHECK-LE: twi 20, 3, 4                    # encoding: [0x04,0x00,0x83,0x0e]
            twlei 3, 4
# CHECK-BE: tw 20, 3, 4                     # encoding: [0x7e,0x83,0x20,0x08]
# CHECK-LE: tw 20, 3, 4                     # encoding: [0x08,0x20,0x83,0x7e]
            twle 3, 4
# CHECK-BE: tdi 20, 3, 4                    # encoding: [0x0a,0x83,0x00,0x04]
# CHECK-LE: tdi 20, 3, 4                    # encoding: [0x04,0x00,0x83,0x0a]
            tdlei 3, 4
# CHECK-BE: td 20, 3, 4                     # encoding: [0x7e,0x83,0x20,0x88]
# CHECK-LE: td 20, 3, 4                     # encoding: [0x88,0x20,0x83,0x7e]
            tdle 3, 4

# CHECK-BE: tweqi 3, 4                      # encoding: [0x0c,0x83,0x00,0x04]
# CHECK-LE: tweqi 3, 4                      # encoding: [0x04,0x00,0x83,0x0c]
            tweqi 3, 4
# CHECK-BE: tweq 3, 4                       # encoding: [0x7c,0x83,0x20,0x08]
# CHECK-LE: tweq 3, 4                       # encoding: [0x08,0x20,0x83,0x7c]
            tweq 3, 4
# CHECK-BE: tdeqi 3, 4                      # encoding: [0x08,0x83,0x00,0x04]
# CHECK-LE: tdeqi 3, 4                      # encoding: [0x04,0x00,0x83,0x08]
            tdeqi 3, 4
# CHECK-BE: tdeq 3, 4                       # encoding: [0x7c,0x83,0x20,0x88]
# CHECK-LE: tdeq 3, 4                       # encoding: [0x88,0x20,0x83,0x7c]
            tdeq 3, 4

# CHECK-BE: twi 12, 3, 4                    # encoding: [0x0d,0x83,0x00,0x04]
# CHECK-LE: twi 12, 3, 4                    # encoding: [0x04,0x00,0x83,0x0d]
            twgei 3, 4
# CHECK-BE: tw 12, 3, 4                     # encoding: [0x7d,0x83,0x20,0x08]
# CHECK-LE: tw 12, 3, 4                     # encoding: [0x08,0x20,0x83,0x7d]
            twge 3, 4
# CHECK-BE: tdi 12, 3, 4                    # encoding: [0x09,0x83,0x00,0x04]
# CHECK-LE: tdi 12, 3, 4                    # encoding: [0x04,0x00,0x83,0x09]
            tdgei 3, 4
# CHECK-BE: td 12, 3, 4                     # encoding: [0x7d,0x83,0x20,0x88]
# CHECK-LE: td 12, 3, 4                     # encoding: [0x88,0x20,0x83,0x7d]
            tdge 3, 4

# CHECK-BE: twgti 3, 4                      # encoding: [0x0d,0x03,0x00,0x04]
# CHECK-LE: twgti 3, 4                      # encoding: [0x04,0x00,0x03,0x0d]
            twgti 3, 4
# CHECK-BE: twgt 3, 4                       # encoding: [0x7d,0x03,0x20,0x08]
# CHECK-LE: twgt 3, 4                       # encoding: [0x08,0x20,0x03,0x7d]
            twgt 3, 4
# CHECK-BE: tdgti 3, 4                      # encoding: [0x09,0x03,0x00,0x04]
# CHECK-LE: tdgti 3, 4                      # encoding: [0x04,0x00,0x03,0x09]
            tdgti 3, 4
# CHECK-BE: tdgt 3, 4                       # encoding: [0x7d,0x03,0x20,0x88]
# CHECK-LE: tdgt 3, 4                       # encoding: [0x88,0x20,0x03,0x7d]
            tdgt 3, 4

# CHECK-BE: twi 12, 3, 4                    # encoding: [0x0d,0x83,0x00,0x04]
# CHECK-LE: twi 12, 3, 4                    # encoding: [0x04,0x00,0x83,0x0d]
            twnli 3, 4
# CHECK-BE: tw 12, 3, 4                     # encoding: [0x7d,0x83,0x20,0x08]
# CHECK-LE: tw 12, 3, 4                     # encoding: [0x08,0x20,0x83,0x7d]
            twnl 3, 4
# CHECK-BE: tdi 12, 3, 4                    # encoding: [0x09,0x83,0x00,0x04]
# CHECK-LE: tdi 12, 3, 4                    # encoding: [0x04,0x00,0x83,0x09]
            tdnli 3, 4
# CHECK-BE: td 12, 3, 4                     # encoding: [0x7d,0x83,0x20,0x88]
# CHECK-LE: td 12, 3, 4                     # encoding: [0x88,0x20,0x83,0x7d]
            tdnl 3, 4

# CHECK-BE: twnei 3, 4                      # encoding: [0x0f,0x03,0x00,0x04]
# CHECK-LE: twnei 3, 4                      # encoding: [0x04,0x00,0x03,0x0f]
            twnei 3, 4
# CHECK-BE: twne 3, 4                       # encoding: [0x7f,0x03,0x20,0x08]
# CHECK-LE: twne 3, 4                       # encoding: [0x08,0x20,0x03,0x7f]
            twne 3, 4
# CHECK-BE: tdnei 3, 4                      # encoding: [0x0b,0x03,0x00,0x04]
# CHECK-LE: tdnei 3, 4                      # encoding: [0x04,0x00,0x03,0x0b]
            tdnei 3, 4
# CHECK-BE: tdne 3, 4                       # encoding: [0x7f,0x03,0x20,0x88]
# CHECK-LE: tdne 3, 4                       # encoding: [0x88,0x20,0x03,0x7f]
            tdne 3, 4

# CHECK-BE: twi 20, 3, 4                    # encoding: [0x0e,0x83,0x00,0x04]
# CHECK-LE: twi 20, 3, 4                    # encoding: [0x04,0x00,0x83,0x0e]
            twngi 3, 4
# CHECK-BE: tw 20, 3, 4                     # encoding: [0x7e,0x83,0x20,0x08]
# CHECK-LE: tw 20, 3, 4                     # encoding: [0x08,0x20,0x83,0x7e]
            twng 3, 4
# CHECK-BE: tdi 20, 3, 4                    # encoding: [0x0a,0x83,0x00,0x04]
# CHECK-LE: tdi 20, 3, 4                    # encoding: [0x04,0x00,0x83,0x0a]
            tdngi 3, 4
# CHECK-BE: td 20, 3, 4                     # encoding: [0x7e,0x83,0x20,0x88]
# CHECK-LE: td 20, 3, 4                     # encoding: [0x88,0x20,0x83,0x7e]
            tdng 3, 4

# CHECK-BE: twllti 3, 4                     # encoding: [0x0c,0x43,0x00,0x04]
# CHECK-LE: twllti 3, 4                     # encoding: [0x04,0x00,0x43,0x0c]
            twllti 3, 4
# CHECK-BE: twllt 3, 4                      # encoding: [0x7c,0x43,0x20,0x08]
# CHECK-LE: twllt 3, 4                      # encoding: [0x08,0x20,0x43,0x7c]
            twllt 3, 4
# CHECK-BE: tdllti 3, 4                     # encoding: [0x08,0x43,0x00,0x04]
# CHECK-LE: tdllti 3, 4                     # encoding: [0x04,0x00,0x43,0x08]
            tdllti 3, 4
# CHECK-BE: tdllt 3, 4                      # encoding: [0x7c,0x43,0x20,0x88]
# CHECK-LE: tdllt 3, 4                      # encoding: [0x88,0x20,0x43,0x7c]
            tdllt 3, 4

# CHECK-BE: twi 6, 3, 4                     # encoding: [0x0c,0xc3,0x00,0x04]
# CHECK-LE: twi 6, 3, 4                     # encoding: [0x04,0x00,0xc3,0x0c]
            twllei 3, 4
# CHECK-BE: tw 6, 3, 4                      # encoding: [0x7c,0xc3,0x20,0x08]
# CHECK-LE: tw 6, 3, 4                      # encoding: [0x08,0x20,0xc3,0x7c]
            twlle 3, 4
# CHECK-BE: tdi 6, 3, 4                     # encoding: [0x08,0xc3,0x00,0x04]
# CHECK-LE: tdi 6, 3, 4                     # encoding: [0x04,0x00,0xc3,0x08]
            tdllei 3, 4
# CHECK-BE: td 6, 3, 4                      # encoding: [0x7c,0xc3,0x20,0x88]
# CHECK-LE: td 6, 3, 4                      # encoding: [0x88,0x20,0xc3,0x7c]
            tdlle 3, 4

# CHECK-BE: twi 5, 3, 4                     # encoding: [0x0c,0xa3,0x00,0x04]
# CHECK-LE: twi 5, 3, 4                     # encoding: [0x04,0x00,0xa3,0x0c]
            twlgei 3, 4
# CHECK-BE: tw 5, 3, 4                      # encoding: [0x7c,0xa3,0x20,0x08]
# CHECK-LE: tw 5, 3, 4                      # encoding: [0x08,0x20,0xa3,0x7c]
            twlge 3, 4
# CHECK-BE: tdi 5, 3, 4                     # encoding: [0x08,0xa3,0x00,0x04]
# CHECK-LE: tdi 5, 3, 4                     # encoding: [0x04,0x00,0xa3,0x08]
            tdlgei 3, 4
# CHECK-BE: td 5, 3, 4                      # encoding: [0x7c,0xa3,0x20,0x88]
# CHECK-LE: td 5, 3, 4                      # encoding: [0x88,0x20,0xa3,0x7c]
            tdlge 3, 4

# CHECK-BE: twlgti 3, 4                     # encoding: [0x0c,0x23,0x00,0x04]
# CHECK-LE: twlgti 3, 4                     # encoding: [0x04,0x00,0x23,0x0c]
            twlgti 3, 4
# CHECK-BE: twlgt 3, 4                      # encoding: [0x7c,0x23,0x20,0x08]
# CHECK-LE: twlgt 3, 4                      # encoding: [0x08,0x20,0x23,0x7c]
            twlgt 3, 4
# CHECK-BE: tdlgti 3, 4                     # encoding: [0x08,0x23,0x00,0x04]
# CHECK-LE: tdlgti 3, 4                     # encoding: [0x04,0x00,0x23,0x08]
            tdlgti 3, 4
# CHECK-BE: tdlgt 3, 4                      # encoding: [0x7c,0x23,0x20,0x88]
# CHECK-LE: tdlgt 3, 4                      # encoding: [0x88,0x20,0x23,0x7c]
            tdlgt 3, 4

# CHECK-BE: twi 5, 3, 4                     # encoding: [0x0c,0xa3,0x00,0x04]
# CHECK-LE: twi 5, 3, 4                     # encoding: [0x04,0x00,0xa3,0x0c]
            twlnli 3, 4
# CHECK-BE: tw 5, 3, 4                      # encoding: [0x7c,0xa3,0x20,0x08]
# CHECK-LE: tw 5, 3, 4                      # encoding: [0x08,0x20,0xa3,0x7c]
            twlnl 3, 4
# CHECK-BE: tdi 5, 3, 4                     # encoding: [0x08,0xa3,0x00,0x04]
# CHECK-LE: tdi 5, 3, 4                     # encoding: [0x04,0x00,0xa3,0x08]
            tdlnli 3, 4
# CHECK-BE: td 5, 3, 4                      # encoding: [0x7c,0xa3,0x20,0x88]
# CHECK-LE: td 5, 3, 4                      # encoding: [0x88,0x20,0xa3,0x7c]
            tdlnl 3, 4

# CHECK-BE: twi 6, 3, 4                     # encoding: [0x0c,0xc3,0x00,0x04]
# CHECK-LE: twi 6, 3, 4                     # encoding: [0x04,0x00,0xc3,0x0c]
            twlngi 3, 4
# CHECK-BE: tw 6, 3, 4                      # encoding: [0x7c,0xc3,0x20,0x08]
# CHECK-LE: tw 6, 3, 4                      # encoding: [0x08,0x20,0xc3,0x7c]
            twlng 3, 4
# CHECK-BE: tdi 6, 3, 4                     # encoding: [0x08,0xc3,0x00,0x04]
# CHECK-LE: tdi 6, 3, 4                     # encoding: [0x04,0x00,0xc3,0x08]
            tdlngi 3, 4
# CHECK-BE: td 6, 3, 4                      # encoding: [0x7c,0xc3,0x20,0x88]
# CHECK-LE: td 6, 3, 4                      # encoding: [0x88,0x20,0xc3,0x7c]
            tdlng 3, 4

# CHECK-BE: twui 3, 4                       # encoding: [0x0f,0xe3,0x00,0x04]
# CHECK-LE: twui 3, 4                       # encoding: [0x04,0x00,0xe3,0x0f]
            twui 3, 4
# CHECK-BE: twu 3, 4                        # encoding: [0x7f,0xe3,0x20,0x08]
# CHECK-LE: twu 3, 4                        # encoding: [0x08,0x20,0xe3,0x7f]
            twu 3, 4
# CHECK-BE: tdui 3, 4                       # encoding: [0x0b,0xe3,0x00,0x04]
# CHECK-LE: tdui 3, 4                       # encoding: [0x04,0x00,0xe3,0x0b]
            tdui 3, 4
# CHECK-BE: tdu 3, 4                         # encoding: [0x7f,0xe3,0x20,0x88]
# CHECK-LE: tdu 3, 4                         # encoding: [0x88,0x20,0xe3,0x7f]
            tdu 3, 4

# CHECK-BE: trap                            # encoding: [0x7f,0xe0,0x00,0x08]
# CHECK-LE: trap                            # encoding: [0x08,0x00,0xe0,0x7f]
            trap

# Rotate and shift mnemonics

# CHECK-BE: rldicr 2, 3, 5, 3               # encoding: [0x78,0x62,0x28,0xc4]
# CHECK-LE: rldicr 2, 3, 5, 3               # encoding: [0xc4,0x28,0x62,0x78]
            extldi 2, 3, 4, 5
# CHECK-BE: rldicr. 2, 3, 5, 3              # encoding: [0x78,0x62,0x28,0xc5]
# CHECK-LE: rldicr. 2, 3, 5, 3              # encoding: [0xc5,0x28,0x62,0x78]
            extldi. 2, 3, 4, 5
# CHECK-BE: rldicl 2, 3, 9, 60              # encoding: [0x78,0x62,0x4f,0x20]
# CHECK-LE: rldicl 2, 3, 9, 60              # encoding: [0x20,0x4f,0x62,0x78]
            extrdi 2, 3, 4, 5
# CHECK-BE: rldicl. 2, 3, 9, 60             # encoding: [0x78,0x62,0x4f,0x21]
# CHECK-LE: rldicl. 2, 3, 9, 60             # encoding: [0x21,0x4f,0x62,0x78]
            extrdi. 2, 3, 4, 5
# CHECK-BE: rldimi 2, 3, 55, 5              # encoding: [0x78,0x62,0xb9,0x4e]
# CHECK-LE: rldimi 2, 3, 55, 5              # encoding: [0x4e,0xb9,0x62,0x78]
            insrdi 2, 3, 4, 5
# CHECK-BE: rldimi. 2, 3, 55, 5             # encoding: [0x78,0x62,0xb9,0x4f]
# CHECK-LE: rldimi. 2, 3, 55, 5             # encoding: [0x4f,0xb9,0x62,0x78]
            insrdi. 2, 3, 4, 5
# CHECK-BE: rotldi 2, 3, 4                  # encoding: [0x78,0x62,0x20,0x00]
# CHECK-LE: rotldi 2, 3, 4                  # encoding: [0x00,0x20,0x62,0x78]
            rotldi 2, 3, 4
# CHECK-BE: rotldi. 2, 3, 4                 # encoding: [0x78,0x62,0x20,0x01]
# CHECK-LE: rotldi. 2, 3, 4                 # encoding: [0x01,0x20,0x62,0x78]
            rotldi. 2, 3, 4
# CHECK-BE: rotldi 2, 3, 60                 # encoding: [0x78,0x62,0xe0,0x02]
# CHECK-LE: rotldi 2, 3, 60                 # encoding: [0x02,0xe0,0x62,0x78]
            rotrdi 2, 3, 4
# CHECK-BE: rotldi. 2, 3, 60                # encoding: [0x78,0x62,0xe0,0x03]
# CHECK-LE: rotldi. 2, 3, 60                # encoding: [0x03,0xe0,0x62,0x78]
            rotrdi. 2, 3, 4
# CHECK-BE: rotld 2, 3, 4                   # encoding: [0x78,0x62,0x20,0x10]
# CHECK-LE: rotld 2, 3, 4                   # encoding: [0x10,0x20,0x62,0x78]
            rotld 2, 3, 4
# CHECK-BE: rotld. 2, 3, 4                  # encoding: [0x78,0x62,0x20,0x11]
# CHECK-LE: rotld. 2, 3, 4                  # encoding: [0x11,0x20,0x62,0x78]
            rotld. 2, 3, 4
# CHECK-BE: sldi 2, 3, 4                    # encoding: [0x78,0x62,0x26,0xe4]
# CHECK-LE: sldi 2, 3, 4                    # encoding: [0xe4,0x26,0x62,0x78]
            sldi 2, 3, 4
# CHECK-BE: rldicr. 2, 3, 4, 59             # encoding: [0x78,0x62,0x26,0xe5]
# CHECK-LE: rldicr. 2, 3, 4, 59             # encoding: [0xe5,0x26,0x62,0x78]
            sldi. 2, 3, 4
# CHECK-BE: rldicl 2, 3, 60, 4              # encoding: [0x78,0x62,0xe1,0x02]
# CHECK-LE: rldicl 2, 3, 60, 4              # encoding: [0x02,0xe1,0x62,0x78]
            srdi 2, 3, 4
# CHECK-BE: rldicl. 2, 3, 60, 4             # encoding: [0x78,0x62,0xe1,0x03]
# CHECK-LE: rldicl. 2, 3, 60, 4             # encoding: [0x03,0xe1,0x62,0x78]
            srdi. 2, 3, 4
# CHECK-BE: clrldi 2, 3, 4                  # encoding: [0x78,0x62,0x01,0x00]
# CHECK-LE: clrldi 2, 3, 4                  # encoding: [0x00,0x01,0x62,0x78]
            clrldi 2, 3, 4
# CHECK-BE: clrldi. 2, 3, 4                 # encoding: [0x78,0x62,0x01,0x01]
# CHECK-LE: clrldi. 2, 3, 4                 # encoding: [0x01,0x01,0x62,0x78]
            clrldi. 2, 3, 4
# CHECK-BE: rldicr 2, 3, 0, 59              # encoding: [0x78,0x62,0x06,0xe4]
# CHECK-LE: rldicr 2, 3, 0, 59              # encoding: [0xe4,0x06,0x62,0x78]
            clrrdi 2, 3, 4
# CHECK-BE: rldicr. 2, 3, 0, 59             # encoding: [0x78,0x62,0x06,0xe5]
# CHECK-LE: rldicr. 2, 3, 0, 59             # encoding: [0xe5,0x06,0x62,0x78]
            clrrdi. 2, 3, 4
# CHECK-BE: rldic 2, 3, 4, 1                # encoding: [0x78,0x62,0x20,0x48]
# CHECK-LE: rldic 2, 3, 4, 1                # encoding: [0x48,0x20,0x62,0x78]
            clrlsldi 2, 3, 5, 4
# CHECK-BE: rldic. 2, 3, 4, 1               # encoding: [0x78,0x62,0x20,0x49]
# CHECK-LE: rldic. 2, 3, 4, 1               # encoding: [0x49,0x20,0x62,0x78]
            clrlsldi. 2, 3, 5, 4

# CHECK-BE: rlwinm 2, 3, 5, 0, 3            # encoding: [0x54,0x62,0x28,0x06]
# CHECK-LE: rlwinm 2, 3, 5, 0, 3            # encoding: [0x06,0x28,0x62,0x54]
            extlwi 2, 3, 4, 5
# CHECK-BE: rlwinm. 2, 3, 5, 0, 3           # encoding: [0x54,0x62,0x28,0x07]
# CHECK-LE: rlwinm. 2, 3, 5, 0, 3           # encoding: [0x07,0x28,0x62,0x54]
            extlwi. 2, 3, 4, 5
# CHECK-BE: rlwinm 2, 3, 9, 28, 31          # encoding: [0x54,0x62,0x4f,0x3e]
# CHECK-LE: rlwinm 2, 3, 9, 28, 31          # encoding: [0x3e,0x4f,0x62,0x54]
            extrwi 2, 3, 4, 5
# CHECK-BE: rlwinm. 2, 3, 9, 28, 31         # encoding: [0x54,0x62,0x4f,0x3f]
# CHECK-LE: rlwinm. 2, 3, 9, 28, 31         # encoding: [0x3f,0x4f,0x62,0x54]
            extrwi. 2, 3, 4, 5
# CHECK-BE: rlwimi 2, 3, 27, 5, 8           # encoding: [0x50,0x62,0xd9,0x50]
# CHECK-LE: rlwimi 2, 3, 27, 5, 8           # encoding: [0x50,0xd9,0x62,0x50]
            inslwi 2, 3, 4, 5
# CHECK-BE: rlwimi. 2, 3, 27, 5, 8          # encoding: [0x50,0x62,0xd9,0x51]
# CHECK-LE: rlwimi. 2, 3, 27, 5, 8          # encoding: [0x51,0xd9,0x62,0x50]
            inslwi. 2, 3, 4, 5
# CHECK-BE: rlwimi 2, 3, 23, 5, 8           # encoding: [0x50,0x62,0xb9,0x50]
# CHECK-LE: rlwimi 2, 3, 23, 5, 8           # encoding: [0x50,0xb9,0x62,0x50]
            insrwi 2, 3, 4, 5
# CHECK-BE: rlwimi. 2, 3, 23, 5, 8          # encoding: [0x50,0x62,0xb9,0x51]
# CHECK-LE: rlwimi. 2, 3, 23, 5, 8          # encoding: [0x51,0xb9,0x62,0x50]
            insrwi. 2, 3, 4, 5
# CHECK-BE: rotlwi 2, 3, 4                  # encoding: [0x54,0x62,0x20,0x3e]
# CHECK-LE: rotlwi 2, 3, 4                  # encoding: [0x3e,0x20,0x62,0x54]
            rotlwi 2, 3, 4
# CHECK-BE: rotlwi. 2, 3, 4                 # encoding: [0x54,0x62,0x20,0x3f]
# CHECK-LE: rotlwi. 2, 3, 4                 # encoding: [0x3f,0x20,0x62,0x54]
            rotlwi. 2, 3, 4
# CHECK-BE: rotlwi 2, 3, 28                 # encoding: [0x54,0x62,0xe0,0x3e]
# CHECK-LE: rotlwi 2, 3, 28                 # encoding: [0x3e,0xe0,0x62,0x54]
            rotrwi 2, 3, 4
# CHECK-BE: rotlwi. 2, 3, 28                # encoding: [0x54,0x62,0xe0,0x3f]
# CHECK-LE: rotlwi. 2, 3, 28                # encoding: [0x3f,0xe0,0x62,0x54]
            rotrwi. 2, 3, 4
# CHECK-BE: rotlw 2, 3, 4                   # encoding: [0x5c,0x62,0x20,0x3e]
# CHECK-LE: rotlw 2, 3, 4                   # encoding: [0x3e,0x20,0x62,0x5c]
            rotlw 2, 3, 4
# CHECK-BE: rotlw. 2, 3, 4                  # encoding: [0x5c,0x62,0x20,0x3f]
# CHECK-LE: rotlw. 2, 3, 4                  # encoding: [0x3f,0x20,0x62,0x5c]
            rotlw. 2, 3, 4
# CHECK-BE: slwi 2, 3, 4                    # encoding: [0x54,0x62,0x20,0x36]
# CHECK-LE: slwi 2, 3, 4                    # encoding: [0x36,0x20,0x62,0x54]
            slwi 2, 3, 4
# CHECK-BE: rlwinm. 2, 3, 4, 0, 27          # encoding: [0x54,0x62,0x20,0x37]
# CHECK-LE: rlwinm. 2, 3, 4, 0, 27          # encoding: [0x37,0x20,0x62,0x54]
            slwi. 2, 3, 4
# CHECK-BE: srwi 2, 3, 4                    # encoding: [0x54,0x62,0xe1,0x3e]
# CHECK-LE: srwi 2, 3, 4                    # encoding: [0x3e,0xe1,0x62,0x54]
            srwi 2, 3, 4
# CHECK-BE: rlwinm. 2, 3, 28, 4, 31         # encoding: [0x54,0x62,0xe1,0x3f]
# CHECK-LE: rlwinm. 2, 3, 28, 4, 31         # encoding: [0x3f,0xe1,0x62,0x54]
            srwi. 2, 3, 4
# CHECK-BE: clrlwi 2, 3, 4                  # encoding: [0x54,0x62,0x01,0x3e]
# CHECK-LE: clrlwi 2, 3, 4                  # encoding: [0x3e,0x01,0x62,0x54]
            clrlwi 2, 3, 4
# CHECK-BE: clrlwi. 2, 3, 4                 # encoding: [0x54,0x62,0x01,0x3f]
# CHECK-LE: clrlwi. 2, 3, 4                 # encoding: [0x3f,0x01,0x62,0x54]
            clrlwi. 2, 3, 4
# CHECK-BE: rlwinm 2, 3, 0, 0, 27           # encoding: [0x54,0x62,0x00,0x36]
# CHECK-LE: rlwinm 2, 3, 0, 0, 27           # encoding: [0x36,0x00,0x62,0x54]
            clrrwi 2, 3, 4
# CHECK-BE: rlwinm. 2, 3, 0, 0, 27          # encoding: [0x54,0x62,0x00,0x37]
# CHECK-LE: rlwinm. 2, 3, 0, 0, 27          # encoding: [0x37,0x00,0x62,0x54]
            clrrwi. 2, 3, 4
# CHECK-BE: rlwinm 2, 3, 4, 1, 27           # encoding: [0x54,0x62,0x20,0x76]
# CHECK-LE: rlwinm 2, 3, 4, 1, 27           # encoding: [0x76,0x20,0x62,0x54]
            clrlslwi 2, 3, 5, 4
# CHECK-BE: rlwinm. 2, 3, 4, 1, 27          # encoding: [0x54,0x62,0x20,0x77]
# CHECK-LE: rlwinm. 2, 3, 4, 1, 27          # encoding: [0x77,0x20,0x62,0x54]
            clrlslwi. 2, 3, 5, 4

# Move to/from special purpose register mnemonics

# CHECK-BE: mtxer 2                         # encoding: [0x7c,0x41,0x03,0xa6]
# CHECK-LE: mtxer 2                         # encoding: [0xa6,0x03,0x41,0x7c]
            mtxer 2
# CHECK-BE: mfxer 2                         # encoding: [0x7c,0x41,0x02,0xa6]
# CHECK-LE: mfxer 2                         # encoding: [0xa6,0x02,0x41,0x7c]
            mfxer 2
# CHECK-BE: mtudscr 2                       # encoding: [0x7c,0x43,0x03,0xa6]
# CHECK-LE: mtudscr 2                       # encoding: [0xa6,0x03,0x43,0x7c]
            mtudscr 2
# CHECK-BE: mfudscr 2                       # encoding: [0x7c,0x43,0x02,0xa6]
# CHECK-LE: mfudscr 2                       # encoding: [0xa6,0x02,0x43,0x7c]
            mfudscr 2
# CHECK-BE: mfrtcu 2                        # encoding: [0x7c,0x44,0x02,0xa6]
# CHECK-LE: mfrtcu 2                        # encoding: [0xa6,0x02,0x44,0x7c]
            mfrtcu 2
# CHECK-BE: mfrtcl 2                        # encoding: [0x7c,0x45,0x02,0xa6]
# CHECK-LE: mfrtcl 2                        # encoding: [0xa6,0x02,0x45,0x7c]
            mfrtcl 2
# CHECK-BE: mtdscr 2                        # encoding: [0x7c,0x51,0x03,0xa6]
# CHECK-LE: mtdscr 2                        # encoding: [0xa6,0x03,0x51,0x7c]
            mtdscr 2
# CHECK-BE: mfdscr 2                        # encoding: [0x7c,0x51,0x02,0xa6]
# CHECK-LE: mfdscr 2                        # encoding: [0xa6,0x02,0x51,0x7c]
            mfdscr 2
# CHECK-BE: mtdsisr 2                       # encoding: [0x7c,0x52,0x03,0xa6]
# CHECK-LE: mtdsisr 2                       # encoding: [0xa6,0x03,0x52,0x7c]
            mtdsisr 2
# CHECK-BE: mfdsisr 2                       # encoding: [0x7c,0x52,0x02,0xa6]
# CHECK-LE: mfdsisr 2                       # encoding: [0xa6,0x02,0x52,0x7c]
            mfdsisr 2
# CHECK-BE: mtdar 2                         # encoding: [0x7c,0x53,0x03,0xa6]
# CHECK-LE: mtdar 2                         # encoding: [0xa6,0x03,0x53,0x7c]
            mtdar 2
# CHECK-BE: mfdar 2                         # encoding: [0x7c,0x53,0x02,0xa6]
# CHECK-LE: mfdar 2                         # encoding: [0xa6,0x02,0x53,0x7c]
            mfdar 2
# CHECK-BE: mtdec 2                         # encoding: [0x7c,0x56,0x03,0xa6]
# CHECK-LE: mtdec 2                         # encoding: [0xa6,0x03,0x56,0x7c]
            mtdec 2
# CHECK-BE: mfdec 2                         # encoding: [0x7c,0x56,0x02,0xa6]
# CHECK-LE: mfdec 2                         # encoding: [0xa6,0x02,0x56,0x7c]
            mfdec 2
# CHECK-BE: mtsdr1 2                        # encoding: [0x7c,0x59,0x03,0xa6]
# CHECK-LE: mtsdr1 2                        # encoding: [0xa6,0x03,0x59,0x7c]
            mtsdr1 2
# CHECK-BE: mfsdr1 2                        # encoding: [0x7c,0x59,0x02,0xa6]
# CHECK-LE: mfsdr1 2                        # encoding: [0xa6,0x02,0x59,0x7c]
            mfsdr1 2
# CHECK-BE: mtsrr0 2                        # encoding: [0x7c,0x5a,0x03,0xa6]
# CHECK-LE: mtsrr0 2                        # encoding: [0xa6,0x03,0x5a,0x7c]
            mtsrr0 2
# CHECK-BE: mfsrr0 2                        # encoding: [0x7c,0x5a,0x02,0xa6]
# CHECK-LE: mfsrr0 2                        # encoding: [0xa6,0x02,0x5a,0x7c]
            mfsrr0 2
# CHECK-BE: mtsrr1 2                        # encoding: [0x7c,0x5b,0x03,0xa6]
# CHECK-LE: mtsrr1 2                        # encoding: [0xa6,0x03,0x5b,0x7c]
            mtsrr1 2
# CHECK-BE: mfsrr1 2                        # encoding: [0x7c,0x5b,0x02,0xa6]
# CHECK-LE: mfsrr1 2                        # encoding: [0xa6,0x02,0x5b,0x7c]
            mfsrr1 2
# CHECK-BE: mtcfar 2                        # encoding: [0x7c,0x5c,0x03,0xa6]
# CHECK-LE: mtcfar 2                        # encoding: [0xa6,0x03,0x5c,0x7c]
            mtcfar 2
# CHECK-BE: mfcfar 2                        # encoding: [0x7c,0x5c,0x02,0xa6]
# CHECK-LE: mfcfar 2                        # encoding: [0xa6,0x02,0x5c,0x7c]
            mfcfar 2
# CHECK-BE: mtamr 2                         # encoding: [0x7c,0x5d,0x03,0xa6]
# CHECK-LE: mtamr 2                         # encoding: [0xa6,0x03,0x5d,0x7c]
            mtamr 2
# CHECK-BE: mfamr 2                         # encoding: [0x7c,0x5d,0x02,0xa6]
# CHECK-LE: mfamr 2                         # encoding: [0xa6,0x02,0x5d,0x7c]
            mfamr 2
# CHECK-BE: mtpid 2                         # encoding: [0x7c,0x50,0x0b,0xa6]
# CHECK-LE: mtpid 2                         # encoding: [0xa6,0x0b,0x50,0x7c]
            mtpid 2
# CHECK-BE: mfpid 2                         # encoding: [0x7c,0x50,0x0a,0xa6]
# CHECK-LE: mfpid 2                         # encoding: [0xa6,0x0a,0x50,0x7c]
            mfpid 2
# CHECK-BE: mtlr 2                          # encoding: [0x7c,0x48,0x03,0xa6]
# CHECK-LE: mtlr 2                          # encoding: [0xa6,0x03,0x48,0x7c]
            mtlr 2
# CHECK-BE: mflr 2                          # encoding: [0x7c,0x48,0x02,0xa6]
# CHECK-LE: mflr 2                          # encoding: [0xa6,0x02,0x48,0x7c]
            mflr 2
# CHECK-BE: mtctr 2                         # encoding: [0x7c,0x49,0x03,0xa6]
# CHECK-LE: mtctr 2                         # encoding: [0xa6,0x03,0x49,0x7c]
            mtctr 2
# CHECK-BE: mfctr 2                         # encoding: [0x7c,0x49,0x02,0xa6]
# CHECK-LE: mfctr 2                         # encoding: [0xa6,0x02,0x49,0x7c]
            mfctr 2
# CHECK-BE: mtuamr 2                        # encoding: [0x7c,0x4d,0x03,0xa6]
# CHECK-LE: mtuamr 2                        # encoding: [0xa6,0x03,0x4d,0x7c]
            mtuamr 2
# CHECK-BE: mfuamr 2                        # encoding: [0x7c,0x4d,0x02,0xa6]
# CHECK-LE: mfuamr 2                        # encoding: [0xa6,0x02,0x4d,0x7c]
            mfuamr 2
# CHECK-BE: mtppr 2                         # encoding: [0x7c,0x40,0xe3,0xa6]
# CHECK-LE: mtppr 2                         # encoding: [0xa6,0xe3,0x40,0x7c]
            mtppr 2
# CHECK-BE: mfppr 2                         # encoding: [0x7c,0x40,0xe2,0xa6]
# CHECK-LE: mfppr 2                         # encoding: [0xa6,0xe2,0x40,0x7c]
            mfppr 2
# CHECK-BE: mfvrsave 2                      # encoding: [0x7c,0x40,0x42,0xa6]
# CHECK-LE: mfvrsave 2                      # encoding: [0xa6,0x42,0x40,0x7c]
            mfvrsave 2
# CHECK-BE: mtvrsave 2                      # encoding: [0x7c,0x40,0x43,0xa6]
# CHECK-LE: mtvrsave 2                      # encoding: [0xa6,0x43,0x40,0x7c]
            mtvrsave 2

# Miscellaneous mnemonics

# CHECK-BE: nop                             # encoding: [0x60,0x00,0x00,0x00]
# CHECK-LE: nop                             # encoding: [0x00,0x00,0x00,0x60]
            nop
# CHECK-BE: xnop                            # encoding: [0x68,0x00,0x00,0x00]
# CHECK-LE: xnop                            # encoding: [0x00,0x00,0x00,0x68]
            xnop
# CHECK-BE: li 2, 128                       # encoding: [0x38,0x40,0x00,0x80]
# CHECK-LE: li 2, 128                       # encoding: [0x80,0x00,0x40,0x38]
            li 2, 128
# CHECK-BE: lis 2, 128                      # encoding: [0x3c,0x40,0x00,0x80]
# CHECK-LE: lis 2, 128                      # encoding: [0x80,0x00,0x40,0x3c]
            lis 2, 128
# CHECK-BE: la 2, 128(4)
# CHECK-LE: la 2, 128(4)
            la 2, 128(4)
# CHECK-BE: mr 2, 3                         # encoding: [0x7c,0x62,0x1b,0x78]
# CHECK-LE: mr 2, 3                         # encoding: [0x78,0x1b,0x62,0x7c]
            mr 2, 3
# CHECK-BE: mr. 2, 3                        # encoding: [0x7c,0x62,0x1b,0x79]
# CHECK-LE: mr. 2, 3                        # encoding: [0x79,0x1b,0x62,0x7c]
            mr. 2, 3
# CHECK-BE: not 2, 3                        # encoding: [0x7c,0x62,0x18,0xf8]
# CHECK-LE: not 2, 3                        # encoding: [0xf8,0x18,0x62,0x7c]
            not 2, 3
# CHECK-BE: not. 2, 3                       # encoding: [0x7c,0x62,0x18,0xf9]
# CHECK-LE: not. 2, 3                       # encoding: [0xf9,0x18,0x62,0x7c]
            not. 2, 3
# CHECK-BE: mtcr 2                          # encoding: [0x7c,0x4f,0xf1,0x20]
# CHECK-LE: mtcr 2                          # encoding: [0x20,0xf1,0x4f,0x7c]
            mtcr 2

# CHECK-BE: mfspr 4, 272                    # encoding: [0x7c,0x90,0x42,0xa6]
# CHECK-LE: mfspr 4, 272                    # encoding: [0xa6,0x42,0x90,0x7c]
            mfsprg %r4, 0

# CHECK-BE: mfspr 4, 273                    # encoding: [0x7c,0x91,0x42,0xa6]
# CHECK-LE: mfspr 4, 273                    # encoding: [0xa6,0x42,0x91,0x7c]
            mfsprg %r4, 1

# CHECK-BE: mfspr 4, 274                    # encoding: [0x7c,0x92,0x42,0xa6]
# CHECK-LE: mfspr 4, 274                    # encoding: [0xa6,0x42,0x92,0x7c]
            mfsprg %r4, 2

# CHECK-BE: mfspr 4, 275                    # encoding: [0x7c,0x93,0x42,0xa6]
# CHECK-LE: mfspr 4, 275                    # encoding: [0xa6,0x42,0x93,0x7c]
            mfsprg %r4, 3

# CHECK-BE: mfspr 2, 272                    # encoding: [0x7c,0x50,0x42,0xa6]
# CHECK-LE: mfspr 2, 272                    # encoding: [0xa6,0x42,0x50,0x7c]
            mfsprg0 %r2
# CHECK-BE: mfspr 2, 273                    # encoding: [0x7c,0x51,0x42,0xa6]
# CHECK-LE: mfspr 2, 273                    # encoding: [0xa6,0x42,0x51,0x7c]
            mfsprg1 %r2
# CHECK-BE: mfspr 2, 274                    # encoding: [0x7c,0x52,0x42,0xa6]
# CHECK-LE: mfspr 2, 274                    # encoding: [0xa6,0x42,0x52,0x7c]
            mfsprg2 %r2
# CHECK-BE: mfspr 2, 275                    # encoding: [0x7c,0x53,0x42,0xa6]
# CHECK-LE: mfspr 2, 275                    # encoding: [0xa6,0x42,0x53,0x7c]
            mfsprg3 %r2

# CHECK-BE: mtspr 272, 4                    # encoding: [0x7c,0x90,0x43,0xa6]
# CHECK-LE: mtspr 272, 4                    # encoding: [0xa6,0x43,0x90,0x7c]
            mtsprg 0, %r4

# CHECK-BE: mtspr 273, 4                    # encoding: [0x7c,0x91,0x43,0xa6]
# CHECK-LE: mtspr 273, 4                    # encoding: [0xa6,0x43,0x91,0x7c]
            mtsprg 1, %r4

# CHECK-BE: mtspr 274, 4                    # encoding: [0x7c,0x92,0x43,0xa6]
# CHECK-LE: mtspr 274, 4                    # encoding: [0xa6,0x43,0x92,0x7c]
            mtsprg 2, %r4

# CHECK-BE: mtspr 275, 4                    # encoding: [0x7c,0x93,0x43,0xa6]
# CHECK-LE: mtspr 275, 4                    # encoding: [0xa6,0x43,0x93,0x7c]
            mtsprg 3, %r4

# CHECK-BE: mtspr 272, 4                    # encoding: [0x7c,0x90,0x43,0xa6]
# CHECK-LE: mtspr 272, 4                    # encoding: [0xa6,0x43,0x90,0x7c]
            mtsprg0 %r4

# CHECK-BE: mtspr 273, 4                    # encoding: [0x7c,0x91,0x43,0xa6]
# CHECK-LE: mtspr 273, 4                    # encoding: [0xa6,0x43,0x91,0x7c]
            mtsprg1 %r4

# CHECK-BE: mtspr 274, 4                    # encoding: [0x7c,0x92,0x43,0xa6]
# CHECK-LE: mtspr 274, 4                    # encoding: [0xa6,0x43,0x92,0x7c]
            mtsprg2 %r4

# CHECK-BE: mtspr 275, 4                    # encoding: [0x7c,0x93,0x43,0xa6]
# CHECK-LE: mtspr 275, 4                    # encoding: [0xa6,0x43,0x93,0x7c]
            mtsprg3 %r4

# Altivec Data Stream instruction:
# CHECK-BE: dss 3                            # encoding: [0x7c,0x60,0x06,0x6c]
# CHECK-LE: dss 3                            # encoding: [0x6c,0x06,0x60,0x7c]
            dss 3
# CHECK-BE: dssall                           # encoding: [0x7e,0x00,0x06,0x6c]
# CHECK-LE: dssall                           # encoding: [0x6c,0x06,0x00,0x7e]
            dssall
# CHECK-BE: dst 12, 11, 3                    # encoding: [0x7c,0x6c,0x5a,0xac]
# CHECK-LE: dst 12, 11, 3                    # encoding: [0xac,0x5a,0x6c,0x7c]
            dst %r12, %r11, 3
# CHECK-BE: dstt 12, 11, 3                   # encoding: [0x7e,0x6c,0x5a,0xac]
# CHECK-LE: dstt 12, 11, 3                   # encoding: [0xac,0x5a,0x6c,0x7e]
            dstt %r12, %r11, 3
# CHECK-BE: dstst 12, 11, 3                  # encoding: [0x7c,0x6c,0x5a,0xec]
# CHECK-LE: dstst 12, 11, 3                  # encoding: [0xec,0x5a,0x6c,0x7c]
            dstst %r12, %r11, 3
# CHECK-BE: dststt 12, 11, 3                 # encoding: [0x7e,0x6c,0x5a,0xec]
# CHECK-LE: dststt 12, 11, 3                 # encoding: [0xec,0x5a,0x6c,0x7e]
            dststt %r12, %r11, 3

# CHECK-BE: tlbia                            # encoding: [0x7c,0x00,0x02,0xe4]
# CHECK-LE: tlbia                            # encoding: [0xe4,0x02,0x00,0x7c]
            tlbia

# CHECK-BE: lswi 8, 6, 7                     # encoding: [0x7d,0x06,0x3c,0xaa]
# CHECK-LE: lswi 8, 6, 7                     # encoding: [0xaa,0x3c,0x06,0x7d]
            lswi %r8, %r6, 7
# CHECK-BE: stswi 8, 6, 7                    # encoding: [0x7d,0x06,0x3d,0xaa]
# CHECK-LE: stswi 8, 6, 7                    # encoding: [0xaa,0x3d,0x06,0x7d]
            stswi %r8, %r6, 7

# CHECK-BE: rfid                             # encoding: [0x4c,0x00,0x00,0x24]
# CHECK-LE: rfid                             # encoding: [0x24,0x00,0x00,0x4c]
            rfid

# CHECK-BE: mfasr 2                          # encoding: [0x7c,0x58,0x42,0xa6]
# CHECK-LE: mfasr 2                          # encoding: [0xa6,0x42,0x58,0x7c]
            mfasr 2
# CHECK-BE: mtasr 2                          # encoding: [0x7c,0x58,0x43,0xa6]
# CHECK-LE: mtasr 2                          # encoding: [0xa6,0x43,0x58,0x7c]
            mtasr 2

# Load and Store Caching Inhibited Instructions
# CHECK-BE: lbzcix 21, 5, 7                  # encoding: [0x7e,0xa5,0x3e,0xaa]
# CHECK-LE: lbzcix 21, 5, 7                  # encoding: [0xaa,0x3e,0xa5,0x7e]
            lbzcix 21, 5, 7
# CHECK-BE: lhzcix 21, 5, 7                  # encoding: [0x7e,0xa5,0x3e,0x6a]
# CHECK-LE: lhzcix 21, 5, 7                  # encoding: [0x6a,0x3e,0xa5,0x7e]
            lhzcix 21, 5, 7
# CHECK-BE: lwzcix 21, 5, 7                  # encoding: [0x7e,0xa5,0x3e,0x2a]
# CHECK-LE: lwzcix 21, 5, 7                  # encoding: [0x2a,0x3e,0xa5,0x7e]
            lwzcix 21, 5, 7
# CHECK-BE: ldcix  21, 5, 7                  # encoding: [0x7e,0xa5,0x3e,0xea]
# CHECK-LE: ldcix  21, 5, 7                  # encoding: [0xea,0x3e,0xa5,0x7e]
            ldcix  21, 5, 7
            
# CHECK-BE: stbcix 21, 5, 7                  # encoding: [0x7e,0xa5,0x3f,0xaa]
# CHECK-LE: stbcix 21, 5, 7                  # encoding: [0xaa,0x3f,0xa5,0x7e]
            stbcix 21, 5, 7
# CHECK-BE: sthcix 21, 5, 7                  # encoding: [0x7e,0xa5,0x3f,0x6a]
# CHECK-LE: sthcix 21, 5, 7                  # encoding: [0x6a,0x3f,0xa5,0x7e]
            sthcix 21, 5, 7
# CHECK-BE: stwcix 21, 5, 7                  # encoding: [0x7e,0xa5,0x3f,0x2a]
# CHECK-LE: stwcix 21, 5, 7                  # encoding: [0x2a,0x3f,0xa5,0x7e]
            stwcix 21, 5, 7
# CHECK-BE: stdcix 21, 5, 7                  # encoding: [0x7e,0xa5,0x3f,0xea]
# CHECK-LE: stdcix 21, 5, 7                  # encoding: [0xea,0x3f,0xa5,0x7e]
            stdcix 21, 5, 7

# Processor-Specific Instructions
# CHECK-BE: attn                             # encoding: [0x00,0x00,0x02,0x00]
# CHECK-LE: attn                             # encoding: [0x00,0x02,0x00,0x00]
            attn

# Copy-Paste Facility (Extended Mnemonics):
# CHECK-BE: copy 2, 19, 0                      # encoding: [0x7c,0x02,0x9e,0x0c]
# CHECK-LE: copy 2, 19, 0                      # encoding: [0x0c,0x9e,0x02,0x7c]
            copy 2, 19
# CHECK-BE: copy 2, 19, 1                      # encoding: [0x7c,0x22,0x9e,0x0c]
# CHECK-LE: copy 2, 19, 1                      # encoding: [0x0c,0x9e,0x22,0x7c]
            copy_first 2, 19
# CHECK-BE: paste 17, 1, 0                     # encoding: [0x7c,0x11,0x0f,0x0c]
# CHECK-LE: paste 17, 1, 0                     # encoding: [0x0c,0x0f,0x11,0x7c]
            paste 17, 1
# CHECK-BE: paste. 17, 1, 1                    # encoding: [0x7c,0x31,0x0f,0x0d]
# CHECK-LE: paste. 17, 1, 1                    # encoding: [0x0d,0x0f,0x31,0x7c]
            paste_last 17, 1
