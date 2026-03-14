
# RUN: llvm-mc -triple powerpc64-unknown-unknown --show-encoding %s | FileCheck -check-prefix=CHECK-BE %s
# RUN: llvm-mc -triple powerpc64le-unknown-unknown --show-encoding %s | FileCheck -check-prefix=CHECK-LE %s

# RUN: llvm-mc -triple powerpc64-unknown-unknown -filetype=obj %s | \
# RUN: llvm-readobj -r - | FileCheck %s -check-prefix=CHECK-BE-REL
# RUN: llvm-mc -triple powerpc64le-unknown-unknown -filetype=obj %s | \
# RUN: llvm-readobj -r - | FileCheck %s -check-prefix=CHECK-LE-REL

# GOT references must result in explicit relocations
# even if the target symbol is local.

target:

# CHECK-BE: addi 4, 3, target@GOT           # encoding: [0x38,0x83,A,A]
# CHECK-LE: addi 4, 3, target@GOT           # encoding: [A,A,0x83,0x38]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@GOT, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@GOT, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_GOT16 target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_GOT16 target 0x0
            addi 4, 3, target@got

# CHECK-BE: ld 1, target@GOT(2)             # encoding: [0xe8,0x22,A,0bAAAAAA00]
# CHECK-LE: ld 1, target@GOT(2)             # encoding: [0bAAAAAA00,A,0x22,0xe8]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@GOT, kind: fixup_ppc_half16ds
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@GOT, kind: fixup_ppc_half16ds
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_GOT16_DS target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_GOT16_DS target 0x0
            ld 1, target@got(2)

# CHECK-BE: addis 3, 2, target@got@ha       # encoding: [0x3c,0x62,A,A]
# CHECK-LE: addis 3, 2, target@got@ha       # encoding: [A,A,0x62,0x3c]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@got@ha, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@got@ha, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_GOT16_HA target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_GOT16_HA target 0x0
            addis 3, 2, target@got@ha

# CHECK-BE: addi 4, 3, target@got@l         # encoding: [0x38,0x83,A,A]
# CHECK-LE: addi 4, 3, target@got@l         # encoding: [A,A,0x83,0x38]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@got@l, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@got@l, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_GOT16_LO target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_GOT16_LO target 0x0
            addi 4, 3, target@got@l

# CHECK-BE: addis 3, 2, target@got@h        # encoding: [0x3c,0x62,A,A]
# CHECK-LE: addis 3, 2, target@got@h        # encoding: [A,A,0x62,0x3c]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@got@h, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@got@h, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_GOT16_HI target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_GOT16_HI target 0x0
            addis 3, 2, target@got@h

# CHECK-BE: lwz 1, target@got@l(3)          # encoding: [0x80,0x23,A,A]
# CHECK-LE: lwz 1, target@got@l(3)          # encoding: [A,A,0x23,0x80]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@got@l, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@got@l, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_GOT16_LO target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_GOT16_LO target 0x0
            lwz 1, target@got@l(3)

# CHECK-BE: ld 1, target@got@l(3)           # encoding: [0xe8,0x23,A,0bAAAAAA00]
# CHECK-LE: ld 1, target@got@l(3)           # encoding: [0bAAAAAA00,A,0x23,0xe8]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@got@l, kind: fixup_ppc_half16ds
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@got@l, kind: fixup_ppc_half16ds
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_GOT16_LO_DS target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_GOT16_LO_DS target 0x0
            ld 1, target@got@l(3)

