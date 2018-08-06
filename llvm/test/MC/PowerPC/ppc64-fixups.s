
# RUN: llvm-mc -triple powerpc64-unknown-unknown --show-encoding %s | FileCheck -check-prefix=CHECK-BE %s
# RUN: llvm-mc -triple powerpc64le-unknown-unknown --show-encoding %s | FileCheck -check-prefix=CHECK-LE %s

# RUN: llvm-mc -triple powerpc64-unknown-unknown -filetype=obj %s | \
# RUN: llvm-readobj -r | FileCheck %s -check-prefix=CHECK-BE-REL
# RUN: llvm-mc -triple powerpc64le-unknown-unknown -filetype=obj %s | \
# RUN: llvm-readobj -r | FileCheck %s -check-prefix=CHECK-LE-REL

# CHECK-BE: b target                        # encoding: [0b010010AA,A,A,0bAAAAAA00]
# CHECK-LE: b target                        # encoding: [0bAAAAAA00,A,A,0b010010AA]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_br24
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_br24
# CHECK-BE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_REL24 target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_REL24 target 0x0
            b target

# CHECK-BE: ba target                       # encoding: [0b010010AA,A,A,0bAAAAAA10]
# CHECK-LE: ba target                       # encoding: [0bAAAAAA10,A,A,0b010010AA]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_br24abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_br24abs
# CHECK-BE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_ADDR24 target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_ADDR24 target 0x0
            ba target

# CHECK-BE: beq 0, target                   # encoding: [0x41,0x82,A,0bAAAAAA00]
# CHECK-LE: beq 0, target                   # encoding: [0bAAAAAA00,A,0x82,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-BE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_REL14 target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_REL14 target 0x0
            beq target

# CHECK-BE: beqa 0, target                  # encoding: [0x41,0x82,A,0bAAAAAA10]
# CHECK-LE: beqa 0, target                  # encoding: [0bAAAAAA10,A,0x82,0x41]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-BE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_ADDR14 target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_ADDR14 target 0x0
            beqa target


# CHECK-BE: li 3, target@l                  # encoding: [0x38,0x60,A,A]
# CHECK-LE: li 3, target@l                  # encoding: [A,A,0x60,0x38]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@l, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@l, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_ADDR16_LO target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_ADDR16_LO target 0x0
            li 3, target@l

# CHECK-BE: addis 3, 3, target@ha           # encoding: [0x3c,0x63,A,A]
# CHECK-LE: addis 3, 3, target@ha           # encoding: [A,A,0x63,0x3c]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@ha, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@ha, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_ADDR16_HA target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_ADDR16_HA target 0x0
            addis 3, 3, target@ha

# CHECK-BE: addis 3, 3, target@higha        # encoding: [0x3c,0x63,A,A]
# CHECK-LE: addis 3, 3, target@higha        # encoding: [A,A,0x63,0x3c]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@higha, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@higha, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_ADDR16_HIGHA target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_ADDR16_HIGHA target 0x0

            addis 3, 3, target@higha

# CHECK-BE: lis 3, target@ha                # encoding: [0x3c,0x60,A,A]
# CHECK-LE: lis 3, target@ha                # encoding: [A,A,0x60,0x3c]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@ha, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@ha, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_ADDR16_HA target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_ADDR16_HA target 0x0
            lis 3, target@ha

# CHECK-BE: addi 4, 3, target@l             # encoding: [0x38,0x83,A,A]
# CHECK-LE: addi 4, 3, target@l             # encoding: [A,A,0x83,0x38]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@l, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@l, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_ADDR16_LO target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_ADDR16_LO target 0x0
            addi 4, 3, target@l

# CHECK-BE: li 3, target@ha                 # encoding: [0x38,0x60,A,A]
# CHECK-LE: li 3, target@ha                 # encoding: [A,A,0x60,0x38]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@ha, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@ha, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_ADDR16_HA target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_ADDR16_HA target 0x0
            li 3, target@ha

# CHECK-BE: lis 3, target@l                 # encoding: [0x3c,0x60,A,A]
# CHECK-LE: lis 3, target@l                 # encoding: [A,A,0x60,0x3c]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@l, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@l, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_ADDR16_LO target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_ADDR16_LO target 0x0
            lis 3, target@l

# CHECK-BE: li 3, target@h                  # encoding: [0x38,0x60,A,A]
# CHECK-LE: li 3, target@h                  # encoding: [A,A,0x60,0x38]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@h, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@h, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_ADDR16_HI target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_ADDR16_HI target 0x0
            li 3, target@h

# CHECK-BE: lis 3, target@h                  # encoding: [0x3c,0x60,A,A]
# CHECK-LE: lis 3, target@h                  # encoding: [A,A,0x60,0x3c]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@h, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@h, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_ADDR16_HI target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_ADDR16_HI target 0x0
            lis 3, target@h

# CHECK-BE: li 3, target@higher             # encoding: [0x38,0x60,A,A]
# CHECK-LE: li 3, target@higher             # encoding: [A,A,0x60,0x38]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@higher, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@higher, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_ADDR16_HIGHER target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_ADDR16_HIGHER target 0x0
            li 3, target@higher

# CHECK-BE: lis 3, target@highest           # encoding: [0x3c,0x60,A,A]
# CHECK-LE: lis 3, target@highest           # encoding: [A,A,0x60,0x3c]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@highest, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@highest, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_ADDR16_HIGHEST target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_ADDR16_HIGHEST target 0x0
            lis 3, target@highest

# CHECK-BE: li 3, target@highera            # encoding: [0x38,0x60,A,A]
# CHECK-LE: li 3, target@highera            # encoding: [A,A,0x60,0x38]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@highera, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@highera, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_ADDR16_HIGHERA target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_ADDR16_HIGHERA target 0x0
            li 3, target@highera

# CHECK-BE: lis 3, target@highesta          # encoding: [0x3c,0x60,A,A]
# CHECK-LE: lis 3, target@highesta          # encoding: [A,A,0x60,0x3c]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@highesta, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@highesta, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_ADDR16_HIGHESTA target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_ADDR16_HIGHESTA target 0x0
            lis 3, target@highesta

# CHECK-BE: lwz 1, target@l(3)              # encoding: [0x80,0x23,A,A]
# CHECK-LE: lwz 1, target@l(3)              # encoding: [A,A,0x23,0x80]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@l, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@l, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_ADDR16_LO target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_ADDR16_LO target 0x0
            lwz 1, target@l(3)

# CHECK-BE: lwz 1, target(3)                # encoding: [0x80,0x23,A,A]
# CHECK-LE: lwz 1, target(3)                # encoding: [A,A,0x23,0x80]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_ADDR16 target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_ADDR16 target 0x0
            lwz 1, target(3)

# CHECK-BE: ld 1, target@l(3)               # encoding: [0xe8,0x23,A,0bAAAAAA00]
# CHECK-LE: ld 1, target@l(3)               # encoding: [0bAAAAAA00,A,0x23,0xe8]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@l, kind: fixup_ppc_half16ds
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@l, kind: fixup_ppc_half16ds
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_ADDR16_LO_DS target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_ADDR16_LO_DS target 0x0
            ld 1, target@l(3)

# CHECK-BE: ld 1, target(3)                 # encoding: [0xe8,0x23,A,0bAAAAAA00]
# CHECK-LE: ld 1, target(3)                 # encoding: [0bAAAAAA00,A,0x23,0xe8]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target, kind: fixup_ppc_half16ds
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_half16ds
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_ADDR16_DS target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_ADDR16_DS target 0x0
            ld 1, target(3)

base:
# CHECK-BE: lwz 1, target-base(3)           # encoding: [0x80,0x23,A,A]
# CHECK-LE: lwz 1, target-base(3)           # encoding: [A,A,0x23,0x80]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target-base, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target-base, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_REL16 target 0x2
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_REL16 target 0x0
            lwz 1, target-base(3)

# CHECK-BE: li 3, target-base@h             # encoding: [0x38,0x60,A,A]
# CHECK-LE: li 3, target-base@h             # encoding: [A,A,0x60,0x38]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target-base@h, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target-base@h, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_REL16_HI target 0x6
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_REL16_HI target 0x4
            li 3, target-base@h

# CHECK-BE: li 3, target-base@l             # encoding: [0x38,0x60,A,A]
# CHECK-LE: li 3, target-base@l             # encoding: [A,A,0x60,0x38]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target-base@l, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target-base@l, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_REL16_LO target 0xA
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_REL16_LO target 0x8
            li 3, target-base@l

# CHECK-BE: li 3, target-base@ha            # encoding: [0x38,0x60,A,A]
# CHECK-LE: li 3, target-base@ha            # encoding: [A,A,0x60,0x38]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target-base@ha, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target-base@ha, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_REL16_HA target 0xE
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_REL16_HA target 0xC
            li 3, target-base@ha

# CHECK-BE: ori 3, 3, target@l              # encoding: [0x60,0x63,A,A]
# CHECK-LE: ori 3, 3, target@l              # encoding: [A,A,0x63,0x60]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@l, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@l, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_ADDR16_LO target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_ADDR16_LO target 0x0
            ori 3, 3, target@l

# CHECK-BE: oris 3, 3, target@h             # encoding: [0x64,0x63,A,A]
# CHECK-LE: oris 3, 3, target@h             # encoding: [A,A,0x63,0x64]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@h, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@h, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_ADDR16_HI target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_ADDR16_HI target 0x0
            oris 3, 3, target@h

# CHECK-BE:  oris 3, 3, target@high         # encoding: [0x64,0x63,A,A]
# CHECK-LE:  oris 3, 3, target@high         # encoding: [A,A,0x63,0x64]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@high, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@high, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_ADDR16_HIGH target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_ADDR16_HIGH target 0x0
            oris 3, 3, target@high

# CHECK-BE: ld 1, target@toc(2)             # encoding: [0xe8,0x22,A,0bAAAAAA00]
# CHECK-LE: ld 1, target@toc(2)             # encoding: [0bAAAAAA00,A,0x22,0xe8]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@toc, kind: fixup_ppc_half16ds
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@toc, kind: fixup_ppc_half16ds
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_TOC16_DS target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_TOC16_DS target 0x0
            ld 1, target@toc(2)

# CHECK-BE: addis 3, 2, target@toc@ha       # encoding: [0x3c,0x62,A,A]
# CHECK-LE: addis 3, 2, target@toc@ha       # encoding: [A,A,0x62,0x3c]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@toc@ha, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@toc@ha, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_TOC16_HA target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_TOC16_HA target 0x0
            addis 3, 2, target@toc@ha

# CHECK-BE: addi 4, 3, target@toc@l         # encoding: [0x38,0x83,A,A]
# CHECK-LE: addi 4, 3, target@toc@l         # encoding: [A,A,0x83,0x38]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@toc@l, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@toc@l, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_TOC16_LO target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_TOC16_LO target 0x0
            addi 4, 3, target@toc@l

# CHECK-BE: addis 3, 2, target@toc@h        # encoding: [0x3c,0x62,A,A]
# CHECK-LE: addis 3, 2, target@toc@h        # encoding: [A,A,0x62,0x3c]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@toc@h, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@toc@h, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_TOC16_HI target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_TOC16_HI target 0x0
            addis 3, 2, target@toc@h

# CHECK-BE: lwz 1, target@toc@l(3)          # encoding: [0x80,0x23,A,A]
# CHECK-LE: lwz 1, target@toc@l(3)          # encoding: [A,A,0x23,0x80]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@toc@l, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@toc@l, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_TOC16_LO target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_TOC16_LO target 0x0
            lwz 1, target@toc@l(3)

# CHECK-BE: ld 1, target@toc@l(3)           # encoding: [0xe8,0x23,A,0bAAAAAA00]
# CHECK-LE: ld 1, target@toc@l(3)           # encoding: [0bAAAAAA00,A,0x23,0xe8]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@toc@l, kind: fixup_ppc_half16ds
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@toc@l, kind: fixup_ppc_half16ds
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_TOC16_LO_DS target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_TOC16_LO_DS target 0x0
            ld 1, target@toc@l(3)

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

# CHECK-BE: addis 3, 2, target@tprel@ha     # encoding: [0x3c,0x62,A,A]
# CHECK-LE: addis 3, 2, target@tprel@ha     # encoding: [A,A,0x62,0x3c]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@tprel@ha, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@tprel@ha, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_TPREL16_HA target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_TPREL16_HA target 0x0
            addis 3, 2, target@tprel@ha

# CHECK-BE: addis 3, 2, target@tprel@higha  # encoding: [0x3c,0x62,A,A]
# CHECK-LE: addis 3, 2, target@tprel@higha  # encoding: [A,A,0x62,0x3c]
# CHECK-BE-NEXT:                            # fixup A - offset: 2, value: target@tprel@higha, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            # fixup A - offset: 0, value: target@tprel@higha, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_TPREL16_HIGHA target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_TPREL16_HIGHA target 0x0
            addis 3, 2, target@tprel@higha

# CHECK-BE: addis 3, 2, target@tprel@high   # encoding: [0x3c,0x62,A,A]
# CHECK-LE: addis 3, 2, target@tprel@high   # encoding: [A,A,0x62,0x3c]
# CHECK-BE-NEXT:                            # fixup A - offset: 2, value: target@tprel@high, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            # fixup A - offset: 0, value: target@tprel@high, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_TPREL16_HIGH target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_TPREL16_HIGH target 0x0
            addis 3, 2, target@tprel@high

# CHECK-BE: addi 3, 3, target@tprel@l       # encoding: [0x38,0x63,A,A]
# CHECK-LE: addi 3, 3, target@tprel@l       # encoding: [A,A,0x63,0x38]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@tprel@l, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@tprel@l, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_TPREL16_LO target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_TPREL16_LO target 0x0
            addi 3, 3, target@tprel@l

# CHECK-BE: addi 3, 3, target@TPREL         # encoding: [0x38,0x63,A,A]
# CHECK-LE: addi 3, 3, target@TPREL         # encoding: [A,A,0x63,0x38]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@TPREL, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@TPREL, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_TPREL16 target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_TPREL16 target 0x0
            addi 3, 3, target@tprel

# CHECK-BE: addi 3, 3, target@tprel@h       # encoding: [0x38,0x63,A,A]
# CHECK-LE: addi 3, 3, target@tprel@h       # encoding: [A,A,0x63,0x38]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@tprel@h, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@tprel@h, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_TPREL16_HI target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_TPREL16_HI target 0x0
            addi 3, 3, target@tprel@h

# CHECK-BE: addi 3, 3, target@tprel@higher  # encoding: [0x38,0x63,A,A]
# CHECK-LE: addi 3, 3, target@tprel@higher  # encoding: [A,A,0x63,0x38]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@tprel@higher, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@tprel@higher, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_TPREL16_HIGHER target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_TPREL16_HIGHER target 0x0
            addi 3, 3, target@tprel@higher

# CHECK-BE: addis 3, 2, target@tprel@highest # encoding: [0x3c,0x62,A,A]
# CHECK-LE: addis 3, 2, target@tprel@highest # encoding: [A,A,0x62,0x3c]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@tprel@highest, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@tprel@highest, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_TPREL16_HIGHEST target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_TPREL16_HIGHEST target 0x0
            addis 3, 2, target@tprel@highest

# CHECK-BE: addi 3, 3, target@tprel@highera  # encoding: [0x38,0x63,A,A]
# CHECK-LE: addi 3, 3, target@tprel@highera  # encoding: [A,A,0x63,0x38]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@tprel@highera, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@tprel@highera, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_TPREL16_HIGHERA target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_TPREL16_HIGHERA target 0x0
            addi 3, 3, target@tprel@highera

# CHECK-BE: addis 3, 2, target@tprel@highesta # encoding: [0x3c,0x62,A,A]
# CHECK-LE: addis 3, 2, target@tprel@highesta # encoding: [A,A,0x62,0x3c]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@tprel@highesta, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@tprel@highesta, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_TPREL16_HIGHESTA target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_TPREL16_HIGHESTA target 0x0
            addis 3, 2, target@tprel@highesta

# CHECK-BE: ld 1, target@tprel@l(3)         # encoding: [0xe8,0x23,A,0bAAAAAA00]
# CHECK-LE: ld 1, target@tprel@l(3)         # encoding: [0bAAAAAA00,A,0x23,0xe8]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@tprel@l, kind: fixup_ppc_half16ds
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@tprel@l, kind: fixup_ppc_half16ds
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_TPREL16_LO_DS target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_TPREL16_LO_DS target 0x0
            ld 1, target@tprel@l(3)

# CHECK-BE: ld 1, target@TPREL(3)           # encoding: [0xe8,0x23,A,0bAAAAAA00]
# CHECK-LE: ld 1, target@TPREL(3)           # encoding: [0bAAAAAA00,A,0x23,0xe8]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@TPREL, kind: fixup_ppc_half16ds
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@TPREL, kind: fixup_ppc_half16ds
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_TPREL16_DS target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_TPREL16_DS target 0x0
            ld 1, target@tprel(3)

# CHECK-BE: addis 3, 2, target@dtprel@ha    # encoding: [0x3c,0x62,A,A]
# CHECK-LE: addis 3, 2, target@dtprel@ha    # encoding: [A,A,0x62,0x3c]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@dtprel@ha, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@dtprel@ha, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_DTPREL16_HA target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_DTPREL16_HA target 0x0
            addis 3, 2, target@dtprel@ha

# CHECK-BE: addis 3, 2, target@dtprel@higha # encoding: [0x3c,0x62,A,A]
# CHECK-LE: addis 3, 2, target@dtprel@higha # encoding: [A,A,0x62,0x3c]
# CHECK-BE-NEXT:                            # fixup A - offset: 2, value: target@dtprel@higha, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            # fixup A - offset: 0, value: target@dtprel@higha, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_DTPREL16_HIGHA target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_DTPREL16_HIGHA target 0x0
            addis 3, 2, target@dtprel@higha

# CHECK-BE: addis 3, 2, target@dtprel@high  # encoding: [0x3c,0x62,A,A]
# CHECK-LE: addis 3, 2, target@dtprel@high  # encoding: [A,A,0x62,0x3c]
# CHECK-BE-NEXT:                            # fixup A - offset: 2, value: target@dtprel@high, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            # fixup A - offset: 0, value: target@dtprel@high, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_DTPREL16_HIGH target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_DTPREL16_HIGH target 0x0
            addis 3, 2, target@dtprel@high

# CHECK-BE: addi 3, 3, target@dtprel@l      # encoding: [0x38,0x63,A,A]
# CHECK-LE: addi 3, 3, target@dtprel@l      # encoding: [A,A,0x63,0x38]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@dtprel@l, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@dtprel@l, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_DTPREL16_LO target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_DTPREL16_LO target 0x0
            addi 3, 3, target@dtprel@l

# CHECK-BE: addi 3, 3, target@DTPREL         # encoding: [0x38,0x63,A,A]
# CHECK-LE: addi 3, 3, target@DTPREL         # encoding: [A,A,0x63,0x38]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@DTPREL, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@DTPREL, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_DTPREL16 target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_DTPREL16 target 0x0
            addi 3, 3, target@dtprel

# CHECK-BE: addi 3, 3, target@dtprel@h       # encoding: [0x38,0x63,A,A]
# CHECK-LE: addi 3, 3, target@dtprel@h       # encoding: [A,A,0x63,0x38]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@dtprel@h, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@dtprel@h, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_DTPREL16_HI target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_DTPREL16_HI target 0x0
            addi 3, 3, target@dtprel@h

# CHECK-BE: addi 3, 3, target@dtprel@higher  # encoding: [0x38,0x63,A,A]
# CHECK-LE: addi 3, 3, target@dtprel@higher  # encoding: [A,A,0x63,0x38]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@dtprel@higher, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@dtprel@higher, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_DTPREL16_HIGHER target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_DTPREL16_HIGHER target 0x0
            addi 3, 3, target@dtprel@higher

# CHECK-BE: addis 3, 2, target@dtprel@highest # encoding: [0x3c,0x62,A,A]
# CHECK-LE: addis 3, 2, target@dtprel@highest # encoding: [A,A,0x62,0x3c]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@dtprel@highest, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@dtprel@highest, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_DTPREL16_HIGHEST target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_DTPREL16_HIGHEST target 0x0
            addis 3, 2, target@dtprel@highest

# CHECK-BE: addi 3, 3, target@dtprel@highera  # encoding: [0x38,0x63,A,A]
# CHECK-LE: addi 3, 3, target@dtprel@highera  # encoding: [A,A,0x63,0x38]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@dtprel@highera, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@dtprel@highera, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_DTPREL16_HIGHERA target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_DTPREL16_HIGHERA target 0x0
            addi 3, 3, target@dtprel@highera

# CHECK-BE: addis 3, 2, target@dtprel@highesta # encoding: [0x3c,0x62,A,A]
# CHECK-LE: addis 3, 2, target@dtprel@highesta # encoding: [A,A,0x62,0x3c]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@dtprel@highesta, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@dtprel@highesta, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_DTPREL16_HIGHESTA target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_DTPREL16_HIGHESTA target 0x0
            addis 3, 2, target@dtprel@highesta

# CHECK-BE: ld 1, target@dtprel@l(3)        # encoding: [0xe8,0x23,A,0bAAAAAA00]
# CHECK-LE: ld 1, target@dtprel@l(3)        # encoding: [0bAAAAAA00,A,0x23,0xe8]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@dtprel@l, kind: fixup_ppc_half16ds
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@dtprel@l, kind: fixup_ppc_half16ds
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_DTPREL16_LO_DS target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_DTPREL16_LO_DS target 0x0
            ld 1, target@dtprel@l(3)

# CHECK-BE: ld 1, target@DTPREL(3)          # encoding: [0xe8,0x23,A,0bAAAAAA00]
# CHECK-LE: ld 1, target@DTPREL(3)          # encoding: [0bAAAAAA00,A,0x23,0xe8]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@DTPREL, kind: fixup_ppc_half16ds
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@DTPREL, kind: fixup_ppc_half16ds
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_DTPREL16_DS target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_DTPREL16_DS target 0x0
            ld 1, target@dtprel(3)


# CHECK-BE: addis 3, 2, target@got@tprel@ha # encoding: [0x3c,0x62,A,A]
# CHECK-LE: addis 3, 2, target@got@tprel@ha # encoding: [A,A,0x62,0x3c]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@got@tprel@ha, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@got@tprel@ha, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_GOT_TPREL16_HA target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_GOT_TPREL16_HA target 0x0
            addis 3, 2, target@got@tprel@ha

# CHECK-BE: ld 1, target@got@tprel@l(3)     # encoding: [0xe8,0x23,A,0bAAAAAA00]
# CHECK-LE: ld 1, target@got@tprel@l(3)     # encoding: [0bAAAAAA00,A,0x23,0xe8]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@got@tprel@l, kind: fixup_ppc_half16ds
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@got@tprel@l, kind: fixup_ppc_half16ds
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_GOT_TPREL16_LO_DS target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_GOT_TPREL16_LO_DS target 0x0
            ld 1, target@got@tprel@l(3)

# CHECK-BE: addis 3, 2, target@got@tprel@h  # encoding: [0x3c,0x62,A,A]
# CHECK-LE: addis 3, 2, target@got@tprel@h  # encoding: [A,A,0x62,0x3c]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@got@tprel@h, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@got@tprel@h, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_GOT_TPREL16_HI target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_GOT_TPREL16_HI target 0x0
            addis 3, 2, target@got@tprel@h

# CHECK-BE: addis 3, 2, target@got@tprel@l  # encoding: [0x3c,0x62,A,A]
# CHECK-LE: addis 3, 2, target@got@tprel@l  # encoding: [A,A,0x62,0x3c]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@got@tprel@l, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@got@tprel@l, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_GOT_TPREL16_LO_DS target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_GOT_TPREL16_LO_DS target 0x0
            addis 3, 2, target@got@tprel@l

# CHECK-BE: addis 3, 2, target@got@tprel    # encoding: [0x3c,0x62,A,A]
# CHECK-LE: addis 3, 2, target@got@tprel    # encoding: [A,A,0x62,0x3c]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@got@tprel, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@got@tprel, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_GOT_TPREL16_DS target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_GOT_TPREL16_DS target 0x0
            addis 3, 2, target@got@tprel

# CHECK-BE: ld 1, target@got@tprel(3)       # encoding: [0xe8,0x23,A,0bAAAAAA00]
# CHECK-LE: ld 1, target@got@tprel(3)       # encoding: [0bAAAAAA00,A,0x23,0xe8]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@got@tprel, kind: fixup_ppc_half16ds
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@got@tprel, kind: fixup_ppc_half16ds
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_GOT_TPREL16_DS target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_GOT_TPREL16_DS target 0x0
            ld 1, target@got@tprel(3)

# CHECK-BE: addis 3, 2, target@got@dtprel@ha # encoding: [0x3c,0x62,A,A]
# CHECK-LE: addis 3, 2, target@got@dtprel@ha # encoding: [A,A,0x62,0x3c]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@got@dtprel@ha, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@got@dtprel@ha, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_GOT_DTPREL16_HA target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_GOT_DTPREL16_HA target 0x0
            addis 3, 2, target@got@dtprel@ha

# CHECK-BE: ld 1, target@got@dtprel@l(3)    # encoding: [0xe8,0x23,A,0bAAAAAA00]
# CHECK-LE: ld 1, target@got@dtprel@l(3)    # encoding: [0bAAAAAA00,A,0x23,0xe8]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@got@dtprel@l, kind: fixup_ppc_half16ds
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@got@dtprel@l, kind: fixup_ppc_half16ds
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_GOT_DTPREL16_LO_DS target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_GOT_DTPREL16_LO_DS target 0x0
            ld 1, target@got@dtprel@l(3)

# CHECK-BE: addis 3, 2, target@got@dtprel@h # encoding: [0x3c,0x62,A,A]
# CHECK-LE: addis 3, 2, target@got@dtprel@h # encoding: [A,A,0x62,0x3c]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@got@dtprel@h, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@got@dtprel@h, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_GOT_DTPREL16_HI target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_GOT_DTPREL16_HI target 0x0
            addis 3, 2, target@got@dtprel@h

# CHECK-BE: addis 3, 2, target@got@dtprel@l # encoding: [0x3c,0x62,A,A]
# CHECK-LE: addis 3, 2, target@got@dtprel@l # encoding: [A,A,0x62,0x3c]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@got@dtprel@l, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@got@dtprel@l, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_GOT_DTPREL16_LO_DS target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_GOT_DTPREL16_LO_DS target 0x0
            addis 3, 2, target@got@dtprel@l

# CHECK-BE: addis 3, 2, target@got@dtprel   # encoding: [0x3c,0x62,A,A]
# CHECK-LE: addis 3, 2, target@got@dtprel   # encoding: [A,A,0x62,0x3c]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@got@dtprel, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@got@dtprel, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_GOT_DTPREL16_DS target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_GOT_DTPREL16_DS target 0x0
            addis 3, 2, target@got@dtprel

# CHECK-BE: ld 1, target@got@dtprel(3)      # encoding: [0xe8,0x23,A,0bAAAAAA00]
# CHECK-LE: ld 1, target@got@dtprel(3)      # encoding: [0bAAAAAA00,A,0x23,0xe8]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@got@dtprel, kind: fixup_ppc_half16ds
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@got@dtprel, kind: fixup_ppc_half16ds
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_GOT_DTPREL16_DS target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_GOT_DTPREL16_DS target 0x0
            ld 1, target@got@dtprel(3)

# CHECK-BE: addis 3, 2, target@got@tlsgd@ha # encoding: [0x3c,0x62,A,A]
# CHECK-LE: addis 3, 2, target@got@tlsgd@ha # encoding: [A,A,0x62,0x3c]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@got@tlsgd@ha, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@got@tlsgd@ha, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_GOT_TLSGD16_HA target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_GOT_TLSGD16_HA target 0x0
            addis 3, 2, target@got@tlsgd@ha

# CHECK-BE: addi 3, 3, target@got@tlsgd@l   # encoding: [0x38,0x63,A,A]
# CHECK-LE: addi 3, 3, target@got@tlsgd@l   # encoding: [A,A,0x63,0x38]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@got@tlsgd@l, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@got@tlsgd@l, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_GOT_TLSGD16_LO target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_GOT_TLSGD16_LO target 0x0
            addi 3, 3, target@got@tlsgd@l

# CHECK-BE: addi 3, 3, target@got@tlsgd@h   # encoding: [0x38,0x63,A,A]
# CHECK-LE: addi 3, 3, target@got@tlsgd@h   # encoding: [A,A,0x63,0x38]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@got@tlsgd@h, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@got@tlsgd@h, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_GOT_TLSGD16_HI target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_GOT_TLSGD16_HI target 0x0
            addi 3, 3, target@got@tlsgd@h

# CHECK-BE: addi 3, 3, target@got@tlsgd     # encoding: [0x38,0x63,A,A]
# CHECK-LE: addi 3, 3, target@got@tlsgd     # encoding: [A,A,0x63,0x38]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@got@tlsgd, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@got@tlsgd, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_GOT_TLSGD16 target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_GOT_TLSGD16 target 0x0
            addi 3, 3, target@got@tlsgd


# CHECK-BE: addis 3, 2, target@got@tlsld@ha # encoding: [0x3c,0x62,A,A]
# CHECK-LE: addis 3, 2, target@got@tlsld@ha # encoding: [A,A,0x62,0x3c]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@got@tlsld@ha, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@got@tlsld@ha, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_GOT_TLSLD16_HA target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_GOT_TLSLD16_HA target 0x0
            addis 3, 2, target@got@tlsld@ha

# CHECK-BE: addi 3, 3, target@got@tlsld@l   # encoding: [0x38,0x63,A,A]
# CHECK-LE: addi 3, 3, target@got@tlsld@l   # encoding: [A,A,0x63,0x38]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@got@tlsld@l, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@got@tlsld@l, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_GOT_TLSLD16_LO target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_GOT_TLSLD16_LO target 0x0
            addi 3, 3, target@got@tlsld@l

# CHECK-BE: addi 3, 3, target@got@tlsld@h   # encoding: [0x38,0x63,A,A]
# CHECK-LE: addi 3, 3, target@got@tlsld@h   # encoding: [A,A,0x63,0x38]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@got@tlsld@h, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@got@tlsld@h, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_GOT_TLSLD16_HI target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_GOT_TLSLD16_HI target 0x0
            addi 3, 3, target@got@tlsld@h

# CHECK-BE: addi 3, 3, target@got@tlsld     # encoding: [0x38,0x63,A,A]
# CHECK-LE: addi 3, 3, target@got@tlsld     # encoding: [A,A,0x63,0x38]
# CHECK-BE-NEXT:                            #   fixup A - offset: 2, value: target@got@tlsld, kind: fixup_ppc_half16
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@got@tlsld, kind: fixup_ppc_half16
# CHECK-BE-REL:                             0x{{[0-9A-F]*[26AE]}} R_PPC64_GOT_TLSLD16 target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_GOT_TLSLD16 target 0x0
            addi 3, 3, target@got@tlsld

# CHECK-BE: bl __tls_get_addr(target@tlsgd) # encoding: [0b010010BB,B,B,0bBBBBBB01]
# CHECK-LE: bl __tls_get_addr(target@tlsgd) # encoding: [0bBBBBBB01,B,B,0b010010BB]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target@tlsgd, kind: fixup_ppc_nofixup
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@tlsgd, kind: fixup_ppc_nofixup
# CHECK-BE-NEXT:                            #   fixup B - offset: 0, value: __tls_get_addr, kind: fixup_ppc_br24
# CHECK-LE-NEXT:                            #   fixup B - offset: 0, value: __tls_get_addr, kind: fixup_ppc_br24
# CHECK-BE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_TLSGD target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_TLSGD target 0x0
# CHECK-BE-REL-NEXT:                        0x{{[0-9A-F]*[048C]}} R_PPC64_REL24 __tls_get_addr 0x0
# CHECK-LE-REL-NEXT:                        0x{{[0-9A-F]*[048C]}} R_PPC64_REL24 __tls_get_addr 0x0
            bl __tls_get_addr(target@tlsgd)

# CHECK-BE: bl __tls_get_addr(target@tlsld) # encoding: [0b010010BB,B,B,0bBBBBBB01]
# CHECK-LE: bl __tls_get_addr(target@tlsld) # encoding: [0bBBBBBB01,B,B,0b010010BB]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target@tlsld, kind: fixup_ppc_nofixup
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@tlsld, kind: fixup_ppc_nofixup
# CHECK-BE-NEXT:                            #   fixup B - offset: 0, value: __tls_get_addr, kind: fixup_ppc_br24
# CHECK-LE-NEXT:                            #   fixup B - offset: 0, value: __tls_get_addr, kind: fixup_ppc_br24
# CHECK-BE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_TLSLD target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_TLSLD target 0x0
# CHECK-BE-REL-NEXT:                        0x{{[0-9A-F]*[048C]}} R_PPC64_REL24 __tls_get_addr 0x0
# CHECK-LE-REL-NEXT:                        0x{{[0-9A-F]*[048C]}} R_PPC64_REL24 __tls_get_addr 0x0
            bl __tls_get_addr(target@tlsld)

# CHECK-BE: add 3, 4, target@tls            # encoding: [0x7c,0x64,0x6a,0x14]
# CHECK-LE: add 3, 4, target@tls            # encoding: [0x14,0x6a,0x64,0x7c]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target@tls, kind: fixup_ppc_nofixup
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target@tls, kind: fixup_ppc_nofixup
# CHECK-BE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_TLS target 0x0
# CHECK-LE-REL:                             0x{{[0-9A-F]*[048C]}} R_PPC64_TLS target 0x0
            add 3, 4, target@tls

# Verify that fixups on constants are resolved at assemble time

# CHECK-BE: ori 1, 2, 65535              # encoding: [0x60,0x41,0xff,0xff]
# CHECK-LE: ori 1, 2, 65535              # encoding: [0xff,0xff,0x41,0x60]
            ori 1, 2, 131071@l
# CHECK-BE: ori 1, 2, 1                  # encoding: [0x60,0x41,0x00,0x01]
# CHECK-LE: ori 1, 2, 1                  # encoding: [0x01,0x00,0x41,0x60]
            ori 1, 2, 131071@h
# CHECK-BE: ori 1, 2, 2                  # encoding: [0x60,0x41,0x00,0x02]
# CHECK-LE: ori 1, 2, 2                  # encoding: [0x02,0x00,0x41,0x60]
            ori 1, 2, 131071@ha
# CHECK-BE: addi 1, 2, -1                # encoding: [0x38,0x22,0xff,0xff]
# CHECK-LE: addi 1, 2, -1                # encoding: [0xff,0xff,0x22,0x38]
            addi 1, 2, 131071@l
# CHECK-BE: addi 1, 2, 1                 # encoding: [0x38,0x22,0x00,0x01]
# CHECK-LE: addi 1, 2, 1                 # encoding: [0x01,0x00,0x22,0x38]
            addi 1, 2, 131071@h
# CHECK-BE: addi 1, 2, 2                 # encoding: [0x38,0x22,0x00,0x02]
# CHECK-LE: addi 1, 2, 2                 # encoding: [0x02,0x00,0x22,0x38]
            addi 1, 2, 131071@ha
# CHECK-BE: addis 1, 2, -4096            # encoding: [0x3c,0x22,0xf0,0x00]
# CHECK-LE: addis 1, 2, -4096            # encoding: [0x00,0xf0,0x22,0x3c]
            addis 1, 2, 0xf0000000@h

# Data relocs
# llvm-mc does not show any "encoding" string for data, so we just check the relocs

# CHECK-BE-REL: .rela.data
# CHECK-LE-REL: .rela.data
	.data

# CHECK-BE-REL: 0x{{[0-9A-F]*[08]}} R_PPC64_TOC - 0x0
# CHECK-LE-REL: 0x{{[0-9A-F]*[08]}} R_PPC64_TOC - 0x0
	.quad .TOC.@tocbase

# CHECK-BE-REL: 0x{{[0-9A-F]*[08]}} R_PPC64_DTPMOD64 target 0x0
# CHECK-LE-REL: 0x{{[0-9A-F]*[08]}} R_PPC64_DTPMOD64 target 0x0
	.quad target@dtpmod

# CHECK-BE-REL: 0x{{[0-9A-F]*[08]}} R_PPC64_TPREL64 target 0x0
# CHECK-LE-REL: 0x{{[0-9A-F]*[08]}} R_PPC64_TPREL64 target 0x0
	.quad target@tprel

# CHECK-BE-REL: 0x{{[0-9A-F]*[08]}} R_PPC64_DTPREL64 target 0x0
# CHECK-LE-REL: 0x{{[0-9A-F]*[08]}} R_PPC64_DTPREL64 target 0x0
	.quad target@dtprel

