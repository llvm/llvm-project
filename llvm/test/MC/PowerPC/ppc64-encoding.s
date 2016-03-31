
# RUN: llvm-mc -triple powerpc64-unknown-unknown --show-encoding %s | FileCheck -check-prefix=CHECK-BE %s
# RUN: llvm-mc -triple powerpc64le-unknown-unknown --show-encoding %s | FileCheck -check-prefix=CHECK-LE %s

# Branch facility

# Branch instructions

# CHECK-BE: b target                        # encoding: [0b010010AA,A,A,0bAAAAAA00]
# CHECK-LE: b target                        # encoding: [0bAAAAAA00,A,A,0b010010AA]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_br24
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_br24
            b target
# CHECK-BE: ba target                       # encoding: [0b010010AA,A,A,0bAAAAAA10]
# CHECK-LE: ba target                       # encoding: [0bAAAAAA10,A,A,0b010010AA]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_br24abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_br24abs
            ba target
# CHECK-BE: bl target                       # encoding: [0b010010AA,A,A,0bAAAAAA01]
# CHECK-LE: bl target                       # encoding: [0bAAAAAA01,A,A,0b010010AA]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_br24
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_br24
            bl target
# CHECK-BE: bla target                      # encoding: [0b010010AA,A,A,0bAAAAAA11]
# CHECK-LE: bla target                      # encoding: [0bAAAAAA11,A,A,0b010010AA]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_br24abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_br24abs
            bla target

# CHECK-BE: bf 10, target                   # encoding: [0x40,0x8a,A,0bAAAAAA00]
# CHECK-LE: bf 10, target                   # encoding: [0bAAAAAA00,A,0x8a,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bc 4, 10, target
# CHECK-BE: bfa 10, target                  # encoding: [0x40,0x8a,A,0bAAAAAA10]
# CHECK-LE: bfa 10, target                  # encoding: [0bAAAAAA10,A,0x8a,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bca 4, 10, target
# CHECK-BE: bfl 10, target                  # encoding: [0x40,0x8a,A,0bAAAAAA01]
# CHECK-LE: bfl 10, target                  # encoding: [0bAAAAAA01,A,0x8a,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14
            bcl 4, 10, target
# CHECK-BE: bfla 10, target                 # encoding: [0x40,0x8a,A,0bAAAAAA11]
# CHECK-LE: bfla 10, target                 # encoding: [0bAAAAAA11,A,0x8a,0x40]
# CHECK-BE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
# CHECK-LE-NEXT:                            #   fixup A - offset: 0, value: target, kind: fixup_ppc_brcond14abs
            bcla 4, 10, target

# CHECK-BE: bclr 4, 10, 3                   # encoding: [0x4c,0x8a,0x18,0x20]
# CHECK-LE: bclr 4, 10, 3                   # encoding: [0x20,0x18,0x8a,0x4c]
            bclr 4, 10, 3
# CHECK-BE: bclr 4, 10                      # encoding: [0x4c,0x8a,0x00,0x20]
# CHECK-LE: bclr 4, 10                      # encoding: [0x20,0x00,0x8a,0x4c]
            bclr 4, 10
# CHECK-BE: bclrl 4, 10, 3                  # encoding: [0x4c,0x8a,0x18,0x21]
# CHECK-LE: bclrl 4, 10, 3                  # encoding: [0x21,0x18,0x8a,0x4c]
            bclrl 4, 10, 3
# CHECK-BE: bclrl 4, 10                     # encoding: [0x4c,0x8a,0x00,0x21]
# CHECK-LE: bclrl 4, 10                     # encoding: [0x21,0x00,0x8a,0x4c]
            bclrl 4, 10
# CHECK-BE: bcctr 4, 10, 3                  # encoding: [0x4c,0x8a,0x1c,0x20]
# CHECK-LE: bcctr 4, 10, 3                  # encoding: [0x20,0x1c,0x8a,0x4c]
            bcctr 4, 10, 3
# CHECK-BE: bcctr 4, 10                     # encoding: [0x4c,0x8a,0x04,0x20]
# CHECK-LE: bcctr 4, 10                     # encoding: [0x20,0x04,0x8a,0x4c]
            bcctr 4, 10
# CHECK-BE: bcctrl 4, 10, 3                 # encoding: [0x4c,0x8a,0x1c,0x21]
# CHECK-LE: bcctrl 4, 10, 3                 # encoding: [0x21,0x1c,0x8a,0x4c]
            bcctrl 4, 10, 3
# CHECK-BE: bcctrl 4, 10                    # encoding: [0x4c,0x8a,0x04,0x21]
# CHECK-LE: bcctrl 4, 10                    # encoding: [0x21,0x04,0x8a,0x4c]
            bcctrl 4, 10

# Condition register instructions

# CHECK-BE: crand 2, 3, 4                   # encoding: [0x4c,0x43,0x22,0x02]
# CHECK-LE: crand 2, 3, 4                   # encoding: [0x02,0x22,0x43,0x4c]
            crand 2, 3, 4
# CHECK-BE: crnand 2, 3, 4                  # encoding: [0x4c,0x43,0x21,0xc2]
# CHECK-LE: crnand 2, 3, 4                  # encoding: [0xc2,0x21,0x43,0x4c]
            crnand 2, 3, 4
# CHECK-BE: cror 2, 3, 4                    # encoding: [0x4c,0x43,0x23,0x82]
# CHECK-LE: cror 2, 3, 4                    # encoding: [0x82,0x23,0x43,0x4c]
            cror 2, 3, 4
# CHECK-BE: crxor 2, 3, 4                   # encoding: [0x4c,0x43,0x21,0x82]
# CHECK-LE: crxor 2, 3, 4                   # encoding: [0x82,0x21,0x43,0x4c]
            crxor 2, 3, 4
# CHECK-BE: crnor 2, 3, 4                   # encoding: [0x4c,0x43,0x20,0x42]
# CHECK-LE: crnor 2, 3, 4                   # encoding: [0x42,0x20,0x43,0x4c]
            crnor 2, 3, 4
# CHECK-BE: creqv 2, 3, 4                   # encoding: [0x4c,0x43,0x22,0x42]
# CHECK-LE: creqv 2, 3, 4                   # encoding: [0x42,0x22,0x43,0x4c]
            creqv 2, 3, 4
# CHECK-BE: crandc 2, 3, 4                  # encoding: [0x4c,0x43,0x21,0x02]
# CHECK-LE: crandc 2, 3, 4                  # encoding: [0x02,0x21,0x43,0x4c]
            crandc 2, 3, 4
# CHECK-BE: crorc 2, 3, 4                   # encoding: [0x4c,0x43,0x23,0x42]
# CHECK-LE: crorc 2, 3, 4                   # encoding: [0x42,0x23,0x43,0x4c]
            crorc 2, 3, 4
# CHECK-BE: mcrf 2, 3                       # encoding: [0x4d,0x0c,0x00,0x00]
# CHECK-LE: mcrf 2, 3                       # encoding: [0x00,0x00,0x0c,0x4d]
            mcrf 2, 3

# System call instruction

# CHECK-BE: sc 1                            # encoding: [0x44,0x00,0x00,0x22]
# CHECK-LE: sc 1                            # encoding: [0x22,0x00,0x00,0x44]
            sc 1
# CHECK-BE: sc                              # encoding: [0x44,0x00,0x00,0x02]
# CHECK-LE: sc                              # encoding: [0x02,0x00,0x00,0x44]
            sc

# Branch history rolling buffer

# CHECK-BE: clrbhrb                         # encoding: [0x7c,0x00,0x03,0x5c]
# CHECK-LE: clrbhrb                         # encoding: [0x5c,0x03,0x00,0x7c]
            clrbhrb
# CHECK-BE: mfbhrbe 9, 983                  # encoding: [0x7d,0x3e,0xba,0x5c]
# CHECK-LE: mfbhrbe 9, 983                  # encoding: [0x5c,0xba,0x3e,0x7d]
            mfbhrbe 9, 983
# CHECK-BE: rfebb 1                         # encoding: [0x4c,0x00,0x09,0x24]
# CHECK-LE: rfebb 1                         # encoding: [0x24,0x09,0x00,0x4c]
            rfebb 1

# Fixed-point facility

# Fixed-point load instructions

# CHECK-BE: lbz 2, 128(4)                   # encoding: [0x88,0x44,0x00,0x80]
# CHECK-LE: lbz 2, 128(4)                   # encoding: [0x80,0x00,0x44,0x88]
            lbz 2, 128(4)
# CHECK-BE: lbzx 2, 3, 4                    # encoding: [0x7c,0x43,0x20,0xae]
# CHECK-LE: lbzx 2, 3, 4                    # encoding: [0xae,0x20,0x43,0x7c]
            lbzx 2, 3, 4
# CHECK-BE: lbzu 2, 128(4)                  # encoding: [0x8c,0x44,0x00,0x80]
# CHECK-LE: lbzu 2, 128(4)                  # encoding: [0x80,0x00,0x44,0x8c]
            lbzu 2, 128(4)
# CHECK-BE: lbzux 2, 3, 4                   # encoding: [0x7c,0x43,0x20,0xee]
# CHECK-LE: lbzux 2, 3, 4                   # encoding: [0xee,0x20,0x43,0x7c]
            lbzux 2, 3, 4
# CHECK-BE: lhz 2, 128(4)                   # encoding: [0xa0,0x44,0x00,0x80]
# CHECK-LE: lhz 2, 128(4)                   # encoding: [0x80,0x00,0x44,0xa0]
            lhz 2, 128(4)
# CHECK-BE: lhzx 2, 3, 4                    # encoding: [0x7c,0x43,0x22,0x2e]
# CHECK-LE: lhzx 2, 3, 4                    # encoding: [0x2e,0x22,0x43,0x7c]
            lhzx 2, 3, 4
# CHECK-BE: lhzu 2, 128(4)                  # encoding: [0xa4,0x44,0x00,0x80]
# CHECK-LE: lhzu 2, 128(4)                  # encoding: [0x80,0x00,0x44,0xa4]
            lhzu 2, 128(4)
# CHECK-BE: lhzux 2, 3, 4                   # encoding: [0x7c,0x43,0x22,0x6e]
# CHECK-LE: lhzux 2, 3, 4                   # encoding: [0x6e,0x22,0x43,0x7c]
            lhzux 2, 3, 4
# CHECK-BE: lha 2, 128(4)                   # encoding: [0xa8,0x44,0x00,0x80]
# CHECK-LE: lha 2, 128(4)                   # encoding: [0x80,0x00,0x44,0xa8]
            lha 2, 128(4)
# CHECK-BE: lhax 2, 3, 4                    # encoding: [0x7c,0x43,0x22,0xae]
# CHECK-LE: lhax 2, 3, 4                    # encoding: [0xae,0x22,0x43,0x7c]
            lhax 2, 3, 4
# CHECK-BE: lhau 2, 128(4)                  # encoding: [0xac,0x44,0x00,0x80]
# CHECK-LE: lhau 2, 128(4)                  # encoding: [0x80,0x00,0x44,0xac]
            lhau 2, 128(4)
# CHECK-BE: lhaux 2, 3, 4                   # encoding: [0x7c,0x43,0x22,0xee]
# CHECK-LE: lhaux 2, 3, 4                   # encoding: [0xee,0x22,0x43,0x7c]
            lhaux 2, 3, 4
# CHECK-BE: lwz 2, 128(4)                   # encoding: [0x80,0x44,0x00,0x80]
# CHECK-LE: lwz 2, 128(4)                   # encoding: [0x80,0x00,0x44,0x80]
            lwz 2, 128(4)
# CHECK-BE: lwzx 2, 3, 4                    # encoding: [0x7c,0x43,0x20,0x2e]
# CHECK-LE: lwzx 2, 3, 4                    # encoding: [0x2e,0x20,0x43,0x7c]
            lwzx 2, 3, 4
# CHECK-BE: lwzu 2, 128(4)                  # encoding: [0x84,0x44,0x00,0x80]
# CHECK-LE: lwzu 2, 128(4)                  # encoding: [0x80,0x00,0x44,0x84]
            lwzu 2, 128(4)
# CHECK-BE: lwzux 2, 3, 4                   # encoding: [0x7c,0x43,0x20,0x6e]
# CHECK-LE: lwzux 2, 3, 4                   # encoding: [0x6e,0x20,0x43,0x7c]
            lwzux 2, 3, 4
# CHECK-BE: lwa 2, 128(4)                   # encoding: [0xe8,0x44,0x00,0x82]
# CHECK-LE: lwa 2, 128(4)                   # encoding: [0x82,0x00,0x44,0xe8]
            lwa 2, 128(4)
# CHECK-BE: lwax 2, 3, 4                    # encoding: [0x7c,0x43,0x22,0xaa]
# CHECK-LE: lwax 2, 3, 4                    # encoding: [0xaa,0x22,0x43,0x7c]
            lwax 2, 3, 4
# CHECK-BE: lwaux 2, 3, 4                   # encoding: [0x7c,0x43,0x22,0xea]
# CHECK-LE: lwaux 2, 3, 4                   # encoding: [0xea,0x22,0x43,0x7c]
            lwaux 2, 3, 4
# CHECK-BE: ld 2, 128(4)                    # encoding: [0xe8,0x44,0x00,0x80]
# CHECK-LE: ld 2, 128(4)                    # encoding: [0x80,0x00,0x44,0xe8]
            ld 2, 128(4)
# CHECK-BE: ldx 2, 3, 4                     # encoding: [0x7c,0x43,0x20,0x2a]
# CHECK-LE: ldx 2, 3, 4                     # encoding: [0x2a,0x20,0x43,0x7c]
            ldx 2, 3, 4
# CHECK-BE: ldu 2, 128(4)                   # encoding: [0xe8,0x44,0x00,0x81]
# CHECK-LE: ldu 2, 128(4)                   # encoding: [0x81,0x00,0x44,0xe8]
            ldu 2, 128(4)
# CHECK-BE: ldux 2, 3, 4                    # encoding: [0x7c,0x43,0x20,0x6a]
# CHECK-LE: ldux 2, 3, 4                    # encoding: [0x6a,0x20,0x43,0x7c]
            ldux 2, 3, 4
# CHECK-BE: ldmx 2, 3, 4                    # encoding: [0x7c,0x43,0x22,0x6a]
# CHECK-LE: ldmx 2, 3, 4                    # encoding: [0x6a,0x22,0x43,0x7c]
            ldmx 2, 3, 4

# Fixed-point store instructions

# CHECK-BE: stb 2, 128(4)                   # encoding: [0x98,0x44,0x00,0x80]
# CHECK-LE: stb 2, 128(4)                   # encoding: [0x80,0x00,0x44,0x98]
            stb 2, 128(4)
# CHECK-BE: stbx 2, 3, 4                    # encoding: [0x7c,0x43,0x21,0xae]
# CHECK-LE: stbx 2, 3, 4                    # encoding: [0xae,0x21,0x43,0x7c]
            stbx 2, 3, 4
# CHECK-BE: stbu 2, 128(4)                  # encoding: [0x9c,0x44,0x00,0x80]
# CHECK-LE: stbu 2, 128(4)                  # encoding: [0x80,0x00,0x44,0x9c]
            stbu 2, 128(4)
# CHECK-BE: stbux 2, 3, 4                   # encoding: [0x7c,0x43,0x21,0xee]
# CHECK-LE: stbux 2, 3, 4                   # encoding: [0xee,0x21,0x43,0x7c]
            stbux 2, 3, 4
# CHECK-BE: sth 2, 128(4)                   # encoding: [0xb0,0x44,0x00,0x80]
# CHECK-LE: sth 2, 128(4)                   # encoding: [0x80,0x00,0x44,0xb0]
            sth 2, 128(4)
# CHECK-BE: sthx 2, 3, 4                    # encoding: [0x7c,0x43,0x23,0x2e]
# CHECK-LE: sthx 2, 3, 4                    # encoding: [0x2e,0x23,0x43,0x7c]
            sthx 2, 3, 4
# CHECK-BE: sthu 2, 128(4)                  # encoding: [0xb4,0x44,0x00,0x80]
# CHECK-LE: sthu 2, 128(4)                  # encoding: [0x80,0x00,0x44,0xb4]
            sthu 2, 128(4)
# CHECK-BE: sthux 2, 3, 4                   # encoding: [0x7c,0x43,0x23,0x6e]
# CHECK-LE: sthux 2, 3, 4                   # encoding: [0x6e,0x23,0x43,0x7c]
            sthux 2, 3, 4
# CHECK-BE: stw 2, 128(4)                   # encoding: [0x90,0x44,0x00,0x80]
# CHECK-LE: stw 2, 128(4)                   # encoding: [0x80,0x00,0x44,0x90]
            stw 2, 128(4)
# CHECK-BE: stwx 2, 3, 4                    # encoding: [0x7c,0x43,0x21,0x2e]
# CHECK-LE: stwx 2, 3, 4                    # encoding: [0x2e,0x21,0x43,0x7c]
            stwx 2, 3, 4
# CHECK-BE: stwu 2, 128(4)                  # encoding: [0x94,0x44,0x00,0x80]
# CHECK-LE: stwu 2, 128(4)                  # encoding: [0x80,0x00,0x44,0x94]
            stwu 2, 128(4)
# CHECK-BE: stwux 2, 3, 4                   # encoding: [0x7c,0x43,0x21,0x6e]
# CHECK-LE: stwux 2, 3, 4                   # encoding: [0x6e,0x21,0x43,0x7c]
            stwux 2, 3, 4
# CHECK-BE: std 2, 128(4)                   # encoding: [0xf8,0x44,0x00,0x80]
# CHECK-LE: std 2, 128(4)                   # encoding: [0x80,0x00,0x44,0xf8]
            std 2, 128(4)
# CHECK-BE: stdx 2, 3, 4                    # encoding: [0x7c,0x43,0x21,0x2a]
# CHECK-LE: stdx 2, 3, 4                    # encoding: [0x2a,0x21,0x43,0x7c]
            stdx 2, 3, 4
# CHECK-BE: stdu 2, 128(4)                  # encoding: [0xf8,0x44,0x00,0x81]
# CHECK-LE: stdu 2, 128(4)                  # encoding: [0x81,0x00,0x44,0xf8]
            stdu 2, 128(4)
# CHECK-BE: stdux 2, 3, 4                   # encoding: [0x7c,0x43,0x21,0x6a]
# CHECK-LE: stdux 2, 3, 4                   # encoding: [0x6a,0x21,0x43,0x7c]
            stdux 2, 3, 4

# Fixed-point load and store with byte reversal instructions

# CHECK-BE: lhbrx 2, 3, 4                   # encoding: [0x7c,0x43,0x26,0x2c]
# CHECK-LE: lhbrx 2, 3, 4                   # encoding: [0x2c,0x26,0x43,0x7c]
            lhbrx 2, 3, 4
# CHECK-BE: sthbrx 2, 3, 4                  # encoding: [0x7c,0x43,0x27,0x2c]
# CHECK-LE: sthbrx 2, 3, 4                  # encoding: [0x2c,0x27,0x43,0x7c]
            sthbrx 2, 3, 4
# CHECK-BE: lwbrx 2, 3, 4                   # encoding: [0x7c,0x43,0x24,0x2c]
# CHECK-LE: lwbrx 2, 3, 4                   # encoding: [0x2c,0x24,0x43,0x7c]
            lwbrx 2, 3, 4
# CHECK-BE: stwbrx 2, 3, 4                  # encoding: [0x7c,0x43,0x25,0x2c]
# CHECK-LE: stwbrx 2, 3, 4                  # encoding: [0x2c,0x25,0x43,0x7c]
            stwbrx 2, 3, 4
# CHECK-BE: ldbrx 2, 3, 4                   # encoding: [0x7c,0x43,0x24,0x28]
# CHECK-LE: ldbrx 2, 3, 4                   # encoding: [0x28,0x24,0x43,0x7c]
            ldbrx 2, 3, 4
# CHECK-BE: stdbrx 2, 3, 4                  # encoding: [0x7c,0x43,0x25,0x28]
# CHECK-LE: stdbrx 2, 3, 4                  # encoding: [0x28,0x25,0x43,0x7c]
            stdbrx 2, 3, 4

# Fixed-point load and store multiple instructions

# CHECK-BE: lmw 2, 128(1)                   # encoding: [0xb8,0x41,0x00,0x80]
# CHECK-LE: lmw 2, 128(1)                   # encoding: [0x80,0x00,0x41,0xb8]
            lmw 2, 128(1)
# CHECK-BE: stmw 2, 128(1)                  # encoding: [0xbc,0x41,0x00,0x80]
# CHECK-LE: stmw 2, 128(1)                  # encoding: [0x80,0x00,0x41,0xbc]
            stmw 2, 128(1)

# FIXME: Fixed-point move assist instructions

# Fixed-point arithmetic instructions

# CHECK-BE: addi 2, 3, 128                  # encoding: [0x38,0x43,0x00,0x80]
# CHECK-LE: addi 2, 3, 128                  # encoding: [0x80,0x00,0x43,0x38]
            addi 2, 3, 128
# CHECK-BE: addis 2, 3, 128                 # encoding: [0x3c,0x43,0x00,0x80]
# CHECK-LE: addis 2, 3, 128                 # encoding: [0x80,0x00,0x43,0x3c]
            addis 2, 3, 128
# CHECK-BE: add 2, 3, 4                     # encoding: [0x7c,0x43,0x22,0x14]
# CHECK-LE: add 2, 3, 4                     # encoding: [0x14,0x22,0x43,0x7c]
            add 2, 3, 4
# CHECK-BE: add. 2, 3, 4                    # encoding: [0x7c,0x43,0x22,0x15]
# CHECK-LE: add. 2, 3, 4                    # encoding: [0x15,0x22,0x43,0x7c]
            add. 2, 3, 4
# FIXME:    addo 2, 3, 4
# FIXME:    addo. 2, 3, 4
# CHECK-BE: subf 2, 3, 4                    # encoding: [0x7c,0x43,0x20,0x50]
# CHECK-LE: subf 2, 3, 4                    # encoding: [0x50,0x20,0x43,0x7c]
            subf 2, 3, 4
# CHECK-BE: subf. 2, 3, 4                   # encoding: [0x7c,0x43,0x20,0x51]
# CHECK-LE: subf. 2, 3, 4                   # encoding: [0x51,0x20,0x43,0x7c]
            subf. 2, 3, 4
# FIXME:    subfo 2, 3, 4
# FIXME:    subfo. 2, 3, 4
# CHECK-BE: addic 2, 3, 128                 # encoding: [0x30,0x43,0x00,0x80]
# CHECK-LE: addic 2, 3, 128                 # encoding: [0x80,0x00,0x43,0x30]
            addic 2, 3, 128
# CHECK-BE: addic. 2, 3, 128                # encoding: [0x34,0x43,0x00,0x80]
# CHECK-LE: addic. 2, 3, 128                # encoding: [0x80,0x00,0x43,0x34]
            addic. 2, 3, 128
# CHECK-BE: subfic 2, 3, 4                  # encoding: [0x20,0x43,0x00,0x04]
# CHECK-LE: subfic 2, 3, 4                  # encoding: [0x04,0x00,0x43,0x20]
            subfic 2, 3, 4

# CHECK-BE: addc 2, 3, 4                    # encoding: [0x7c,0x43,0x20,0x14]
# CHECK-LE: addc 2, 3, 4                    # encoding: [0x14,0x20,0x43,0x7c]
            addc 2, 3, 4
# CHECK-BE: addc. 2, 3, 4                   # encoding: [0x7c,0x43,0x20,0x15]
# CHECK-LE: addc. 2, 3, 4                   # encoding: [0x15,0x20,0x43,0x7c]
            addc. 2, 3, 4
# FIXME:    addco 2, 3, 4
# FIXME:    addco. 2, 3, 4
# CHECK-BE: subfc 2, 3, 4                   # encoding: [0x7c,0x43,0x20,0x10]
# CHECK-LE: subfc 2, 3, 4                   # encoding: [0x10,0x20,0x43,0x7c]
            subfc 2, 3, 4
# CHECK-BE: subfc 2, 3, 4                   # encoding: [0x7c,0x43,0x20,0x10]
# CHECK-LE: subfc 2, 3, 4                   # encoding: [0x10,0x20,0x43,0x7c]
            subfc 2, 3, 4
# FIXME:    subfco 2, 3, 4
# FIXME:    subfco. 2, 3, 4

# CHECK-BE: adde 2, 3, 4                    # encoding: [0x7c,0x43,0x21,0x14]
# CHECK-LE: adde 2, 3, 4                    # encoding: [0x14,0x21,0x43,0x7c]
            adde 2, 3, 4
# CHECK-BE: adde. 2, 3, 4                   # encoding: [0x7c,0x43,0x21,0x15]
# CHECK-LE: adde. 2, 3, 4                   # encoding: [0x15,0x21,0x43,0x7c]
            adde. 2, 3, 4
# FIXME:    addeo 2, 3, 4
# FIXME:    addeo. 2, 3, 4
# CHECK-BE: subfe 2, 3, 4                   # encoding: [0x7c,0x43,0x21,0x10]
# CHECK-LE: subfe 2, 3, 4                   # encoding: [0x10,0x21,0x43,0x7c]
            subfe 2, 3, 4
# CHECK-BE: subfe. 2, 3, 4                  # encoding: [0x7c,0x43,0x21,0x11]
# CHECK-LE: subfe. 2, 3, 4                  # encoding: [0x11,0x21,0x43,0x7c]
            subfe. 2, 3, 4
# FIXME:    subfeo 2, 3, 4
# FIXME:    subfeo. 2, 3, 4

# CHECK-BE: addme 2, 3                      # encoding: [0x7c,0x43,0x01,0xd4]
# CHECK-LE: addme 2, 3                      # encoding: [0xd4,0x01,0x43,0x7c]
            addme 2, 3
# CHECK-BE: addme. 2, 3                     # encoding: [0x7c,0x43,0x01,0xd5]
# CHECK-LE: addme. 2, 3                     # encoding: [0xd5,0x01,0x43,0x7c]
            addme. 2, 3
# FIXME:    addmeo 2, 3
# FIXME:    addmeo. 2, 3
# CHECK-BE: subfme 2, 3                     # encoding: [0x7c,0x43,0x01,0xd0]
# CHECK-LE: subfme 2, 3                     # encoding: [0xd0,0x01,0x43,0x7c]
            subfme 2, 3
# CHECK-BE: subfme. 2, 3                    # encoding: [0x7c,0x43,0x01,0xd1]
# CHECK-LE: subfme. 2, 3                    # encoding: [0xd1,0x01,0x43,0x7c]
            subfme. 2, 3
# FIXME:    subfmeo 2, 3
# FIXME:    subfmeo. 2, 3

# CHECK-BE: addze 2, 3                      # encoding: [0x7c,0x43,0x01,0x94]
# CHECK-LE: addze 2, 3                      # encoding: [0x94,0x01,0x43,0x7c]
            addze 2, 3
# CHECK-BE: addze. 2, 3                     # encoding: [0x7c,0x43,0x01,0x95]
# CHECK-LE: addze. 2, 3                     # encoding: [0x95,0x01,0x43,0x7c]
            addze. 2, 3
# FIXME:    addzeo 2, 3
# FIXME:    addzeo. 2, 3
# CHECK-BE: subfze 2, 3                     # encoding: [0x7c,0x43,0x01,0x90]
# CHECK-LE: subfze 2, 3                     # encoding: [0x90,0x01,0x43,0x7c]
            subfze 2, 3
# CHECK-BE: subfze. 2, 3                    # encoding: [0x7c,0x43,0x01,0x91]
# CHECK-LE: subfze. 2, 3                    # encoding: [0x91,0x01,0x43,0x7c]
            subfze. 2, 3
# FIXME:    subfzeo 2, 3
# FIXME:    subfzeo. 2, 3

# CHECK-BE: neg 2, 3                        # encoding: [0x7c,0x43,0x00,0xd0]
# CHECK-LE: neg 2, 3                        # encoding: [0xd0,0x00,0x43,0x7c]
            neg 2, 3
# CHECK-BE: neg. 2, 3                       # encoding: [0x7c,0x43,0x00,0xd1]
# CHECK-LE: neg. 2, 3                       # encoding: [0xd1,0x00,0x43,0x7c]
            neg. 2, 3
# FIXME:    nego 2, 3
# FIXME:    nego. 2, 3

# CHECK-BE: mulli 2, 3, 128                 # encoding: [0x1c,0x43,0x00,0x80]
# CHECK-LE: mulli 2, 3, 128                 # encoding: [0x80,0x00,0x43,0x1c]
            mulli 2, 3, 128
# CHECK-BE: mulhw 2, 3, 4                   # encoding: [0x7c,0x43,0x20,0x96]
# CHECK-LE: mulhw 2, 3, 4                   # encoding: [0x96,0x20,0x43,0x7c]
            mulhw 2, 3, 4
# CHECK-BE: mulhw. 2, 3, 4                  # encoding: [0x7c,0x43,0x20,0x97]
# CHECK-LE: mulhw. 2, 3, 4                  # encoding: [0x97,0x20,0x43,0x7c]
            mulhw. 2, 3, 4
# CHECK-BE: mullw 2, 3, 4                   # encoding: [0x7c,0x43,0x21,0xd6]
# CHECK-LE: mullw 2, 3, 4                   # encoding: [0xd6,0x21,0x43,0x7c]
            mullw 2, 3, 4
# CHECK-BE: mullw. 2, 3, 4                  # encoding: [0x7c,0x43,0x21,0xd7]
# CHECK-LE: mullw. 2, 3, 4                  # encoding: [0xd7,0x21,0x43,0x7c]
            mullw. 2, 3, 4
# FIXME:    mullwo 2, 3, 4
# FIXME:    mullwo. 2, 3, 4
# CHECK-BE: mulhwu 2, 3, 4                  # encoding: [0x7c,0x43,0x20,0x16]
# CHECK-LE: mulhwu 2, 3, 4                  # encoding: [0x16,0x20,0x43,0x7c]
            mulhwu 2, 3, 4
# CHECK-BE: mulhwu. 2, 3, 4                 # encoding: [0x7c,0x43,0x20,0x17]
# CHECK-LE: mulhwu. 2, 3, 4                 # encoding: [0x17,0x20,0x43,0x7c]
            mulhwu. 2, 3, 4

# CHECK-BE: divw 2, 3, 4                    # encoding: [0x7c,0x43,0x23,0xd6]
# CHECK-LE: divw 2, 3, 4                    # encoding: [0xd6,0x23,0x43,0x7c]
            divw 2, 3, 4
# CHECK-BE: divw. 2, 3, 4                   # encoding: [0x7c,0x43,0x23,0xd7]
# CHECK-LE: divw. 2, 3, 4                   # encoding: [0xd7,0x23,0x43,0x7c]
            divw. 2, 3, 4
# FIXME:    divwo 2, 3, 4
# FIXME:    divwo. 2, 3, 4
# CHECK-BE: divwu 2, 3, 4                   # encoding: [0x7c,0x43,0x23,0x96]
# CHECK-LE: divwu 2, 3, 4                   # encoding: [0x96,0x23,0x43,0x7c]
            divwu 2, 3, 4
# CHECK-BE: divwu. 2, 3, 4                  # encoding: [0x7c,0x43,0x23,0x97]
# CHECK-LE: divwu. 2, 3, 4                  # encoding: [0x97,0x23,0x43,0x7c]
            divwu. 2, 3, 4
# FIXME:    divwuo 2, 3, 4
# FIXME:    divwuo. 2, 3, 4
# CHECK-BE: divwe 2, 3, 4                   # encoding: [0x7c,0x43,0x23,0x56]
# CHECK-LE: divwe 2, 3, 4                   # encoding: [0x56,0x23,0x43,0x7c]
            divwe 2, 3, 4
# CHECK-BE: divwe. 2, 3, 4                  # encoding: [0x7c,0x43,0x23,0x57]
# CHECK-LE: divwe. 2, 3, 4                  # encoding: [0x57,0x23,0x43,0x7c]
            divwe. 2, 3, 4
# FIXME:    divweo 2, 3, 4
# FIXME:    divweo. 2, 3, 4
# CHECK-BE: divweu 2, 3, 4                  # encoding: [0x7c,0x43,0x23,0x16]
# CHECK-LE: divweu 2, 3, 4                  # encoding: [0x16,0x23,0x43,0x7c]
            divweu 2, 3, 4
# CHECK-BE: divweu. 2, 3, 4                 # encoding: [0x7c,0x43,0x23,0x17]
# CHECK-LE: divweu. 2, 3, 4                 # encoding: [0x17,0x23,0x43,0x7c]
            divweu. 2, 3, 4
# FIXME:    divweuo 2, 3, 4
# FIXME:    divweuo. 2, 3, 4

# CHECK-BE: mulld 2, 3, 4                   # encoding: [0x7c,0x43,0x21,0xd2]
# CHECK-LE: mulld 2, 3, 4                   # encoding: [0xd2,0x21,0x43,0x7c]
            mulld 2, 3, 4
# CHECK-BE: mulld. 2, 3, 4                  # encoding: [0x7c,0x43,0x21,0xd3]
# CHECK-LE: mulld. 2, 3, 4                  # encoding: [0xd3,0x21,0x43,0x7c]
            mulld. 2, 3, 4
# FIXME:    mulldo 2, 3, 4
# FIXME:    mulldo. 2, 3, 4
# CHECK-BE: mulhd 2, 3, 4                   # encoding: [0x7c,0x43,0x20,0x92]
# CHECK-LE: mulhd 2, 3, 4                   # encoding: [0x92,0x20,0x43,0x7c]
            mulhd 2, 3, 4
# CHECK-BE: mulhd. 2, 3, 4                  # encoding: [0x7c,0x43,0x20,0x93]
# CHECK-LE: mulhd. 2, 3, 4                  # encoding: [0x93,0x20,0x43,0x7c]
            mulhd. 2, 3, 4
# CHECK-BE: mulhdu 2, 3, 4                  # encoding: [0x7c,0x43,0x20,0x12]
# CHECK-LE: mulhdu 2, 3, 4                  # encoding: [0x12,0x20,0x43,0x7c]
            mulhdu 2, 3, 4
# CHECK-BE: mulhdu. 2, 3, 4                 # encoding: [0x7c,0x43,0x20,0x13]
# CHECK-LE: mulhdu. 2, 3, 4                 # encoding: [0x13,0x20,0x43,0x7c]
            mulhdu. 2, 3, 4

# CHECK-BE: divd 2, 3, 4                    # encoding: [0x7c,0x43,0x23,0xd2]
# CHECK-LE: divd 2, 3, 4                    # encoding: [0xd2,0x23,0x43,0x7c]
            divd 2, 3, 4
# CHECK-BE: divd. 2, 3, 4                   # encoding: [0x7c,0x43,0x23,0xd3]
# CHECK-LE: divd. 2, 3, 4                   # encoding: [0xd3,0x23,0x43,0x7c]
            divd. 2, 3, 4
# FIXME:    divdo 2, 3, 4
# FIXME:    divdo. 2, 3, 4
# CHECK-BE: divdu 2, 3, 4                   # encoding: [0x7c,0x43,0x23,0x92]
# CHECK-LE: divdu 2, 3, 4                   # encoding: [0x92,0x23,0x43,0x7c]
            divdu 2, 3, 4
# CHECK-BE: divdu. 2, 3, 4                  # encoding: [0x7c,0x43,0x23,0x93]
# CHECK-LE: divdu. 2, 3, 4                  # encoding: [0x93,0x23,0x43,0x7c]
            divdu. 2, 3, 4
# FIXME:    divduo 2, 3, 4
# FIXME:    divduo. 2, 3, 4
# CHECK-BE: divde 2, 3, 4                   # encoding: [0x7c,0x43,0x23,0x52]
# CHECK-LE: divde 2, 3, 4                   # encoding: [0x52,0x23,0x43,0x7c]
            divde 2, 3, 4
# CHECK-BE: divde. 2, 3, 4                  # encoding: [0x7c,0x43,0x23,0x53]
# CHECK-LE: divde. 2, 3, 4                  # encoding: [0x53,0x23,0x43,0x7c]
            divde. 2, 3, 4
# FIXME:    divdeo 2, 3, 4
# FIXME:    divdeo. 2, 3, 4
# CHECK-BE: divdeu 2, 3, 4                  # encoding: [0x7c,0x43,0x23,0x12]
# CHECK-LE: divdeu 2, 3, 4                  # encoding: [0x12,0x23,0x43,0x7c]
            divdeu 2, 3, 4
# CHECK-BE: divdeu. 2, 3, 4                 # encoding: [0x7c,0x43,0x23,0x13]
# CHECK-LE: divdeu. 2, 3, 4                 # encoding: [0x13,0x23,0x43,0x7c]
            divdeu. 2, 3, 4
# FIXME:    divdeuo 2, 3, 4
# FIXME:    divdeuo. 2, 3, 4

# Fixed-point compare instructions

# CHECK-BE: cmpdi 2, 3, 128                 # encoding: [0x2d,0x23,0x00,0x80]
# CHECK-LE: cmpdi 2, 3, 128                 # encoding: [0x80,0x00,0x23,0x2d]
            cmpi 2, 1, 3, 128
# CHECK-BE: cmpd 2, 3, 4                    # encoding: [0x7d,0x23,0x20,0x00]
# CHECK-LE: cmpd 2, 3, 4                    # encoding: [0x00,0x20,0x23,0x7d]
            cmp 2, 1, 3, 4
# CHECK-BE: cmpldi 2, 3, 128                # encoding: [0x29,0x23,0x00,0x80]
# CHECK-LE: cmpldi 2, 3, 128                # encoding: [0x80,0x00,0x23,0x29]
            cmpli 2, 1, 3, 128
# CHECK-BE: cmpld 2, 3, 4                   # encoding: [0x7d,0x23,0x20,0x40]
# CHECK-LE: cmpld 2, 3, 4                   # encoding: [0x40,0x20,0x23,0x7d]
            cmpl 2, 1, 3, 4

# CHECK-BE: cmpwi 2, 3, 128                 # encoding: [0x2d,0x03,0x00,0x80]
# CHECK-LE: cmpwi 2, 3, 128                 # encoding: [0x80,0x00,0x03,0x2d]
            cmpi 2, 0, 3, 128
# CHECK-BE: cmpw 2, 3, 4                    # encoding: [0x7d,0x03,0x20,0x00]
# CHECK-LE: cmpw 2, 3, 4                    # encoding: [0x00,0x20,0x03,0x7d]
            cmp 2, 0, 3, 4
# CHECK-BE: cmplwi 2, 3, 128                # encoding: [0x29,0x03,0x00,0x80]
# CHECK-LE: cmplwi 2, 3, 128                # encoding: [0x80,0x00,0x03,0x29]
            cmpli 2, 0, 3, 128
# CHECK-BE: cmplw 2, 3, 4                   # encoding: [0x7d,0x03,0x20,0x40]
# CHECK-LE: cmplw 2, 3, 4                   # encoding: [0x40,0x20,0x03,0x7d]
            cmpl 2, 0, 3, 4

# Fixed-point trap instructions

# CHECK-BE: twllti 3, 4                     # encoding: [0x0c,0x43,0x00,0x04]
# CHECK-LE: twllti 3, 4                     # encoding: [0x04,0x00,0x43,0x0c]
            twi 2, 3, 4
# CHECK-BE: twllt 3, 4                      # encoding: [0x7c,0x43,0x20,0x08]
# CHECK-LE: twllt 3, 4                      # encoding: [0x08,0x20,0x43,0x7c]
            tw 2, 3, 4
# CHECK-BE: tdllti 3, 4                     # encoding: [0x08,0x43,0x00,0x04]
# CHECK-LE: tdllti 3, 4                     # encoding: [0x04,0x00,0x43,0x08]
            tdi 2, 3, 4
# CHECK-BE: tdllt 3, 4                      # encoding: [0x7c,0x43,0x20,0x88]
# CHECK-LE: tdllt 3, 4                      # encoding: [0x88,0x20,0x43,0x7c]
            td 2, 3, 4

# Fixed-point select

# CHECK-BE: isel 2, 3, 4, 5                 # encoding: [0x7c,0x43,0x21,0x5e]
# CHECK-LE: isel 2, 3, 4, 5                 # encoding: [0x5e,0x21,0x43,0x7c]
            isel 2, 3, 4, 5

# Fixed-point logical instructions

# CHECK-BE: andi. 2, 3, 128                 # encoding: [0x70,0x62,0x00,0x80]
# CHECK-LE: andi. 2, 3, 128                 # encoding: [0x80,0x00,0x62,0x70]
            andi. 2, 3, 128
# CHECK-BE: andis. 2, 3, 128                # encoding: [0x74,0x62,0x00,0x80]
# CHECK-LE: andis. 2, 3, 128                # encoding: [0x80,0x00,0x62,0x74]
            andis. 2, 3, 128
# CHECK-BE: ori 2, 3, 128                   # encoding: [0x60,0x62,0x00,0x80]
# CHECK-LE: ori 2, 3, 128                   # encoding: [0x80,0x00,0x62,0x60]
            ori 2, 3, 128
# CHECK-BE: oris 2, 3, 128                  # encoding: [0x64,0x62,0x00,0x80]
# CHECK-LE: oris 2, 3, 128                  # encoding: [0x80,0x00,0x62,0x64]
            oris 2, 3, 128
# CHECK-BE: xori 2, 3, 128                  # encoding: [0x68,0x62,0x00,0x80]
# CHECK-LE: xori 2, 3, 128                  # encoding: [0x80,0x00,0x62,0x68]
            xori 2, 3, 128
# CHECK-BE: xoris 2, 3, 128                 # encoding: [0x6c,0x62,0x00,0x80]
# CHECK-LE: xoris 2, 3, 128                 # encoding: [0x80,0x00,0x62,0x6c]
            xoris 2, 3, 128
# CHECK-BE: and 2, 3, 4                     # encoding: [0x7c,0x62,0x20,0x38]
# CHECK-LE: and 2, 3, 4                     # encoding: [0x38,0x20,0x62,0x7c]
            and 2, 3, 4
# CHECK-BE: and. 2, 3, 4                    # encoding: [0x7c,0x62,0x20,0x39]
# CHECK-LE: and. 2, 3, 4                    # encoding: [0x39,0x20,0x62,0x7c]
            and. 2, 3, 4
# CHECK-BE: xor 2, 3, 4                     # encoding: [0x7c,0x62,0x22,0x78]
# CHECK-LE: xor 2, 3, 4                     # encoding: [0x78,0x22,0x62,0x7c]
            xor 2, 3, 4
# CHECK-BE: xor. 2, 3, 4                    # encoding: [0x7c,0x62,0x22,0x79]
# CHECK-LE: xor. 2, 3, 4                    # encoding: [0x79,0x22,0x62,0x7c]
            xor. 2, 3, 4
# CHECK-BE: nand 2, 3, 4                    # encoding: [0x7c,0x62,0x23,0xb8]
# CHECK-LE: nand 2, 3, 4                    # encoding: [0xb8,0x23,0x62,0x7c]
            nand 2, 3, 4
# CHECK-BE: nand. 2, 3, 4                   # encoding: [0x7c,0x62,0x23,0xb9]
# CHECK-LE: nand. 2, 3, 4                   # encoding: [0xb9,0x23,0x62,0x7c]
            nand. 2, 3, 4
# CHECK-BE: or 2, 3, 4                      # encoding: [0x7c,0x62,0x23,0x78]
# CHECK-LE: or 2, 3, 4                      # encoding: [0x78,0x23,0x62,0x7c]
            or 2, 3, 4
# CHECK-BE: or. 2, 3, 4                     # encoding: [0x7c,0x62,0x23,0x79]
# CHECK-LE: or. 2, 3, 4                     # encoding: [0x79,0x23,0x62,0x7c]
            or. 2, 3, 4
# CHECK-BE: nor 2, 3, 4                     # encoding: [0x7c,0x62,0x20,0xf8]
# CHECK-LE: nor 2, 3, 4                     # encoding: [0xf8,0x20,0x62,0x7c]
            nor 2, 3, 4
# CHECK-BE: nor. 2, 3, 4                    # encoding: [0x7c,0x62,0x20,0xf9]
# CHECK-LE: nor. 2, 3, 4                    # encoding: [0xf9,0x20,0x62,0x7c]
            nor. 2, 3, 4
# CHECK-BE: eqv 2, 3, 4                     # encoding: [0x7c,0x62,0x22,0x38]
# CHECK-LE: eqv 2, 3, 4                     # encoding: [0x38,0x22,0x62,0x7c]
            eqv 2, 3, 4
# CHECK-BE: eqv. 2, 3, 4                    # encoding: [0x7c,0x62,0x22,0x39]
# CHECK-LE: eqv. 2, 3, 4                    # encoding: [0x39,0x22,0x62,0x7c]
            eqv. 2, 3, 4
# CHECK-BE: andc 2, 3, 4                    # encoding: [0x7c,0x62,0x20,0x78]
# CHECK-LE: andc 2, 3, 4                    # encoding: [0x78,0x20,0x62,0x7c]
            andc 2, 3, 4
# CHECK-BE: andc. 2, 3, 4                   # encoding: [0x7c,0x62,0x20,0x79]
# CHECK-LE: andc. 2, 3, 4                   # encoding: [0x79,0x20,0x62,0x7c]
            andc. 2, 3, 4
# CHECK-BE: orc 2, 3, 4                     # encoding: [0x7c,0x62,0x23,0x38]
# CHECK-LE: orc 2, 3, 4                     # encoding: [0x38,0x23,0x62,0x7c]
            orc 2, 3, 4
# CHECK-BE: orc. 2, 3, 4                    # encoding: [0x7c,0x62,0x23,0x39]
# CHECK-LE: orc. 2, 3, 4                    # encoding: [0x39,0x23,0x62,0x7c]
            orc. 2, 3, 4

# CHECK-BE: extsb 2, 3                      # encoding: [0x7c,0x62,0x07,0x74]
# CHECK-LE: extsb 2, 3                      # encoding: [0x74,0x07,0x62,0x7c]
            extsb 2, 3
# CHECK-BE: extsb. 2, 3                     # encoding: [0x7c,0x62,0x07,0x75]
# CHECK-LE: extsb. 2, 3                     # encoding: [0x75,0x07,0x62,0x7c]
            extsb. 2, 3
# CHECK-BE: extsh 2, 3                      # encoding: [0x7c,0x62,0x07,0x34]
# CHECK-LE: extsh 2, 3                      # encoding: [0x34,0x07,0x62,0x7c]
            extsh 2, 3
# CHECK-BE: extsh. 2, 3                     # encoding: [0x7c,0x62,0x07,0x35]
# CHECK-LE: extsh. 2, 3                     # encoding: [0x35,0x07,0x62,0x7c]
            extsh. 2, 3

# CHECK-BE: cntlzw 2, 3                     # encoding: [0x7c,0x62,0x00,0x34]
# CHECK-LE: cntlzw 2, 3                     # encoding: [0x34,0x00,0x62,0x7c]
            cntlzw 2, 3
# CHECK-BE: cntlzw. 2, 3                    # encoding: [0x7c,0x62,0x00,0x35]
# CHECK-LE: cntlzw. 2, 3                    # encoding: [0x35,0x00,0x62,0x7c]
            cntlzw. 2, 3
#
# The POWER variant of cntlzw
# CHECK-BE: cntlzw 2, 3                     # encoding: [0x7c,0x62,0x00,0x34]
# CHECK-LE: cntlzw 2, 3                     # encoding: [0x34,0x00,0x62,0x7c]
            cntlz 2, 3
# CHECK-BE: cntlzw. 2, 3                    # encoding: [0x7c,0x62,0x00,0x35]
# CHECK-LE: cntlzw. 2, 3                    # encoding: [0x35,0x00,0x62,0x7c]
            cntlz. 2, 3
            cmpb 7, 21, 4
# CHECK-BE: cmpb 7, 21, 4                   # encoding: [0x7e,0xa7,0x23,0xf8]
# CHECK-LE: cmpb 7, 21, 4                   # encoding: [0xf8,0x23,0xa7,0x7e]
# FIXME:    popcntb 2, 3
# CHECK-BE: popcntw 2, 3                    # encoding: [0x7c,0x62,0x02,0xf4]
# CHECK-LE: popcntw 2, 3                    # encoding: [0xf4,0x02,0x62,0x7c]
            popcntw 2, 3
# FIXME:    prtyd 2, 3
# FIXME:    prtyw 2, 3

# CHECK-BE: extsw 2, 3                      # encoding: [0x7c,0x62,0x07,0xb4]
# CHECK-LE: extsw 2, 3                      # encoding: [0xb4,0x07,0x62,0x7c]
            extsw 2, 3
# CHECK-BE: extsw. 2, 3                     # encoding: [0x7c,0x62,0x07,0xb5]
# CHECK-LE: extsw. 2, 3                     # encoding: [0xb5,0x07,0x62,0x7c]
            extsw. 2, 3

# CHECK-BE: cntlzd 2, 3                     # encoding: [0x7c,0x62,0x00,0x74]
# CHECK-LE: cntlzd 2, 3                     # encoding: [0x74,0x00,0x62,0x7c]
            cntlzd 2, 3
# CHECK-BE: cntlzd. 2, 3                    # encoding: [0x7c,0x62,0x00,0x75]
# CHECK-LE: cntlzd. 2, 3                    # encoding: [0x75,0x00,0x62,0x7c]
            cntlzd. 2, 3
# CHECK-BE: popcntd 2, 3                    # encoding: [0x7c,0x62,0x03,0xf4]
# CHECK-LE: popcntd 2, 3                    # encoding: [0xf4,0x03,0x62,0x7c]
            popcntd 2, 3
# CHECK-BE: bpermd 2, 3, 4                  # encoding: [0x7c,0x62,0x21,0xf8]
# CHECK-LE: bpermd 2, 3, 4                  # encoding: [0xf8,0x21,0x62,0x7c]
            bpermd 2, 3, 4

# Fixed-point rotate and shift instructions

# CHECK-BE: rlwinm 2, 3, 4, 5, 6            # encoding: [0x54,0x62,0x21,0x4c]
# CHECK-LE: rlwinm 2, 3, 4, 5, 6            # encoding: [0x4c,0x21,0x62,0x54]
            rlwinm 2, 3, 4, 5, 6
# CHECK-BE: rlwinm. 2, 3, 4, 5, 6           # encoding: [0x54,0x62,0x21,0x4d]
# CHECK-LE: rlwinm. 2, 3, 4, 5, 6           # encoding: [0x4d,0x21,0x62,0x54]
            rlwinm. 2, 3, 4, 5, 6
# CHECK-BE: rlwnm 2, 3, 4, 5, 6             # encoding: [0x5c,0x62,0x21,0x4c]
# CHECK-LE: rlwnm 2, 3, 4, 5, 6             # encoding: [0x4c,0x21,0x62,0x5c]
            rlwnm 2, 3, 4, 5, 6
# CHECK-BE: rlwnm. 2, 3, 4, 5, 6            # encoding: [0x5c,0x62,0x21,0x4d]
# CHECK-LE: rlwnm. 2, 3, 4, 5, 6            # encoding: [0x4d,0x21,0x62,0x5c]
            rlwnm. 2, 3, 4, 5, 6
# CHECK-BE: rlwimi 2, 3, 4, 5, 6            # encoding: [0x50,0x62,0x21,0x4c]
# CHECK-LE: rlwimi 2, 3, 4, 5, 6            # encoding: [0x4c,0x21,0x62,0x50]
            rlwimi 2, 3, 4, 5, 6
# CHECK-BE: rlwimi. 2, 3, 4, 5, 6           # encoding: [0x50,0x62,0x21,0x4d]
# CHECK-LE: rlwimi. 2, 3, 4, 5, 6           # encoding: [0x4d,0x21,0x62,0x50]
            rlwimi. 2, 3, 4, 5, 6
# CHECK-BE: rldicl 2, 3, 4, 5               # encoding: [0x78,0x62,0x21,0x40]
# CHECK-LE: rldicl 2, 3, 4, 5               # encoding: [0x40,0x21,0x62,0x78]
            rldicl 2, 3, 4, 5
# CHECK-BE: rldicl. 2, 3, 4, 5              # encoding: [0x78,0x62,0x21,0x41]
# CHECK-LE: rldicl. 2, 3, 4, 5              # encoding: [0x41,0x21,0x62,0x78]
            rldicl. 2, 3, 4, 5
# CHECK-BE: rldicr 2, 3, 4, 5               # encoding: [0x78,0x62,0x21,0x44]
# CHECK-LE: rldicr 2, 3, 4, 5               # encoding: [0x44,0x21,0x62,0x78]
            rldicr 2, 3, 4, 5
# CHECK-BE: rldicr. 2, 3, 4, 5              # encoding: [0x78,0x62,0x21,0x45]
# CHECK-LE: rldicr. 2, 3, 4, 5              # encoding: [0x45,0x21,0x62,0x78]
            rldicr. 2, 3, 4, 5
# CHECK-BE: rldic 2, 3, 4, 5                # encoding: [0x78,0x62,0x21,0x48]
# CHECK-LE: rldic 2, 3, 4, 5                # encoding: [0x48,0x21,0x62,0x78]
            rldic 2, 3, 4, 5
# CHECK-BE: rldic. 2, 3, 4, 5               # encoding: [0x78,0x62,0x21,0x49]
# CHECK-LE: rldic. 2, 3, 4, 5               # encoding: [0x49,0x21,0x62,0x78]
            rldic. 2, 3, 4, 5
# CHECK-BE: rldcl 2, 3, 4, 5                # encoding: [0x78,0x62,0x21,0x50]
# CHECK-LE: rldcl 2, 3, 4, 5                # encoding: [0x50,0x21,0x62,0x78]
            rldcl 2, 3, 4, 5
# CHECK-BE: rldcl. 2, 3, 4, 5               # encoding: [0x78,0x62,0x21,0x51]
# CHECK-LE: rldcl. 2, 3, 4, 5               # encoding: [0x51,0x21,0x62,0x78]
            rldcl. 2, 3, 4, 5
# CHECK-BE: rldcr 2, 3, 4, 5                # encoding: [0x78,0x62,0x21,0x52]
# CHECK-LE: rldcr 2, 3, 4, 5                # encoding: [0x52,0x21,0x62,0x78]
            rldcr 2, 3, 4, 5
# CHECK-BE: rldcr. 2, 3, 4, 5               # encoding: [0x78,0x62,0x21,0x53]
# CHECK-LE: rldcr. 2, 3, 4, 5               # encoding: [0x53,0x21,0x62,0x78]
            rldcr. 2, 3, 4, 5
# CHECK-BE: rldimi 2, 3, 4, 5               # encoding: [0x78,0x62,0x21,0x4c]
# CHECK-LE: rldimi 2, 3, 4, 5               # encoding: [0x4c,0x21,0x62,0x78]
            rldimi 2, 3, 4, 5
# CHECK-BE: rldimi. 2, 3, 4, 5              # encoding: [0x78,0x62,0x21,0x4d]
# CHECK-LE: rldimi. 2, 3, 4, 5              # encoding: [0x4d,0x21,0x62,0x78]
            rldimi. 2, 3, 4, 5

# Aliases that take bit masks...

# CHECK-BE: rlwinm  0, 0, 30, 31, 31        # encoding: [0x54,0x00,0xf7,0xfe]
            rlwinm  0, 0, 30, 1
# CHECK-BE: rlwinm. 0, 0, 30, 31, 31        # encoding: [0x54,0x00,0xf7,0xff]
            rlwinm. 0, 0, 30, 1
# CHECK-BE: rlwinm  0, 0, 30, 31, 0         # encoding: [0x54,0x00,0xf7,0xc0]
            rlwinm  0, 0, 30, 2147483649
# CHECK-BE: rlwinm. 0, 0, 30, 31, 0         # encoding: [0x54,0x00,0xf7,0xc1]
            rlwinm. 0, 0, 30, 2147483649
# CHECK-BE: rlwimi  0, 0, 30, 31, 31        # encoding: [0x50,0x00,0xf7,0xfe]
            rlwimi  0, 0, 30, 1
# CHECK-BE: rlwimi. 0, 0, 30, 31, 31        # encoding: [0x50,0x00,0xf7,0xff]
            rlwimi. 0, 0, 30, 1
# CHECK-BE: rlwimi  0, 0, 30, 31, 0         # encoding: [0x50,0x00,0xf7,0xc0]
            rlwimi  0, 0, 30, 2147483649
# CHECK-BE: rlwimi. 0, 0, 30, 31, 0         # encoding: [0x50,0x00,0xf7,0xc1]
            rlwimi. 0, 0, 30, 2147483649
# CHECK-BE: rlwnm   0, 0, 30, 31, 31        # encoding: [0x5c,0x00,0xf7,0xfe]
            rlwnm  0, 0, 30, 1
# CHECK-BE: rlwnm.  0, 0, 30, 31, 31        # encoding: [0x5c,0x00,0xf7,0xff]
            rlwnm. 0, 0, 30, 1
# CHECK-BE: rlwnm   0, 0, 30, 31, 0         # encoding: [0x5c,0x00,0xf7,0xc0]
            rlwnm  0, 0, 30, 2147483649
# CHECK-BE: rlwnm.  0, 0, 30, 31, 0         # encoding: [0x5c,0x00,0xf7,0xc1]
            rlwnm. 0, 0, 30, 2147483649

# CHECK-BE: slw 2, 3, 4                     # encoding: [0x7c,0x62,0x20,0x30]
# CHECK-LE: slw 2, 3, 4                     # encoding: [0x30,0x20,0x62,0x7c]
            slw 2, 3, 4
# CHECK-BE: slw. 2, 3, 4                    # encoding: [0x7c,0x62,0x20,0x31]
# CHECK-LE: slw. 2, 3, 4                    # encoding: [0x31,0x20,0x62,0x7c]
            slw. 2, 3, 4
# CHECK-BE: srw 2, 3, 4                     # encoding: [0x7c,0x62,0x24,0x30]
# CHECK-LE: srw 2, 3, 4                     # encoding: [0x30,0x24,0x62,0x7c]
            srw 2, 3, 4
# CHECK-BE: srw. 2, 3, 4                    # encoding: [0x7c,0x62,0x24,0x31]
# CHECK-LE: srw. 2, 3, 4                    # encoding: [0x31,0x24,0x62,0x7c]
            srw. 2, 3, 4
# CHECK-BE: srawi 2, 3, 4                   # encoding: [0x7c,0x62,0x26,0x70]
# CHECK-LE: srawi 2, 3, 4                   # encoding: [0x70,0x26,0x62,0x7c]
            srawi 2, 3, 4
# CHECK-BE: srawi. 2, 3, 4                  # encoding: [0x7c,0x62,0x26,0x71]
# CHECK-LE: srawi. 2, 3, 4                  # encoding: [0x71,0x26,0x62,0x7c]
            srawi. 2, 3, 4
# CHECK-BE: sraw 2, 3, 4                    # encoding: [0x7c,0x62,0x26,0x30]
# CHECK-LE: sraw 2, 3, 4                    # encoding: [0x30,0x26,0x62,0x7c]
            sraw 2, 3, 4
# CHECK-BE: sraw. 2, 3, 4                   # encoding: [0x7c,0x62,0x26,0x31]
# CHECK-LE: sraw. 2, 3, 4                   # encoding: [0x31,0x26,0x62,0x7c]
            sraw. 2, 3, 4
# CHECK-BE: sld 2, 3, 4                     # encoding: [0x7c,0x62,0x20,0x36]
# CHECK-LE: sld 2, 3, 4                     # encoding: [0x36,0x20,0x62,0x7c]
            sld 2, 3, 4
# CHECK-BE: sld. 2, 3, 4                    # encoding: [0x7c,0x62,0x20,0x37]
# CHECK-LE: sld. 2, 3, 4                    # encoding: [0x37,0x20,0x62,0x7c]
            sld. 2, 3, 4
# CHECK-BE: srd 2, 3, 4                     # encoding: [0x7c,0x62,0x24,0x36]
# CHECK-LE: srd 2, 3, 4                     # encoding: [0x36,0x24,0x62,0x7c]
            srd 2, 3, 4
# CHECK-BE: srd. 2, 3, 4                    # encoding: [0x7c,0x62,0x24,0x37]
# CHECK-LE: srd. 2, 3, 4                    # encoding: [0x37,0x24,0x62,0x7c]
            srd. 2, 3, 4
# CHECK-BE: sradi 2, 3, 4                   # encoding: [0x7c,0x62,0x26,0x74]
# CHECK-LE: sradi 2, 3, 4                   # encoding: [0x74,0x26,0x62,0x7c]
            sradi 2, 3, 4
# CHECK-BE: sradi. 2, 3, 4                  # encoding: [0x7c,0x62,0x26,0x75]
# CHECK-LE: sradi. 2, 3, 4                  # encoding: [0x75,0x26,0x62,0x7c]
            sradi. 2, 3, 4
# CHECK-BE: srad 2, 3, 4                    # encoding: [0x7c,0x62,0x26,0x34]
# CHECK-LE: srad 2, 3, 4                    # encoding: [0x34,0x26,0x62,0x7c]
            srad 2, 3, 4
# CHECK-BE: srad. 2, 3, 4                   # encoding: [0x7c,0x62,0x26,0x35]
# CHECK-LE: srad. 2, 3, 4                   # encoding: [0x35,0x26,0x62,0x7c]
            srad. 2, 3, 4

# FIXME: BCD assist instructions

# Move to/from system register instructions

# CHECK-BE: mtspr 600, 2                    # encoding: [0x7c,0x58,0x93,0xa6]
# CHECK-LE: mtspr 600, 2                    # encoding: [0xa6,0x93,0x58,0x7c]
            mtspr 600, 2
# CHECK-BE: mfspr 2, 600                    # encoding: [0x7c,0x58,0x92,0xa6]
# CHECK-LE: mfspr 2, 600                    # encoding: [0xa6,0x92,0x58,0x7c]
            mfspr 2, 600
# CHECK-BE: mtcrf 123, 2                    # encoding: [0x7c,0x47,0xb1,0x20]
# CHECK-LE: mtcrf 123, 2                    # encoding: [0x20,0xb1,0x47,0x7c]
            mtcrf 123, 2
# CHECK-BE: mfcr 2                          # encoding: [0x7c,0x40,0x00,0x26]
# CHECK-LE: mfcr 2                          # encoding: [0x26,0x00,0x40,0x7c]
            mfcr 2
# CHECK-BE: mtocrf 16, 2                    # encoding: [0x7c,0x51,0x01,0x20]
# CHECK-LE: mtocrf 16, 2                    # encoding: [0x20,0x01,0x51,0x7c]
            mtocrf 16, 2
# CHECK-BE: mfocrf 16, 8                    # encoding: [0x7e,0x10,0x80,0x26]
# CHECK-LE: mfocrf 16, 8                    # encoding: [0x26,0x80,0x10,0x7e]
            mfocrf 16, 8
# CHECK-BE: mcrxrx 7                        # encoding: [0x7f,0x80,0x04,0x80]
# CHECK-LE: mcrxrx 7                        # encoding: [0x80,0x04,0x80,0x7f]
            mcrxrx 7

# Move to/from segment register
# CHECK-BE: mtsr    12, 10                    # encoding: [0x7d,0x4c,0x01,0xa4]
# CHECK-LE: mtsr    12, 10                    # encoding: [0xa4,0x01,0x4c,0x7d]
            mtsr    12,%r10
# CHECK-BE: mfsr    10, 12                    # encoding: [0x7d,0x4c,0x04,0xa6]
# CHECK-LE: mfsr    10, 12                    # encoding: [0xa6,0x04,0x4c,0x7d]
            mfsr    %r10,12

# CHECK-BE: mtsrin  10, 12                    # encoding: [0x7d,0x40,0x61,0xe4]
# CHECK-LE: mtsrin  10, 12                    # encoding: [0xe4,0x61,0x40,0x7d]
            mtsrin  %r10,%r12
# CHECK-BE: mfsrin  10, 12                    # encoding: [0x7d,0x40,0x65,0x26]
# CHECK-LE: mfsrin  10, 12                    # encoding: [0x26,0x65,0x40,0x7d]
            mfsrin  %r10,%r12
