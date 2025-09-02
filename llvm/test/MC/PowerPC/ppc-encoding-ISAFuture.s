# RUN: llvm-mc -triple powerpc64-unknown-linux-gnu --show-encoding %s | \
# RUN:   FileCheck -check-prefix=CHECK-BE %s
# RUN: llvm-mc -triple powerpc64le-unknown-linux-gnu --show-encoding %s | \
# RUN:   FileCheck -check-prefix=CHECK-LE %s
# RUN: llvm-mc -triple powerpc-unknown-aix-gnu --show-encoding %s | \
# RUN:   FileCheck -check-prefix=CHECK-BE %s

# CHECK-BE: dmxxextfdmr512 2, 34, 1, 0    # encoding: [0xf0,0x82,0x17,0x12]
# CHECK-LE: dmxxextfdmr512 2, 34, 1, 0    # encoding: [0x12,0x17,0x82,0xf0]
            dmxxextfdmr512 2, 34, 1, 0

# CHECK-BE: dmxxextfdmr512 2, 34, 1, 1    # encoding: [0xf0,0x83,0x17,0x12]
# CHECK-LE: dmxxextfdmr512 2, 34, 1, 1    # encoding: [0x12,0x17,0x83,0xf0]
            dmxxextfdmr512 2, 34, 1, 1

# CHECK-BE: dmxxextfdmr256 8, 3, 0        # encoding: [0xf1,0x80,0x47,0x90]
# CHECK-LE: dmxxextfdmr256 8, 3, 0        # encoding: [0x90,0x47,0x80,0xf1]
            dmxxextfdmr256 8, 3, 0

# CHECK-BE: dmxxextfdmr256 8, 3, 3        # encoding: [0xf1,0x81,0x4f,0x90]
# CHECK-LE: dmxxextfdmr256 8, 3, 3        # encoding: [0x90,0x4f,0x81,0xf1]
            dmxxextfdmr256 8, 3, 3

# CHECK-BE: dmxxinstdmr512 1, 2, 34, 0   # encoding: [0xf0,0x82,0x17,0x52]
# CHECK-LE: dmxxinstdmr512 1, 2, 34, 0   # encoding: [0x52,0x17,0x82,0xf0]
            dmxxinstdmr512 1, 2, 34, 0

# CHECK-BE: dmxxinstdmr512 1, 2, 34, 1   # encoding: [0xf0,0x83,0x17,0x52]
# CHECK-LE: dmxxinstdmr512 1, 2, 34, 1   # encoding: [0x52,0x17,0x83,0xf0]
            dmxxinstdmr512 1, 2, 34, 1

# CHECK-BE: dmxxinstdmr256 3, 8, 0       # encoding: [0xf1,0x80,0x47,0x94]
# CHECK-LE: dmxxinstdmr256 3, 8, 0       # encoding: [0x94,0x47,0x80,0xf1]
            dmxxinstdmr256 3, 8, 0

# CHECK-BE: dmxxinstdmr256 3, 8, 3       # encoding: [0xf1,0x81,0x4f,0x94]
# CHECK-LE: dmxxinstdmr256 3, 8, 3       # encoding: [0x94,0x4f,0x81,0xf1]
            dmxxinstdmr256 3, 8, 3

# CHECK-BE: dmsetdmrz 3                   # encoding: [0x7d,0x82,0x01,0x62]
# CHECK-LE: dmsetdmrz 3                   # encoding: [0x62,0x01,0x82,0x7d]
            dmsetdmrz 3

# CHECK-BE: dmmr 4, 5                     # encoding: [0x7e,0x06,0xa1,0x62]
# CHECK-LE: dmmr 4, 5                     # encoding: [0x62,0xa1,0x06,0x7e]
            dmmr 4, 5

# CHECK-BE: dmxor 6, 7                    # encoding: [0x7f,0x07,0xe1,0x62]
# CHECK-LE: dmxor 6, 7                    # encoding: [0x62,0xe1,0x07,0x7f]
            dmxor 6, 7

# CHECK-BE: subfus 3, 0, 4, 5          # encoding: [0x7c,0x64,0x28,0x90]
# CHECK-LE: subfus 3, 0, 4, 5          # encoding: [0x90,0x28,0x64,0x7c]
            subfus 3, 0, 4, 5

# CHECK-BE: subfus 3, 1, 4, 5          # encoding: [0x7c,0x64,0x2c,0x90]
# CHECK-LE: subfus 3, 1, 4, 5          # encoding: [0x90,0x2c,0x64,0x7c]
            subfus 3, 1, 4, 5

# CHECK-BE: subfus. 3, 0, 4, 5         # encoding: [0x7c,0x64,0x28,0x91]
# CHECK-LE: subfus. 3, 0, 4, 5         # encoding: [0x91,0x28,0x64,0x7c]
            subfus. 3, 0, 4, 5

# CHECK-BE: subfus. 3, 1, 4, 5         # encoding: [0x7c,0x64,0x2c,0x91]
# CHECK-LE: subfus. 3, 1, 4, 5         # encoding: [0x91,0x2c,0x64,0x7c]
            subfus. 3, 1, 4, 5

# CHECK-BE: lxvrl 1, 1, 2                 # encoding: [0x7c,0x21,0x14,0x1a]
# CHECK-LE: lxvrl 1, 1, 2                 # encoding: [0x1a,0x14,0x21,0x7c]
            lxvrl 1, 1, 2

# CHECK-BE: lxvrll 0, 3, 4                # encoding: [0x7c,0x03,0x24,0x5a]
# CHECK-LE: lxvrll 0, 3, 4                # encoding: [0x5a,0x24,0x03,0x7c]
            lxvrll 0, 3, 4

# CHECK-BE: stxvrl 2, 0, 1                # encoding: [0x7c,0x40,0x0d,0x1a]
# CHECK-LE: stxvrl 2, 0, 1                # encoding: [0x1a,0x0d,0x40,0x7c]
            stxvrl 2, 0, 1

# CHECK-BE: stxvrll 3, 1, 5               # encoding: [0x7c,0x61,0x2d,0x5a]
# CHECK-LE: stxvrll 3, 1, 5               # encoding: [0x5a,0x2d,0x61,0x7c]
            stxvrll 3, 1, 5

# CHECK-BE: lxvprl 6, 1, 5                # encoding: [0x7c,0xc1,0x2c,0x9a]
# CHECK-LE: lxvprl 6, 1, 5                # encoding: [0x9a,0x2c,0xc1,0x7c]
            lxvprl 6, 1, 5

# CHECK-BE: lxvprll 6, 2, 1               # encoding: [0x7c,0xc2,0x0c,0xda]
# CHECK-LE: lxvprll 6, 2, 1               # encoding: [0xda,0x0c,0xc2,0x7c]
            lxvprll 6, 2, 1

# CHECK-BE: stxvprl 0, 1, 2               # encoding: [0x7c,0x01,0x15,0x9a]
# CHECK-LE: stxvprl 0, 1, 2               # encoding: [0x9a,0x15,0x01,0x7c]
            stxvprl 0, 1, 2

# CHECK-BE: stxvprll 6, 0, 1              # encoding: [0x7c,0xc0,0x0d,0xda]
# CHECK-LE: stxvprll 6, 0, 1              # encoding: [0xda,0x0d,0xc0,0x7c]
            stxvprll 6, 0, 1

            dmxvi8gerx4 1, 2, 4
# CHECK-BE: dmxvi8gerx4 1, 2, 4                     # encoding: [0xec,0x82,0x20,0x58]
# CHECK-LE: dmxvi8gerx4 1, 2, 4                     # encoding: [0x58,0x20,0x82,0xec]

            dmxvi8gerx4pp 1, 0, 2
# CHECK-BE: dmxvi8gerx4pp 1, 0, 2                   # encoding: [0xec,0x80,0x10,0x50]
# CHECK-LE: dmxvi8gerx4pp 1, 0, 2                   # encoding: [0x50,0x10,0x80,0xec]

            pmdmxvi8gerx4 0, 2, 4, 8, 4, 4
# CHECK-BE: pmdmxvi8gerx4 0, 2, 4, 8, 4, 4          # encoding: [0x07,0x90,0x40,0x84,
# CHECK-BE-SAME:                                                 0xec,0x02,0x20,0x58]
# CHECK-LE: pmdmxvi8gerx4 0, 2, 4, 8, 4, 4          # encoding: [0x84,0x40,0x90,0x07,
# CHECK-LE-SAME:                                                 0x58,0x20,0x02,0xec]

            pmdmxvi8gerx4pp 1, 0, 4, 8, 4, 4
#CHECK-BE:  pmdmxvi8gerx4pp 1, 0, 4, 8, 4, 4        # encoding: [0x07,0x90,0x40,0x84,
#CHECK-BE-SAME:                                                  0xec,0x80,0x20,0x50]
#CHECK-LE: pmdmxvi8gerx4pp 1, 0, 4, 8, 4, 4        # encoding: [0x84,0x40,0x90,0x07,
#CHECK-LE-SAME:                                                 0x50,0x20,0x80,0xec]

            dmxvi8gerx4spp 1, 2, 4
#CHECK-BE:  dmxvi8gerx4spp 1, 2, 4                  # encoding: [0xec,0x82,0x23,0x10]
#CHECK-LE:  dmxvi8gerx4spp 1, 2, 4                  # encoding: [0x10,0x23,0x82,0xec]

            pmdmxvi8gerx4spp 0, 2, 4, 8, 4, 4
#CHECK-BE:  pmdmxvi8gerx4spp 0, 2, 4, 8, 4, 4       # encoding: [0x07,0x90,0x40,0x84,
#CHECK-BE-SAME:                                                  0xec,0x02,0x23,0x10]
#CHECK-LE:  pmdmxvi8gerx4spp 0, 2, 4, 8, 4, 4       # encoding: [0x84,0x40,0x90,0x07,
#CHECK-LE-SAME:                                                  0x10,0x23,0x02,0xec]

            dmxvbf16gerx2 1, 2, 4
#CHECK-BE:  dmxvbf16gerx2 1, 2, 4                   # encoding: [0xec,0x82,0x22,0xd8]
#CHECK-LE:  dmxvbf16gerx2 1, 2, 4                   # encoding: [0xd8,0x22,0x82,0xec]

            dmxvbf16gerx2pp 1, 2, 4
#CHECK-BE:  dmxvbf16gerx2pp 1, 2, 4                 # encoding: [0xec,0x82,0x22,0x50]
#CHECK-LE:  dmxvbf16gerx2pp 1, 2, 4                 # encoding: [0x50,0x22,0x82,0xec]

            dmxvbf16gerx2pn 1, 2, 4
#CHECK-BE:  dmxvbf16gerx2pn 1, 2, 4                 # encoding: [0xec,0x82,0x25,0x98]
#CHECK-LE:  dmxvbf16gerx2pn 1, 2, 4                 # encoding: [0x98,0x25,0x82,0xec]

            dmxvbf16gerx2np 1, 2, 4
#CHECK-BE:  dmxvbf16gerx2np 1, 2, 4                 # encoding: [0xec,0x82,0x23,0x98]
#CHECK-LE:  dmxvbf16gerx2np 1, 2, 4                 # encoding: [0x98,0x23,0x82,0xec]

            dmxvbf16gerx2nn 1, 2, 4
#CHECK-BE:  dmxvbf16gerx2nn 1, 2, 4                 # encoding: [0xec,0x82,0x27,0x50]
#CHECK-LE:  dmxvbf16gerx2nn 1, 2, 4                 # encoding: [0x50,0x27,0x82,0xec]

            pmdmxvbf16gerx2 1, 2, 4, 8, 4, 2
#CHECK-BE:  pmdmxvbf16gerx2 1, 2, 4, 8, 4, 2        # encoding: [0x07,0x90,0x80,0x84,
#CHECK-BE-SAME:                                                  0xec,0x82,0x22,0xd8]
#CHECK-LE:  pmdmxvbf16gerx2 1, 2, 4, 8, 4, 2        # encoding: [0x84,0x80,0x90,0x07,
#CHECK-LE-SAME:                                                  0xd8,0x22,0x82,0xec]

            pmdmxvbf16gerx2pp 1, 2, 4, 8, 4, 2
#CHECK-BE:  pmdmxvbf16gerx2pp 1, 2, 4, 8, 4, 2      # encoding: [0x07,0x90,0x80,0x84,
#CHECK-BE-SAME:                                                  0xec,0x82,0x22,0x50]
#CHECK-LE:  pmdmxvbf16gerx2pp 1, 2, 4, 8, 4, 2      # encoding: [0x84,0x80,0x90,0x07,
#CHECK-LE-SAME:                                                  0x50,0x22,0x82,0xec]

            pmdmxvbf16gerx2pn 1, 2, 4, 8, 4, 2
#CHECK-BE:  pmdmxvbf16gerx2pn 1, 2, 4, 8, 4, 2      # encoding: [0x07,0x90,0x80,0x84,
#CHECK-BE-SAME:                                                  0xec,0x82,0x25,0x98]
#CHECK-LE:  pmdmxvbf16gerx2pn 1, 2, 4, 8, 4, 2      # encoding: [0x84,0x80,0x90,0x07,
#CHECK-LE-SAME:                                                  0x98,0x25,0x82,0xec]

            pmdmxvbf16gerx2np 1, 2, 4, 8, 4, 2
#CHECK-BE:  pmdmxvbf16gerx2np 1, 2, 4, 8, 4, 2      # encoding: [0x07,0x90,0x80,0x84,
#CHECK-BE-SAME:                                                  0xec,0x82,0x23,0x98]
#CHECK-LE:  pmdmxvbf16gerx2np 1, 2, 4, 8, 4, 2      # encoding: [0x84,0x80,0x90,0x07,
#CHECK-LE-SAME:                                                  0x98,0x23,0x82,0xec]

            pmdmxvbf16gerx2nn 1, 2, 4, 8, 4, 2
#CHECK-BE:  pmdmxvbf16gerx2nn 1, 2, 4, 8, 4, 2      # encoding: [0x07,0x90,0x80,0x84,
#CHECK-BE-SAME:                                                  0xec,0x82,0x27,0x50]
#CHECK-LE:  pmdmxvbf16gerx2nn 1, 2, 4, 8, 4, 2      # encoding: [0x84,0x80,0x90,0x07,
#CHECK-LE-SAME:                                                  0x50,0x27,0x82,0xec]

            dmxvf16gerx2 1, 0, 2
#CHECK-BE:  dmxvf16gerx2 1, 0, 2                    # encoding: [0xec,0x80,0x12,0x18]
#CHECK-LE:  dmxvf16gerx2 1, 0, 2                    # encoding: [0x18,0x12,0x80,0xec]

            dmxvf16gerx2pp 1, 0, 2
#CHECK-BE:  dmxvf16gerx2pp 1, 0, 2                  # encoding: [0xec,0x80,0x12,0x10]
#CHECK-LE:  dmxvf16gerx2pp 1, 0, 2                  # encoding: [0x10,0x12,0x80,0xec]

            dmxvf16gerx2pn 1, 0, 2
#CHECK-BE:  dmxvf16gerx2pn 1, 0, 2                  # encoding: [0xec,0x80,0x14,0x98]
#CHECK-LE:  dmxvf16gerx2pn 1, 0, 2                  # encoding: [0x98,0x14,0x80,0xec]

            dmxvf16gerx2np 1, 0, 2
#CHECK-BE:  dmxvf16gerx2np 1, 0, 2                  # encoding: [0xec,0x80,0x12,0x98]
#CHECK-LE:  dmxvf16gerx2np 1, 0, 2                  # encoding: [0x98,0x12,0x80,0xec]

            dmxvf16gerx2nn 1, 0, 2
#CHECK-BE:  dmxvf16gerx2nn 1, 0, 2                  # encoding: [0xec,0x80,0x16,0x50]
#CHECK-LE:  dmxvf16gerx2nn 1, 0, 2                  # encoding: [0x50,0x16,0x80,0xec]

            pmdmxvf16gerx2 0, 2, 4, 12, 5, 3
#CHECK-BE:  pmdmxvf16gerx2 0, 2, 4, 12, 5, 3        # encoding: [0x07,0x90,0xc0,0xc5,
#CHECK-BE-SAME:                                                  0xec,0x02,0x22,0x18]
#CHECK-LE:  pmdmxvf16gerx2 0, 2, 4, 12, 5, 3        # encoding: [0xc5,0xc0,0x90,0x07,
#CHECK-LE-SAME:                                                  0x18,0x22,0x02,0xec]

            pmdmxvf16gerx2pp 0, 2, 4, 12, 5, 3
#CHECK-BE:  pmdmxvf16gerx2pp 0, 2, 4, 12, 5, 3        # encoding: [0x07,0x90,0xc0,0xc5,
#CHECK-BE-SAME:                                                    0xec,0x02,0x22,0x10]
#CHECK-LE:  pmdmxvf16gerx2pp 0, 2, 4, 12, 5, 3        # encoding: [0xc5,0xc0,0x90,0x07,
#CHECK-LE-SAME:                                                    0x10,0x22,0x02,0xec]

            pmdmxvf16gerx2pn 0, 2, 4, 12, 5, 3
#CHECK-BE:  pmdmxvf16gerx2pn 0, 2, 4, 12, 5, 3        # encoding: [0x07,0x90,0xc0,0xc5,
#CHECK-BE-SAME:                                                    0xec,0x02,0x24,0x98]
#CHECK-LE:  pmdmxvf16gerx2pn 0, 2, 4, 12, 5, 3        # encoding: [0xc5,0xc0,0x90,0x07,
#CHECK-LE-SAME:                                                    0x98,0x24,0x02,0xec]

            pmdmxvf16gerx2np 0, 2, 4, 12, 5, 3
#CHECK-BE:  pmdmxvf16gerx2np 0, 2, 4, 12, 5, 3        # encoding: [0x07,0x90,0xc0,0xc5,
#CHECK-BE-SAME:                                                    0xec,0x02,0x22,0x98]
#CHECK-LE:  pmdmxvf16gerx2np 0, 2, 4, 12, 5, 3        # encoding: [0xc5,0xc0,0x90,0x07,
#CHECK-LE-SAME:                                                    0x98,0x22,0x02,0xec]

            pmdmxvf16gerx2nn 0, 2, 4, 12, 5, 3
#CHECK-BE:  pmdmxvf16gerx2nn 0, 2, 4, 12, 5, 3        # encoding: [0x07,0x90,0xc0,0xc5,
#CHECK-BE-SAME:                                                    0xec,0x02,0x26,0x50]
#CHECK-LE:  pmdmxvf16gerx2nn 0, 2, 4, 12, 5, 3        # encoding: [0xc5,0xc0,0x90,0x07,
#CHECK-LE-SAME:                                                    0x50,0x26,0x02,0xec]

            dmsha2hash 0, 2, 0
#CHECK-BE:  dmsha256hash 0, 2        # encoding: [0x7c,0x0e,0x41,0x62]
#CHECK-LE:  dmsha256hash 0, 2        # encoding: [0x62,0x41,0x0e,0x7c]

            dmsha2hash 0, 2, 1
#CHECK-BE:  dmsha512hash 0, 2        # encoding: [0x7c,0x2e,0x41,0x62]
#CHECK-LE:  dmsha512hash 0, 2        # encoding: [0x62,0x41,0x2e,0x7c]

            dmsha3hash 0, 5
#CHECK-BE:  dmsha3hash 0, 5       # encoding: [0x7c,0x0f,0x29,0x62]
#CHECK-LE:  dmsha3hash 0, 5       # encoding: [0x62,0x29,0x0f,0x7c]

            dmsha3dw 0
#CHECK-BE:  dmsha3dw 0            # encoding: [0x7c,0x0f,0x01,0x62]
#CHECK-LE:  dmsha3dw 0            # encoding: [0x62,0x01,0x0f,0x7c]

            dmcryshash 0
#CHECK-BE:  dmcryshash 0          # encoding: [0x7c,0x0f,0x61,0x62]
#CHECK-LE:  dmcryshash 0          # encoding: [0x62,0x61,0x0f,0x7c]

            dmxxshapad 0, 1, 2, 1, 3
#CHECK-BE:  dmxxshapad 0, 1, 2, 1, 3       # encoding: [0xf0,0x17,0x0e,0x94]
#CHECK-LE:  dmxxshapad 0, 1, 2, 1, 3       # encoding: [0x94,0x0e,0x17,0xf0]

            dmxxsha3512pad 0, 1, 1
#CHECK-BE:  dmxxsha3512pad 0, 1, 1         # encoding: [0xf0,0x04,0x0e,0x94]
#CHECK-LE:  dmxxsha3512pad 0, 1, 1         # encoding: [0x94,0x0e,0x04,0xf0]

            dmxxsha3384pad  0, 1, 1
#CHECK-BE:  dmxxsha3384pad  0, 1, 1        # encoding: [0xf0,0x05,0x0e,0x94]
#CHECK-LE:  dmxxsha3384pad  0, 1, 1        # encoding: [0x94,0x0e,0x05,0xf0]

            dmxxsha3256pad  0, 1, 1
#CHECK-BE:  dmxxsha3256pad  0, 1, 1        # encoding: [0xf0,0x06,0x0e,0x94]
#CHECK-LE:  dmxxsha3256pad  0, 1, 1        # encoding: [0x94,0x0e,0x06,0xf0]

            dmxxsha3224pad  0, 1, 1
#CHECK-BE:  dmxxsha3224pad  0, 1, 1        # encoding: [0xf0,0x07,0x0e,0x94]
#CHECK-LE:  dmxxsha3224pad  0, 1, 1        # encoding: [0x94,0x0e,0x07,0xf0]

            dmxxshake256pad 0, 1, 1
#CHECK-BE:  dmxxshake256pad 0, 1, 1        # encoding: [0xf0,0x0c,0x0e,0x94]
#CHECK-LE:  dmxxshake256pad 0, 1, 1        # encoding: [0x94,0x0e,0x0c,0xf0]

            dmxxshake128pad 0, 1, 1
#CHECK-BE:  dmxxshake128pad 0, 1, 1        # encoding: [0xf0,0x0d,0x0e,0x94]
#CHECK-LE:  dmxxshake128pad 0, 1, 1        # encoding: [0x94,0x0e,0x0d,0xf0]

            dmxxsha384512pad 0, 1
#CHECK-BE:  dmxxsha384512pad 0, 1          # encoding: [0xf0,0x10,0x0e,0x94]
#CHECK-LE:  dmxxsha384512pad 0, 1          # encoding: [0x94,0x0e,0x10,0xf0]

            dmxxsha224256pad 0, 1
#CHECK-BE:  dmxxsha224256pad 0, 1          # encoding: [0xf0,0x18,0x0e,0x94]
#CHECK-LE:  dmxxsha224256pad 0, 1          # encoding: [0x94,0x0e,0x18,0xf0]

           vupkhsntob 2, 4
#CHECK-BE: vupkhsntob 2, 4                 # encoding: [0x10,0x40,0x21,0x83]
#CHECK-LE: vupkhsntob 2, 4                 # encoding: [0x83,0x21,0x40,0x10]

           vupklsntob 2, 4
#CHECK-BE: vupklsntob 2, 4                 # encoding: [0x10,0x41,0x21,0x83]
#CHECK-LE: vupklsntob 2, 4                 # encoding: [0x83,0x21,0x41,0x10]

        vupkint4tobf16  2, 4, 3
#CHECK-BE: vupkint4tobf16  2, 4, 3         # encoding: [0x10,0x4b,0x21,0x83]
#CHECK-LE: vupkint4tobf16  2, 4, 3         # encoding: [0x83,0x21,0x4b,0x10]

           vupkint8tobf16  1, 3, 1
#CHECK-BE: vupkint8tobf16  1, 3, 1         # encoding: [0x10,0x23,0x19,0x83]
#CHECK-LE: vupkint8tobf16  1, 3, 1         # encoding: [0x83,0x19,0x23,0x10]

           vupkint4tofp32 3, 5, 2
#CHECK-BE: vupkint4tofp32 3, 5, 2         # encoding: [0x10,0x72,0x29,0x83]
#CHECK-LE: vupkint4tofp32 3, 5, 2         # encoding: [0x83,0x29,0x72,0x10]

           vupkint8tofp32 3, 5, 2
#CHECK-BE: vupkint8tofp32 3, 5, 2         # encoding: [0x10,0x6e,0x29,0x83]
#CHECK-LE: vupkint8tofp32 3, 5, 2         # encoding: [0x83,0x29,0x6e,0x10]

           vucmprhn 0, 2, 3
#CHECK-BE: vucmprhn 0, 2, 3                # encoding: [0x10,0x02,0x18,0x03]
#CHECK-LE: vucmprhn 0, 2, 3                # encoding: [0x03,0x18,0x02,0x10]

           vucmprln 3, 5, 6
#CHECK-BE: vucmprln 3, 5, 6                # encoding: [0x10,0x65,0x30,0x43]
#CHECK-LE: vucmprln 3, 5, 6                # encoding: [0x43,0x30,0x65,0x10]

           vucmprhb 1, 3, 6
#CHECK-BE: vucmprhb 1, 3, 6                # encoding: [0x10,0x23,0x30,0x83]
#CHECK-LE: vucmprhb 1, 3, 6                # encoding: [0x83,0x30,0x23,0x10]

           vucmprlb 2, 4, 5
#CHECK-BE: vucmprlb 2, 4, 5                # encoding: [0x10,0x44,0x28,0xc3]
#CHECK-LE: vucmprlb 2, 4, 5                # encoding: [0xc3,0x28,0x44,0x10]

           vucmprlh 2, 4, 5
#CHECK-BE: vucmprlh 2, 4, 5               # encoding: [0x10,0x44,0x29,0x43]
#CHECK-LE: vucmprlh 2, 4, 5               # encoding: [0x43,0x29,0x44,0x10]

           vucmprhh 1, 3, 6
#CHECK-BE: vucmprhh 1, 3, 6               # encoding: [0x10,0x23,0x31,0x03]
#CHECK-LE: vucmprhh 1, 3, 6               # encoding: [0x03,0x31,0x23,0x10]
