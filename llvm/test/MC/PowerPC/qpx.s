# RUN: llvm-mc -triple powerpc64-bgq-linux --show-encoding %s | FileCheck %s

# CHECK: qvfabs 3, 5                     # encoding: [0x10,0x60,0x2a,0x10]
         qvfabs %q3, %q5

# CHECK: qvfabs 3, 5                     # encoding: [0x10,0x60,0x2a,0x10]
         qvfabs 3, 5
# CHECK: qvfadd 3, 4, 5                  # encoding: [0x10,0x64,0x28,0x2a]
         qvfadd 3, 4, 5
# CHECK: qvfadds 3, 4, 5                 # encoding: [0x00,0x64,0x28,0x2a]
         qvfadds 3, 4, 5
# CHECK: qvfandc 3, 4, 5                 # encoding: [0x10,0x64,0x2a,0x08]
         qvfandc 3, 4, 5
# CHECK: qvfand 3, 4, 5                  # encoding: [0x10,0x64,0x28,0x88]
         qvfand 3, 4, 5
# CHECK: qvfcfid 3, 5                    # encoding: [0x10,0x60,0x2e,0x9c]
         qvfcfid 3, 5
# CHECK: qvfcfids 3, 5                   # encoding: [0x00,0x60,0x2e,0x9c]
         qvfcfids 3, 5
# CHECK: qvfcfidu 3, 5                   # encoding: [0x10,0x60,0x2f,0x9c]
         qvfcfidu 3, 5
# CHECK: qvfcfidus 3, 5                  # encoding: [0x00,0x60,0x2f,0x9c]
         qvfcfidus 3, 5
# CHECK: qvfclr 3                        # encoding: [0x10,0x63,0x18,0x08]
         qvfclr 3
# CHECK: qvfcpsgn 3, 4, 5                # encoding: [0x10,0x64,0x28,0x10]
         qvfcpsgn 3, 4, 5
# CHECK: qvfctfb 3, 4                    # encoding: [0x10,0x64,0x22,0x88]
         qvfctfb 3, 4
# CHECK: qvfctid 3, 5                    # encoding: [0x10,0x60,0x2e,0x5c]
         qvfctid 3, 5
# CHECK: qvfctidu 3, 5                   # encoding: [0x10,0x60,0x2f,0x5c]
         qvfctidu 3, 5
# CHECK: qvfctiduz 3, 5                  # encoding: [0x10,0x60,0x2f,0x5e]
         qvfctiduz 3, 5
# CHECK: qvfctidz 3, 5                   # encoding: [0x10,0x60,0x2e,0x5e]
         qvfctidz 3, 5
# CHECK: qvfctiw 3, 5                    # encoding: [0x10,0x60,0x28,0x1c]
         qvfctiw 3, 5
# CHECK: qvfctiwu 3, 5                   # encoding: [0x10,0x60,0x29,0x1c]
         qvfctiwu 3, 5
# CHECK: qvfctiwuz 3, 5                  # encoding: [0x10,0x60,0x29,0x1e]
         qvfctiwuz 3, 5
# CHECK: qvfctiwz 3, 5                   # encoding: [0x10,0x60,0x28,0x1e]
         qvfctiwz 3, 5
# CHECK: qvfequ 3, 4, 5                  # encoding: [0x10,0x64,0x2c,0x88]
         qvfequ 3, 4, 5
# CHECK: qvflogical 3, 4, 5, 12          # encoding: [0x10,0x64,0x2e,0x08]
         qvflogical 3, 4, 5, 12
# CHECK: qvfmadd 3, 4, 6, 5              # encoding: [0x10,0x64,0x29,0xba]
         qvfmadd 3, 4, 6, 5
# CHECK: qvfmadds 3, 4, 6, 5             # encoding: [0x00,0x64,0x29,0xba]
         qvfmadds 3, 4, 6, 5
# CHECK: qvfmr 3, 5                      # encoding: [0x10,0x60,0x28,0x90]
         qvfmr 3, 5
# CHECK: qvfmsub 3, 4, 6, 5              # encoding: [0x10,0x64,0x29,0xb8]
         qvfmsub 3, 4, 6, 5
# CHECK: qvfmsubs 3, 4, 6, 5             # encoding: [0x00,0x64,0x29,0xb8]
         qvfmsubs 3, 4, 6, 5
# CHECK: qvfmul 3, 4, 6                  # encoding: [0x10,0x64,0x01,0xb2]
         qvfmul 3, 4, 6
# CHECK: qvfmuls 3, 4, 6                 # encoding: [0x00,0x64,0x01,0xb2]
         qvfmuls 3, 4, 6
# CHECK: qvfnabs 3, 5                    # encoding: [0x10,0x60,0x29,0x10]
         qvfnabs 3, 5
# CHECK: qvfnand 3, 4, 5                 # encoding: [0x10,0x64,0x2f,0x08]
         qvfnand 3, 4, 5
# CHECK: qvfneg 3, 5                     # encoding: [0x10,0x60,0x28,0x50]
         qvfneg 3, 5
# CHECK: qvfnmadd 3, 4, 6, 5             # encoding: [0x10,0x64,0x29,0xbe]
         qvfnmadd 3, 4, 6, 5
# CHECK: qvfnmadds 3, 4, 6, 5            # encoding: [0x00,0x64,0x29,0xbe]
         qvfnmadds 3, 4, 6, 5
# CHECK: qvfnmsub 3, 4, 6, 5             # encoding: [0x10,0x64,0x29,0xbc]
         qvfnmsub 3, 4, 6, 5
# CHECK: qvfnmsubs 3, 4, 6, 5            # encoding: [0x00,0x64,0x29,0xbc]
         qvfnmsubs 3, 4, 6, 5
# CHECK: qvfnor 3, 4, 5                  # encoding: [0x10,0x64,0x2c,0x08]
         qvfnor 3, 4, 5
# CHECK: qvfnot 3, 4                     # encoding: [0x10,0x64,0x25,0x08]
         qvfnot 3, 4
# CHECK: qvforc 3, 4, 5                  # encoding: [0x10,0x64,0x2e,0x88]
         qvforc 3, 4, 5
# CHECK: qvfor 3, 4, 5                   # encoding: [0x10,0x64,0x2b,0x88]
         qvfor 3, 4, 5
# CHECK: qvfperm 3, 4, 5, 6              # encoding: [0x10,0x64,0x29,0x8c]
         qvfperm 3, 4, 5, 6
# CHECK: qvfre 3, 5                      # encoding: [0x10,0x60,0x28,0x30]
         qvfre 3, 5
# CHECK: qvfres 3, 5                     # encoding: [0x00,0x60,0x28,0x30]
         qvfres 3, 5
# CHECK: qvfrim 3, 5                     # encoding: [0x10,0x60,0x2b,0xd0]
         qvfrim 3, 5
# CHECK: qvfrin 3, 5                     # encoding: [0x10,0x60,0x2b,0x10]
         qvfrin 3, 5
# CHECK: qvfrip 3, 5                     # encoding: [0x10,0x60,0x2b,0x90]
         qvfrip 3, 5
# CHECK: qvfriz 3, 5                     # encoding: [0x10,0x60,0x2b,0x50]
         qvfriz 3, 5
# CHECK: qvfrsp 3, 5                     # encoding: [0x10,0x60,0x28,0x18]
         qvfrsp 3, 5
# CHECK: qvfrsqrte 3, 5                  # encoding: [0x10,0x60,0x28,0x34]
         qvfrsqrte 3, 5
# CHECK: qvfrsqrtes 3, 5                 # encoding: [0x00,0x60,0x28,0x34]
         qvfrsqrtes 3, 5
# CHECK: qvfsel 3, 4, 6, 5               # encoding: [0x10,0x64,0x29,0xae]
         qvfsel 3, 4, 6, 5
# CHECK: qvfset 3                        # encoding: [0x10,0x63,0x1f,0x88]
         qvfset 3
# CHECK: qvfsub 3, 4, 5                  # encoding: [0x10,0x64,0x28,0x28]
         qvfsub 3, 4, 5
# CHECK: qvfsubs 3, 4, 5                 # encoding: [0x00,0x64,0x28,0x28]
         qvfsubs 3, 4, 5
# CHECK: qvfxmadd 3, 4, 6, 5             # encoding: [0x10,0x64,0x29,0x92]
         qvfxmadd 3, 4, 6, 5
# CHECK: qvfxmadds 3, 4, 6, 5            # encoding: [0x00,0x64,0x29,0x92]
         qvfxmadds 3, 4, 6, 5
# CHECK: qvfxmul 3, 4, 6                 # encoding: [0x10,0x64,0x01,0xa2]
         qvfxmul 3, 4, 6
# CHECK: qvfxmuls 3, 4, 6                # encoding: [0x00,0x64,0x01,0xa2]
         qvfxmuls 3, 4, 6
# CHECK: qvfxor 3, 4, 5                  # encoding: [0x10,0x64,0x2b,0x08]
         qvfxor 3, 4, 5
# CHECK: qvfxxcpnmadd 3, 4, 6, 5         # encoding: [0x10,0x64,0x29,0x86]
         qvfxxcpnmadd 3, 4, 6, 5
# CHECK: qvfxxcpnmadds 3, 4, 6, 5        # encoding: [0x00,0x64,0x29,0x86]
         qvfxxcpnmadds 3, 4, 6, 5
# CHECK: qvfxxmadd 3, 4, 6, 5            # encoding: [0x10,0x64,0x29,0x82]
         qvfxxmadd 3, 4, 6, 5
# CHECK: qvfxxmadds 3, 4, 6, 5           # encoding: [0x00,0x64,0x29,0x82]
         qvfxxmadds 3, 4, 6, 5
# CHECK: qvfxxnpmadd 3, 4, 6, 5          # encoding: [0x10,0x64,0x29,0x96]
         qvfxxnpmadd 3, 4, 6, 5
# CHECK: qvfxxnpmadds 3, 4, 6, 5         # encoding: [0x00,0x64,0x29,0x96]
         qvfxxnpmadds 3, 4, 6, 5
# CHECK: qvlfcduxa 3, 9, 11              # encoding: [0x7c,0x69,0x58,0xcf]
         qvlfcduxa 3, 9, 11
# CHECK: qvlfcdux 3, 9, 11               # encoding: [0x7c,0x69,0x58,0xce]
         qvlfcdux 3, 9, 11
# CHECK: qvlfcdxa 3, 10, 11              # encoding: [0x7c,0x6a,0x58,0x8f]
         qvlfcdxa 3, 10, 11
# CHECK: qvlfcdx 3, 10, 11               # encoding: [0x7c,0x6a,0x58,0x8e]
         qvlfcdx 3, 10, 11
# CHECK: qvlfcsuxa 3, 9, 11              # encoding: [0x7c,0x69,0x58,0x4f]
         qvlfcsuxa 3, 9, 11
# CHECK: qvlfcsux 3, 9, 11               # encoding: [0x7c,0x69,0x58,0x4e]
         qvlfcsux 3, 9, 11
# CHECK: qvlfcsxa 3, 10, 11              # encoding: [0x7c,0x6a,0x58,0x0f]
         qvlfcsxa 3, 10, 11
# CHECK: qvlfcsx 3, 10, 11               # encoding: [0x7c,0x6a,0x58,0x0e]
         qvlfcsx 3, 10, 11
# CHECK: qvlfduxa 3, 9, 11               # encoding: [0x7c,0x69,0x5c,0xcf]
         qvlfduxa 3, 9, 11
# CHECK: qvlfdux 3, 9, 11                # encoding: [0x7c,0x69,0x5c,0xce]
         qvlfdux 3, 9, 11
# CHECK: qvlfdxa 3, 10, 11               # encoding: [0x7c,0x6a,0x5c,0x8f]
         qvlfdxa 3, 10, 11
# CHECK: qvlfdx 3, 10, 11                # encoding: [0x7c,0x6a,0x5c,0x8e]
         qvlfdx 3, 10, 11
# CHECK: qvlfiwaxa 3, 10, 11             # encoding: [0x7c,0x6a,0x5e,0xcf]
         qvlfiwaxa 3, 10, 11
# CHECK: qvlfiwax 3, 10, 11              # encoding: [0x7c,0x6a,0x5e,0xce]
         qvlfiwax 3, 10, 11
# CHECK: qvlfiwzxa 3, 10, 11             # encoding: [0x7c,0x6a,0x5e,0x8f]
         qvlfiwzxa 3, 10, 11
# CHECK: qvlfiwzx 3, 10, 11              # encoding: [0x7c,0x6a,0x5e,0x8e]
         qvlfiwzx 3, 10, 11
# CHECK: qvlfsuxa 3, 9, 11               # encoding: [0x7c,0x69,0x5c,0x4f]
         qvlfsuxa 3, 9, 11
# CHECK: qvlfsux 3, 9, 11                # encoding: [0x7c,0x69,0x5c,0x4e]
         qvlfsux 3, 9, 11
# CHECK: qvlfsxa 3, 10, 11               # encoding: [0x7c,0x6a,0x5c,0x0f]
         qvlfsxa 3, 10, 11
# CHECK: qvlfsx 3, 10, 11                # encoding: [0x7c,0x6a,0x5c,0x0e]
         qvlfsx 3, 10, 11
# CHECK: qvlpcldx 3, 10, 11              # encoding: [0x7c,0x6a,0x5c,0x8c]
         qvlpcldx 3, 10, 11
# CHECK: qvlpclsx 3, 10, 11              # encoding: [0x7c,0x6a,0x5c,0x0c]
         qvlpclsx 3, 10, 11
# CHECK: qvlpcrdx 3, 10, 11              # encoding: [0x7c,0x6a,0x58,0x8c]
         qvlpcrdx 3, 10, 11
# CHECK: qvlpcrsx 3, 10, 11              # encoding: [0x7c,0x6a,0x58,0x0c]
         qvlpcrsx 3, 10, 11
# CHECK: qvstfcduxa 2, 9, 11             # encoding: [0x7c,0x49,0x59,0xcf]
         qvstfcduxa 2, 9, 11
# CHECK: qvstfcduxia 2, 9, 11            # encoding: [0x7c,0x49,0x59,0xcb]
         qvstfcduxia 2, 9, 11
# CHECK: qvstfcduxi 2, 9, 11             # encoding: [0x7c,0x49,0x59,0xca]
         qvstfcduxi 2, 9, 11
# CHECK: qvstfcdux 2, 9, 11              # encoding: [0x7c,0x49,0x59,0xce]
         qvstfcdux 2, 9, 11
# CHECK: qvstfcdxa 2, 10, 11             # encoding: [0x7c,0x4a,0x59,0x8f]
         qvstfcdxa 2, 10, 11
# CHECK: qvstfcdxia 2, 10, 11            # encoding: [0x7c,0x4a,0x59,0x8b]
         qvstfcdxia 2, 10, 11
# CHECK: qvstfcdxi 2, 10, 11             # encoding: [0x7c,0x4a,0x59,0x8a]
         qvstfcdxi 2, 10, 11
# CHECK: qvstfcdx 2, 10, 11              # encoding: [0x7c,0x4a,0x59,0x8e]
         qvstfcdx 2, 10, 11
# CHECK: qvstfcsuxa 2, 9, 11             # encoding: [0x7c,0x49,0x59,0x4f]
         qvstfcsuxa 2, 9, 11
# CHECK: qvstfcsuxia 2, 9, 11            # encoding: [0x7c,0x49,0x59,0x4b]
         qvstfcsuxia 2, 9, 11
# CHECK: qvstfcsuxi 2, 9, 11             # encoding: [0x7c,0x49,0x59,0x4a]
         qvstfcsuxi 2, 9, 11
# CHECK: qvstfcsux 2, 9, 11              # encoding: [0x7c,0x49,0x59,0x4e]
         qvstfcsux 2, 9, 11
# CHECK: qvstfcsxa 2, 10, 11             # encoding: [0x7c,0x4a,0x59,0x0f]
         qvstfcsxa 2, 10, 11
# CHECK: qvstfcsxia 2, 10, 11            # encoding: [0x7c,0x4a,0x59,0x0b]
         qvstfcsxia 2, 10, 11
# CHECK: qvstfcsxi 2, 10, 11             # encoding: [0x7c,0x4a,0x59,0x0a]
         qvstfcsxi 2, 10, 11
# CHECK: qvstfcsx 2, 10, 11              # encoding: [0x7c,0x4a,0x59,0x0e]
         qvstfcsx 2, 10, 11
# CHECK: qvstfduxa 2, 9, 11              # encoding: [0x7c,0x49,0x5d,0xcf]
         qvstfduxa 2, 9, 11
# CHECK: qvstfduxia 2, 9, 11             # encoding: [0x7c,0x49,0x5d,0xcb]
         qvstfduxia 2, 9, 11
# CHECK: qvstfduxi 2, 9, 11              # encoding: [0x7c,0x49,0x5d,0xca]
         qvstfduxi 2, 9, 11
# CHECK: qvstfdux 2, 9, 11               # encoding: [0x7c,0x49,0x5d,0xce]
         qvstfdux 2, 9, 11
# CHECK: qvstfdxa 2, 10, 11              # encoding: [0x7c,0x4a,0x5d,0x8f]
         qvstfdxa 2, 10, 11
# CHECK: qvstfdxia 2, 10, 11             # encoding: [0x7c,0x4a,0x5d,0x8b]
         qvstfdxia 2, 10, 11
# CHECK: qvstfdxi 2, 10, 11              # encoding: [0x7c,0x4a,0x5d,0x8a]
         qvstfdxi 2, 10, 11
# CHECK: qvstfdx 2, 10, 11               # encoding: [0x7c,0x4a,0x5d,0x8e]
         qvstfdx 2, 10, 11
# CHECK: qvstfiwxa 2, 10, 11             # encoding: [0x7c,0x4a,0x5f,0x8f]
         qvstfiwxa 2, 10, 11
# CHECK: qvstfiwx 2, 10, 11              # encoding: [0x7c,0x4a,0x5f,0x8e]
         qvstfiwx 2, 10, 11
# CHECK: qvstfsuxa 2, 9, 11              # encoding: [0x7c,0x49,0x5d,0x4f]
         qvstfsuxa 2, 9, 11
# CHECK: qvstfsuxia 2, 9, 11             # encoding: [0x7c,0x49,0x5d,0x4b]
         qvstfsuxia 2, 9, 11
# CHECK: qvstfsuxi 2, 9, 11              # encoding: [0x7c,0x49,0x5d,0x4a]
         qvstfsuxi 2, 9, 11
# CHECK: qvstfsux 2, 9, 11               # encoding: [0x7c,0x49,0x5d,0x4e]
         qvstfsux 2, 9, 11
# CHECK: qvstfsxa 2, 10, 11              # encoding: [0x7c,0x4a,0x5d,0x0f]
         qvstfsxa 2, 10, 11
# CHECK: qvstfsxia 2, 10, 11             # encoding: [0x7c,0x4a,0x5d,0x0b]
         qvstfsxia 2, 10, 11
# CHECK: qvstfsxi 2, 10, 11              # encoding: [0x7c,0x4a,0x5d,0x0a]
         qvstfsxi 2, 10, 11
# CHECK: qvstfsx 2, 10, 11               # encoding: [0x7c,0x4a,0x5d,0x0e]
         qvstfsx 2, 10, 11

