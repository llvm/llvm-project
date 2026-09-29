
# RUN: llvm-mc -triple powerpc64-unknown-unknown --show-encoding %s | FileCheck -check-prefix=CHECK-BE %s
# RUN: llvm-mc -triple powerpc64le-unknown-unknown --show-encoding %s | FileCheck -check-prefix=CHECK-LE %s

# Floating-point facility

# Floating-point load instructions

# CHECK-BE: lfs 2, 128(4)                   # encoding: [0xc0,0x44,0x00,0x80]
# CHECK-LE: lfs 2, 128(4)                   # encoding: [0x80,0x00,0x44,0xc0]
            lfs 2, 128(4)
# CHECK-BE: lfsx 2, 3, 4                    # encoding: [0x7c,0x43,0x24,0x2e]
# CHECK-LE: lfsx 2, 3, 4                    # encoding: [0x2e,0x24,0x43,0x7c]
            lfsx 2, 3, 4
# CHECK-BE: lfsu 2, 128(4)                  # encoding: [0xc4,0x44,0x00,0x80]
# CHECK-LE: lfsu 2, 128(4)                  # encoding: [0x80,0x00,0x44,0xc4]
            lfsu 2, 128(4)
# CHECK-BE: lfsux 2, 3, 4                   # encoding: [0x7c,0x43,0x24,0x6e]
# CHECK-LE: lfsux 2, 3, 4                   # encoding: [0x6e,0x24,0x43,0x7c]
            lfsux 2, 3, 4
# CHECK-BE: lfd 2, 128(4)                   # encoding: [0xc8,0x44,0x00,0x80]
# CHECK-LE: lfd 2, 128(4)                   # encoding: [0x80,0x00,0x44,0xc8]
            lfd 2, 128(4)
# CHECK-BE: lfdx 2, 3, 4                    # encoding: [0x7c,0x43,0x24,0xae]
# CHECK-LE: lfdx 2, 3, 4                    # encoding: [0xae,0x24,0x43,0x7c]
            lfdx 2, 3, 4
# CHECK-BE: lfdu 2, 128(4)                  # encoding: [0xcc,0x44,0x00,0x80]
# CHECK-LE: lfdu 2, 128(4)                  # encoding: [0x80,0x00,0x44,0xcc]
            lfdu 2, 128(4)
# CHECK-BE: lfdux 2, 3, 4                   # encoding: [0x7c,0x43,0x24,0xee]
# CHECK-LE: lfdux 2, 3, 4                   # encoding: [0xee,0x24,0x43,0x7c]
            lfdux 2, 3, 4
# CHECK-BE: lfiwax 2, 3, 4                  # encoding: [0x7c,0x43,0x26,0xae]
# CHECK-LE: lfiwax 2, 3, 4                  # encoding: [0xae,0x26,0x43,0x7c]
            lfiwax 2, 3, 4
# CHECK-BE: lfiwzx 2, 3, 4                  # encoding: [0x7c,0x43,0x26,0xee]
# CHECK-LE: lfiwzx 2, 3, 4                  # encoding: [0xee,0x26,0x43,0x7c]
            lfiwzx 2, 3, 4

# Floating-point store instructions

# CHECK-BE: stfs 2, 128(4)                  # encoding: [0xd0,0x44,0x00,0x80]
# CHECK-LE: stfs 2, 128(4)                  # encoding: [0x80,0x00,0x44,0xd0]
            stfs 2, 128(4)
# CHECK-BE: stfsx 2, 3, 4                   # encoding: [0x7c,0x43,0x25,0x2e]
# CHECK-LE: stfsx 2, 3, 4                   # encoding: [0x2e,0x25,0x43,0x7c]
            stfsx 2, 3, 4
# CHECK-BE: stfsu 2, 128(4)                 # encoding: [0xd4,0x44,0x00,0x80]
# CHECK-LE: stfsu 2, 128(4)                 # encoding: [0x80,0x00,0x44,0xd4]
            stfsu 2, 128(4)
# CHECK-BE: stfsux 2, 3, 4                  # encoding: [0x7c,0x43,0x25,0x6e]
# CHECK-LE: stfsux 2, 3, 4                  # encoding: [0x6e,0x25,0x43,0x7c]
            stfsux 2, 3, 4
# CHECK-BE: stfd 2, 128(4)                  # encoding: [0xd8,0x44,0x00,0x80]
# CHECK-LE: stfd 2, 128(4)                  # encoding: [0x80,0x00,0x44,0xd8]
            stfd 2, 128(4)
# CHECK-BE: stfdx 2, 3, 4                   # encoding: [0x7c,0x43,0x25,0xae]
# CHECK-LE: stfdx 2, 3, 4                   # encoding: [0xae,0x25,0x43,0x7c]
            stfdx 2, 3, 4
# CHECK-BE: stfdu 2, 128(4)                 # encoding: [0xdc,0x44,0x00,0x80]
# CHECK-LE: stfdu 2, 128(4)                 # encoding: [0x80,0x00,0x44,0xdc]
            stfdu 2, 128(4)
# CHECK-BE: stfdux 2, 3, 4                  # encoding: [0x7c,0x43,0x25,0xee]
# CHECK-LE: stfdux 2, 3, 4                  # encoding: [0xee,0x25,0x43,0x7c]
            stfdux 2, 3, 4
# CHECK-BE: stfiwx 2, 3, 4                  # encoding: [0x7c,0x43,0x27,0xae]
# CHECK-LE: stfiwx 2, 3, 4                  # encoding: [0xae,0x27,0x43,0x7c]
            stfiwx 2, 3, 4

# Floating-point move instructions

# CHECK-BE: fmr 2, 3                        # encoding: [0xfc,0x40,0x18,0x90]
# CHECK-LE: fmr 2, 3                        # encoding: [0x90,0x18,0x40,0xfc]
            fmr 2, 3
# CHECK-BE: fmr. 2, 3                       # encoding: [0xfc,0x40,0x18,0x91]
# CHECK-LE: fmr. 2, 3                       # encoding: [0x91,0x18,0x40,0xfc]
            fmr. 2, 3
# CHECK-BE: fneg 2, 3                       # encoding: [0xfc,0x40,0x18,0x50]
# CHECK-LE: fneg 2, 3                       # encoding: [0x50,0x18,0x40,0xfc]
            fneg 2, 3
# CHECK-BE: fneg. 2, 3                      # encoding: [0xfc,0x40,0x18,0x51]
# CHECK-LE: fneg. 2, 3                      # encoding: [0x51,0x18,0x40,0xfc]
            fneg. 2, 3
# CHECK-BE: fabs 2, 3                       # encoding: [0xfc,0x40,0x1a,0x10]
# CHECK-LE: fabs 2, 3                       # encoding: [0x10,0x1a,0x40,0xfc]
            fabs 2, 3
# CHECK-BE: fabs. 2, 3                      # encoding: [0xfc,0x40,0x1a,0x11]
# CHECK-LE: fabs. 2, 3                      # encoding: [0x11,0x1a,0x40,0xfc]
            fabs. 2, 3
# CHECK-BE: fnabs 2, 3                      # encoding: [0xfc,0x40,0x19,0x10]
# CHECK-LE: fnabs 2, 3                      # encoding: [0x10,0x19,0x40,0xfc]
            fnabs 2, 3
# CHECK-BE: fnabs. 2, 3                     # encoding: [0xfc,0x40,0x19,0x11]
# CHECK-LE: fnabs. 2, 3                     # encoding: [0x11,0x19,0x40,0xfc]
            fnabs. 2, 3
# CHECK-BE: fcpsgn 2, 3, 4                  # encoding: [0xfc,0x43,0x20,0x10]
# CHECK-LE: fcpsgn 2, 3, 4                  # encoding: [0x10,0x20,0x43,0xfc]
            fcpsgn 2, 3, 4
# CHECK-BE: fcpsgn. 2, 3, 4                 # encoding: [0xfc,0x43,0x20,0x11]
# CHECK-LE: fcpsgn. 2, 3, 4                 # encoding: [0x11,0x20,0x43,0xfc]
            fcpsgn. 2, 3, 4

# Floating-point arithmetic instructions

# CHECK-BE: fadd 2, 3, 4                    # encoding: [0xfc,0x43,0x20,0x2a]
# CHECK-LE: fadd 2, 3, 4                    # encoding: [0x2a,0x20,0x43,0xfc]
            fadd 2, 3, 4
# CHECK-BE: fadd. 2, 3, 4                   # encoding: [0xfc,0x43,0x20,0x2b]
# CHECK-LE: fadd. 2, 3, 4                   # encoding: [0x2b,0x20,0x43,0xfc]
            fadd. 2, 3, 4
# CHECK-BE: fadds 2, 3, 4                   # encoding: [0xec,0x43,0x20,0x2a]
# CHECK-LE: fadds 2, 3, 4                   # encoding: [0x2a,0x20,0x43,0xec]
            fadds 2, 3, 4
# CHECK-BE: fadds. 2, 3, 4                  # encoding: [0xec,0x43,0x20,0x2b]
# CHECK-LE: fadds. 2, 3, 4                  # encoding: [0x2b,0x20,0x43,0xec]
            fadds. 2, 3, 4
# CHECK-BE: fsub 2, 3, 4                    # encoding: [0xfc,0x43,0x20,0x28]
# CHECK-LE: fsub 2, 3, 4                    # encoding: [0x28,0x20,0x43,0xfc]
            fsub 2, 3, 4
# CHECK-BE: fsub. 2, 3, 4                   # encoding: [0xfc,0x43,0x20,0x29]
# CHECK-LE: fsub. 2, 3, 4                   # encoding: [0x29,0x20,0x43,0xfc]
            fsub. 2, 3, 4
# CHECK-BE: fsubs 2, 3, 4                   # encoding: [0xec,0x43,0x20,0x28]
# CHECK-LE: fsubs 2, 3, 4                   # encoding: [0x28,0x20,0x43,0xec]
            fsubs 2, 3, 4
# CHECK-BE: fsubs. 2, 3, 4                  # encoding: [0xec,0x43,0x20,0x29]
# CHECK-LE: fsubs. 2, 3, 4                  # encoding: [0x29,0x20,0x43,0xec]
            fsubs. 2, 3, 4

# CHECK-BE: fmul 2, 3, 4                    # encoding: [0xfc,0x43,0x01,0x32]
# CHECK-LE: fmul 2, 3, 4                    # encoding: [0x32,0x01,0x43,0xfc]
            fmul 2, 3, 4
# CHECK-BE: fmul. 2, 3, 4                   # encoding: [0xfc,0x43,0x01,0x33]
# CHECK-LE: fmul. 2, 3, 4                   # encoding: [0x33,0x01,0x43,0xfc]
            fmul. 2, 3, 4
# CHECK-BE: fmuls 2, 3, 4                   # encoding: [0xec,0x43,0x01,0x32]
# CHECK-LE: fmuls 2, 3, 4                   # encoding: [0x32,0x01,0x43,0xec]
            fmuls 2, 3, 4
# CHECK-BE: fmuls. 2, 3, 4                  # encoding: [0xec,0x43,0x01,0x33]
# CHECK-LE: fmuls. 2, 3, 4                  # encoding: [0x33,0x01,0x43,0xec]
            fmuls. 2, 3, 4
# CHECK-BE: fdiv 2, 3, 4                    # encoding: [0xfc,0x43,0x20,0x24]
# CHECK-LE: fdiv 2, 3, 4                    # encoding: [0x24,0x20,0x43,0xfc]
            fdiv 2, 3, 4
# CHECK-BE: fdiv. 2, 3, 4                   # encoding: [0xfc,0x43,0x20,0x25]
# CHECK-LE: fdiv. 2, 3, 4                   # encoding: [0x25,0x20,0x43,0xfc]
            fdiv. 2, 3, 4
# CHECK-BE: fdivs 2, 3, 4                   # encoding: [0xec,0x43,0x20,0x24]
# CHECK-LE: fdivs 2, 3, 4                   # encoding: [0x24,0x20,0x43,0xec]
            fdivs 2, 3, 4
# CHECK-BE: fdivs. 2, 3, 4                  # encoding: [0xec,0x43,0x20,0x25]
# CHECK-LE: fdivs. 2, 3, 4                  # encoding: [0x25,0x20,0x43,0xec]
            fdivs. 2, 3, 4
# CHECK-BE: fsqrt 2, 3                      # encoding: [0xfc,0x40,0x18,0x2c]
# CHECK-LE: fsqrt 2, 3                      # encoding: [0x2c,0x18,0x40,0xfc]
            fsqrt 2, 3
# CHECK-BE: fsqrt. 2, 3                     # encoding: [0xfc,0x40,0x18,0x2d]
# CHECK-LE: fsqrt. 2, 3                     # encoding: [0x2d,0x18,0x40,0xfc]
            fsqrt. 2, 3
# CHECK-BE: fsqrts 2, 3                     # encoding: [0xec,0x40,0x18,0x2c]
# CHECK-LE: fsqrts 2, 3                     # encoding: [0x2c,0x18,0x40,0xec]
            fsqrts 2, 3
# CHECK-BE: fsqrts. 2, 3                    # encoding: [0xec,0x40,0x18,0x2d]
# CHECK-LE: fsqrts. 2, 3                    # encoding: [0x2d,0x18,0x40,0xec]
            fsqrts. 2, 3

# CHECK-BE: fre 2, 3                        # encoding: [0xfc,0x40,0x18,0x30]
# CHECK-LE: fre 2, 3                        # encoding: [0x30,0x18,0x40,0xfc]
            fre 2, 3
# CHECK-BE: fre. 2, 3                       # encoding: [0xfc,0x40,0x18,0x31]
# CHECK-LE: fre. 2, 3                       # encoding: [0x31,0x18,0x40,0xfc]
            fre. 2, 3
# CHECK-BE: fres 2, 3                       # encoding: [0xec,0x40,0x18,0x30]
# CHECK-LE: fres 2, 3                       # encoding: [0x30,0x18,0x40,0xec]
            fres 2, 3
# CHECK-BE: fres. 2, 3                      # encoding: [0xec,0x40,0x18,0x31]
# CHECK-LE: fres. 2, 3                      # encoding: [0x31,0x18,0x40,0xec]
            fres. 2, 3
# CHECK-BE: frsqrte 2, 3                    # encoding: [0xfc,0x40,0x18,0x34]
# CHECK-LE: frsqrte 2, 3                    # encoding: [0x34,0x18,0x40,0xfc]
            frsqrte 2, 3
# CHECK-BE: frsqrte. 2, 3                   # encoding: [0xfc,0x40,0x18,0x35]
# CHECK-LE: frsqrte. 2, 3                   # encoding: [0x35,0x18,0x40,0xfc]
            frsqrte. 2, 3
# CHECK-BE: frsqrtes 2, 3                   # encoding: [0xec,0x40,0x18,0x34]
# CHECK-LE: frsqrtes 2, 3                   # encoding: [0x34,0x18,0x40,0xec]
            frsqrtes 2, 3
# CHECK-BE: frsqrtes. 2, 3                  # encoding: [0xec,0x40,0x18,0x35]
# CHECK-LE: frsqrtes. 2, 3                  # encoding: [0x35,0x18,0x40,0xec]
            frsqrtes. 2, 3

# CHECK-BE: ftdiv 2, 3, 4                   # encoding: [0xfd,0x03,0x21,0x00]
# CHECK-LE: ftdiv 2, 3, 4                   # encoding: [0x00,0x21,0x03,0xfd]
            ftdiv 2, 3, 4

# CHECK-BE: ftsqrt 2, 3                    # encoding: [0xfd,0x00,0x19,0x40]
# CHECK-LE: ftsqrt 2, 3                    # encoding: [0x40,0x19,0x00,0xfd]
            ftsqrt 2, 3

# CHECK-BE: fmadd 2, 3, 4, 5                # encoding: [0xfc,0x43,0x29,0x3a]
# CHECK-LE: fmadd 2, 3, 4, 5                # encoding: [0x3a,0x29,0x43,0xfc]
            fmadd 2, 3, 4, 5
# CHECK-BE: fmadd. 2, 3, 4, 5               # encoding: [0xfc,0x43,0x29,0x3b]
# CHECK-LE: fmadd. 2, 3, 4, 5               # encoding: [0x3b,0x29,0x43,0xfc]
            fmadd. 2, 3, 4, 5
# CHECK-BE: fmadds 2, 3, 4, 5               # encoding: [0xec,0x43,0x29,0x3a]
# CHECK-LE: fmadds 2, 3, 4, 5               # encoding: [0x3a,0x29,0x43,0xec]
            fmadds 2, 3, 4, 5
# CHECK-BE: fmadds. 2, 3, 4, 5              # encoding: [0xec,0x43,0x29,0x3b]
# CHECK-LE: fmadds. 2, 3, 4, 5              # encoding: [0x3b,0x29,0x43,0xec]
            fmadds. 2, 3, 4, 5
# CHECK-BE: fmsub 2, 3, 4, 5                # encoding: [0xfc,0x43,0x29,0x38]
# CHECK-LE: fmsub 2, 3, 4, 5                # encoding: [0x38,0x29,0x43,0xfc]
            fmsub 2, 3, 4, 5
# CHECK-BE: fmsub. 2, 3, 4, 5               # encoding: [0xfc,0x43,0x29,0x39]
# CHECK-LE: fmsub. 2, 3, 4, 5               # encoding: [0x39,0x29,0x43,0xfc]
            fmsub. 2, 3, 4, 5
# CHECK-BE: fmsubs 2, 3, 4, 5               # encoding: [0xec,0x43,0x29,0x38]
# CHECK-LE: fmsubs 2, 3, 4, 5               # encoding: [0x38,0x29,0x43,0xec]
            fmsubs 2, 3, 4, 5
# CHECK-BE: fmsubs. 2, 3, 4, 5              # encoding: [0xec,0x43,0x29,0x39]
# CHECK-LE: fmsubs. 2, 3, 4, 5              # encoding: [0x39,0x29,0x43,0xec]
            fmsubs. 2, 3, 4, 5
# CHECK-BE: fnmadd 2, 3, 4, 5               # encoding: [0xfc,0x43,0x29,0x3e]
# CHECK-LE: fnmadd 2, 3, 4, 5               # encoding: [0x3e,0x29,0x43,0xfc]
            fnmadd 2, 3, 4, 5
# CHECK-BE: fnmadd. 2, 3, 4, 5              # encoding: [0xfc,0x43,0x29,0x3f]
# CHECK-LE: fnmadd. 2, 3, 4, 5              # encoding: [0x3f,0x29,0x43,0xfc]
            fnmadd. 2, 3, 4, 5
# CHECK-BE: fnmadds 2, 3, 4, 5              # encoding: [0xec,0x43,0x29,0x3e]
# CHECK-LE: fnmadds 2, 3, 4, 5              # encoding: [0x3e,0x29,0x43,0xec]
            fnmadds 2, 3, 4, 5
# CHECK-BE: fnmadds. 2, 3, 4, 5             # encoding: [0xec,0x43,0x29,0x3f]
# CHECK-LE: fnmadds. 2, 3, 4, 5             # encoding: [0x3f,0x29,0x43,0xec]
            fnmadds. 2, 3, 4, 5
# CHECK-BE: fnmsub 2, 3, 4, 5               # encoding: [0xfc,0x43,0x29,0x3c]
# CHECK-LE: fnmsub 2, 3, 4, 5               # encoding: [0x3c,0x29,0x43,0xfc]
            fnmsub 2, 3, 4, 5
# CHECK-BE: fnmsub. 2, 3, 4, 5              # encoding: [0xfc,0x43,0x29,0x3d]
# CHECK-LE: fnmsub. 2, 3, 4, 5              # encoding: [0x3d,0x29,0x43,0xfc]
            fnmsub. 2, 3, 4, 5
# CHECK-BE: fnmsubs 2, 3, 4, 5              # encoding: [0xec,0x43,0x29,0x3c]
# CHECK-LE: fnmsubs 2, 3, 4, 5              # encoding: [0x3c,0x29,0x43,0xec]
            fnmsubs 2, 3, 4, 5
# CHECK-BE: fnmsubs. 2, 3, 4, 5             # encoding: [0xec,0x43,0x29,0x3d]
# CHECK-LE: fnmsubs. 2, 3, 4, 5             # encoding: [0x3d,0x29,0x43,0xec]
            fnmsubs. 2, 3, 4, 5

# Floating-point rounding and conversion instructions

# CHECK-BE: frsp 2, 3                       # encoding: [0xfc,0x40,0x18,0x18]
# CHECK-LE: frsp 2, 3                       # encoding: [0x18,0x18,0x40,0xfc]
            frsp 2, 3
# CHECK-BE: frsp. 2, 3                      # encoding: [0xfc,0x40,0x18,0x19]
# CHECK-LE: frsp. 2, 3                      # encoding: [0x19,0x18,0x40,0xfc]
            frsp. 2, 3

# CHECK-BE: fctid 2, 3                      # encoding: [0xfc,0x40,0x1e,0x5c]
# CHECK-LE: fctid 2, 3                      # encoding: [0x5c,0x1e,0x40,0xfc]
            fctid 2, 3
# CHECK-BE: fctid. 2, 3                     # encoding: [0xfc,0x40,0x1e,0x5d]
# CHECK-LE: fctid. 2, 3                     # encoding: [0x5d,0x1e,0x40,0xfc]
            fctid. 2, 3

# CHECK-BE: fctidu 2, 3                      # encoding: [0xfc,0x40,0x1f,0x5c]
# CHECK-LE: fctidu 2, 3                      # encoding: [0x5c,0x1f,0x40,0xfc]
            fctidu 2, 3
# CHECK-BE: fctidu. 2, 3                     # encoding: [0xfc,0x40,0x1f,0x5d]
# CHECK-LE: fctidu. 2, 3                     # encoding: [0x5d,0x1f,0x40,0xfc]
            fctidu. 2, 3

# CHECK-BE: fctidz 2, 3                     # encoding: [0xfc,0x40,0x1e,0x5e]
# CHECK-LE: fctidz 2, 3                     # encoding: [0x5e,0x1e,0x40,0xfc]
            fctidz 2, 3
# CHECK-BE: fctidz. 2, 3                    # encoding: [0xfc,0x40,0x1e,0x5f]
# CHECK-LE: fctidz. 2, 3                    # encoding: [0x5f,0x1e,0x40,0xfc]
            fctidz. 2, 3

# CHECK-BE: fctiduz 2, 3                    # encoding: [0xfc,0x40,0x1f,0x5e]
# CHECK-LE: fctiduz 2, 3                    # encoding: [0x5e,0x1f,0x40,0xfc]
            fctiduz 2, 3
# CHECK-BE: fctiduz. 2, 3                   # encoding: [0xfc,0x40,0x1f,0x5f]
# CHECK-LE: fctiduz. 2, 3                   # encoding: [0x5f,0x1f,0x40,0xfc]
            fctiduz. 2, 3

# CHECK-BE: fctiw 2, 3                      # encoding: [0xfc,0x40,0x18,0x1c]
# CHECK-LE: fctiw 2, 3                      # encoding: [0x1c,0x18,0x40,0xfc]
            fctiw 2, 3
# CHECK-BE: fctiw. 2, 3                     # encoding: [0xfc,0x40,0x18,0x1d]
# CHECK-LE: fctiw. 2, 3                     # encoding: [0x1d,0x18,0x40,0xfc]
            fctiw. 2, 3

# CHECK-BE: fctiwu 2, 3                      # encoding: [0xfc,0x40,0x19,0x1c]
# CHECK-LE: fctiwu 2, 3                      # encoding: [0x1c,0x19,0x40,0xfc]
            fctiwu 2, 3
# CHECK-BE: fctiwu. 2, 3                     # encoding: [0xfc,0x40,0x19,0x1d]
# CHECK-LE: fctiwu. 2, 3                     # encoding: [0x1d,0x19,0x40,0xfc]
            fctiwu. 2, 3

# CHECK-BE: fctiwz 2, 3                     # encoding: [0xfc,0x40,0x18,0x1e]
# CHECK-LE: fctiwz 2, 3                     # encoding: [0x1e,0x18,0x40,0xfc]
            fctiwz 2, 3
# CHECK-BE: fctiwz. 2, 3                    # encoding: [0xfc,0x40,0x18,0x1f]
# CHECK-LE: fctiwz. 2, 3                    # encoding: [0x1f,0x18,0x40,0xfc]
            fctiwz. 2, 3
# CHECK-BE: fctiwuz 2, 3                    # encoding: [0xfc,0x40,0x19,0x1e]
# CHECK-LE: fctiwuz 2, 3                    # encoding: [0x1e,0x19,0x40,0xfc]
            fctiwuz 2, 3
# CHECK-BE: fctiwuz. 2, 3                   # encoding: [0xfc,0x40,0x19,0x1f]
# CHECK-LE: fctiwuz. 2, 3                   # encoding: [0x1f,0x19,0x40,0xfc]
            fctiwuz. 2, 3
# CHECK-BE: fcfid 2, 3                      # encoding: [0xfc,0x40,0x1e,0x9c]
# CHECK-LE: fcfid 2, 3                      # encoding: [0x9c,0x1e,0x40,0xfc]
            fcfid 2, 3
# CHECK-BE: fcfid. 2, 3                     # encoding: [0xfc,0x40,0x1e,0x9d]
# CHECK-LE: fcfid. 2, 3                     # encoding: [0x9d,0x1e,0x40,0xfc]
            fcfid. 2, 3
# CHECK-BE: fcfidu 2, 3                     # encoding: [0xfc,0x40,0x1f,0x9c]
# CHECK-LE: fcfidu 2, 3                     # encoding: [0x9c,0x1f,0x40,0xfc]
            fcfidu 2, 3
# CHECK-BE: fcfidu. 2, 3                    # encoding: [0xfc,0x40,0x1f,0x9d]
# CHECK-LE: fcfidu. 2, 3                    # encoding: [0x9d,0x1f,0x40,0xfc]
            fcfidu. 2, 3
# CHECK-BE: fcfids 2, 3                     # encoding: [0xec,0x40,0x1e,0x9c]
# CHECK-LE: fcfids 2, 3                     # encoding: [0x9c,0x1e,0x40,0xec]
            fcfids 2, 3
# CHECK-BE: fcfids. 2, 3                    # encoding: [0xec,0x40,0x1e,0x9d]
# CHECK-LE: fcfids. 2, 3                    # encoding: [0x9d,0x1e,0x40,0xec]
            fcfids. 2, 3
# CHECK-BE: fcfidus 2, 3                    # encoding: [0xec,0x40,0x1f,0x9c]
# CHECK-LE: fcfidus 2, 3                    # encoding: [0x9c,0x1f,0x40,0xec]
            fcfidus 2, 3
# CHECK-BE: fcfidus. 2, 3                   # encoding: [0xec,0x40,0x1f,0x9d]
# CHECK-LE: fcfidus. 2, 3                   # encoding: [0x9d,0x1f,0x40,0xec]
            fcfidus. 2, 3
# CHECK-BE: frin 2, 3                       # encoding: [0xfc,0x40,0x1b,0x10]
# CHECK-LE: frin 2, 3                       # encoding: [0x10,0x1b,0x40,0xfc]
            frin 2, 3
# CHECK-BE: frin. 2, 3                      # encoding: [0xfc,0x40,0x1b,0x11]
# CHECK-LE: frin. 2, 3                      # encoding: [0x11,0x1b,0x40,0xfc]
            frin. 2, 3
# CHECK-BE: frip 2, 3                       # encoding: [0xfc,0x40,0x1b,0x90]
# CHECK-LE: frip 2, 3                       # encoding: [0x90,0x1b,0x40,0xfc]
            frip 2, 3
# CHECK-BE: frip. 2, 3                      # encoding: [0xfc,0x40,0x1b,0x91]
# CHECK-LE: frip. 2, 3                      # encoding: [0x91,0x1b,0x40,0xfc]
            frip. 2, 3
# CHECK-BE: friz 2, 3                       # encoding: [0xfc,0x40,0x1b,0x50]
# CHECK-LE: friz 2, 3                       # encoding: [0x50,0x1b,0x40,0xfc]
            friz 2, 3
# CHECK-BE: friz. 2, 3                      # encoding: [0xfc,0x40,0x1b,0x51]
# CHECK-LE: friz. 2, 3                      # encoding: [0x51,0x1b,0x40,0xfc]
            friz. 2, 3
# CHECK-BE: frim 2, 3                       # encoding: [0xfc,0x40,0x1b,0xd0]
# CHECK-LE: frim 2, 3                       # encoding: [0xd0,0x1b,0x40,0xfc]
            frim 2, 3
# CHECK-BE: frim. 2, 3                      # encoding: [0xfc,0x40,0x1b,0xd1]
# CHECK-LE: frim. 2, 3                      # encoding: [0xd1,0x1b,0x40,0xfc]
            frim. 2, 3

# Floating-point compare instructions

# CHECK-BE: fcmpu 2, 3, 4                   # encoding: [0xfd,0x03,0x20,0x00]
# CHECK-LE: fcmpu 2, 3, 4                   # encoding: [0x00,0x20,0x03,0xfd]
            fcmpu 2, 3, 4
# FIXME:    fcmpo 2, 3, 4

# Floating-point select instruction

# CHECK-BE: fsel 2, 3, 4, 5                 # encoding: [0xfc,0x43,0x29,0x2e]
# CHECK-LE: fsel 2, 3, 4, 5                 # encoding: [0x2e,0x29,0x43,0xfc]
            fsel 2, 3, 4, 5
# CHECK-BE: fsel. 2, 3, 4, 5                # encoding: [0xfc,0x43,0x29,0x2f]
# CHECK-LE: fsel. 2, 3, 4, 5                # encoding: [0x2f,0x29,0x43,0xfc]
            fsel. 2, 3, 4, 5

# Floating-point status and control register instructions

# CHECK-BE: mffs 2                          # encoding: [0xfc,0x40,0x04,0x8e]
# CHECK-LE: mffs 2                          # encoding: [0x8e,0x04,0x40,0xfc]
            mffs 2
# CHECK-BE: mffs. 7                         # encoding: [0xfc,0xe0,0x04,0x8f]
# CHECK-LE: mffs. 7                         # encoding: [0x8f,0x04,0xe0,0xfc]
            mffs. 7
# CHECK-BE: mffsce 2                        # encoding: [0xfc,0x41,0x04,0x8e]
# CHECK-LE: mffsce 2                        # encoding: [0x8e,0x04,0x41,0xfc]
            mffsce 2
# CHECK-BE: mffscdrn 2, 3                   # encoding: [0xfc,0x54,0x1c,0x8e]
# CHECK-LE: mffscdrn 2, 3                   # encoding: [0x8e,0x1c,0x54,0xfc]
            mffscdrn 2, 3
# CHECK-BE: mffscdrni 2, 3                  # encoding: [0xfc,0x55,0x1c,0x8e]
# CHECK-LE: mffscdrni 2, 3                  # encoding: [0x8e,0x1c,0x55,0xfc]
            mffscdrni 2, 3
# CHECK-BE: mffscrn 2, 3                    # encoding: [0xfc,0x56,0x1c,0x8e]
# CHECK-LE: mffscrn 2, 3                    # encoding: [0x8e,0x1c,0x56,0xfc]
            mffscrn 2, 3
# CHECK-BE: mffscrni 2, 3                   # encoding: [0xfc,0x57,0x1c,0x8e]
# CHECK-LE: mffscrni 2, 3                   # encoding: [0x8e,0x1c,0x57,0xfc]
            mffscrni 2, 3
# CHECK-BE: mffsl 2                         # encoding: [0xfc,0x58,0x04,0x8e]
# CHECK-LE: mffsl 2                         # encoding: [0x8e,0x04,0x58,0xfc]
            mffsl 2
# CHECK-BE: mcrfs 4, 5                      # encoding: [0xfe,0x14,0x00,0x80]
# CHECK-LE: mcrfs 4, 5                      # encoding: [0x80,0x00,0x14,0xfe]
            mcrfs 4, 5
# CHECK-BE: mtfsfi 5, 2, 1                  # encoding: [0xfe,0x81,0x21,0x0c]
# CHECK-LE: mtfsfi 5, 2, 1                  # encoding: [0x0c,0x21,0x81,0xfe]
            mtfsfi 5, 2, 1
# CHECK-BE: mtfsfi. 5, 2, 1                 # encoding: [0xfe,0x81,0x21,0x0d]
# CHECK-LE: mtfsfi. 5, 2, 1                 # encoding: [0x0d,0x21,0x81,0xfe]
            mtfsfi. 5, 2, 1
# CHECK-BE: mtfsfi 6, 2                     # encoding: [0xff,0x00,0x21,0x0c]
# CHECK-LE: mtfsfi 6, 2                     # encoding: [0x0c,0x21,0x00,0xff]
            mtfsfi 6, 2
# CHECK-BE: mtfsfi. 6, 2                    # encoding: [0xff,0x00,0x21,0x0d]
# CHECK-LE: mtfsfi. 6, 2                    # encoding: [0x0d,0x21,0x00,0xff]
            mtfsfi. 6, 2
# CHECK-BE: mtfsf 127, 8, 1, 1              # encoding: [0xfe,0xff,0x45,0x8e]
# CHECK-LE: mtfsf 127, 8, 1, 1              # encoding: [0x8e,0x45,0xff,0xfe]
            mtfsf 127, 8, 1, 1
# CHECK-BE: mtfsf. 125, 8, 1, 1             # encoding: [0xfe,0xfb,0x45,0x8f]
# CHECK-LE: mtfsf. 125, 8, 1, 1             # encoding: [0x8f,0x45,0xfb,0xfe]
            mtfsf. 125, 8, 1, 1
# CHECK-BE: mtfsf 127, 6                    # encoding: [0xfc,0xfe,0x35,0x8e]
# CHECK-LE: mtfsf 127, 6                    # encoding: [0x8e,0x35,0xfe,0xfc]
            mtfsf 127, 6
# CHECK-BE: mtfsf. 125, 6                   # encoding: [0xfc,0xfa,0x35,0x8f]
# CHECK-LE: mtfsf. 125, 6                   # encoding: [0x8f,0x35,0xfa,0xfc]
            mtfsf. 125, 6
# CHECK-BE: mtfsb0 31                       # encoding: [0xff,0xe0,0x00,0x8c]
# CHECK-LE: mtfsb0 31                       # encoding: [0x8c,0x00,0xe0,0xff]
            mtfsb0 31
# FIXME:    mtfsb0. 31
# CHECK-BE: mtfsb1 31                       # encoding: [0xff,0xe0,0x00,0x4c]
# CHECK-LE: mtfsb1 31                       # encoding: [0x4c,0x00,0xe0,0xff]
            mtfsb1 31
# FIXME:    mtfsb1. 31

