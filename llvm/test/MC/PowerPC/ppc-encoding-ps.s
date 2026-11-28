# RUN: llvm-mc -triple powerpc-unknown-unknown -mcpu=750cl --show-encoding %s | FileCheck %s

# CHECK: psq_l 2, 8(3), 0, 1              # encoding: [0xe0,0x43,0x10,0x08]
         psq_l 2, 8(3), 0, 1
# CHECK: psq_l 5, -8(0), 1, 7             # encoding: [0xe0,0xa0,0xff,0xf8]
         psq_l 5, -8(0), 1, 7
# CHECK: psq_l 31, 2047(31), 0, 0         # encoding: [0xe3,0xff,0x07,0xff]
         psq_l 31, 2047(31), 0, 0
# CHECK: psq_l 0, -2048(1), 1, 3          # encoding: [0xe0,0x01,0xb8,0x00]
         psq_l 0, -2048(1), 1, 3
# CHECK: psq_lu 4, 12(5), 1, 2            # encoding: [0xe4,0x85,0xa0,0x0c]
         psq_lu 4, 12(5), 1, 2
# CHECK: psq_lx 6, 7, 8, 0, 3             # encoding: [0x10,0xc7,0x41,0x8c]
         psq_lx 6, 7, 8, 0, 3
# CHECK: psq_lux 9, 10, 11, 1, 4          # encoding: [0x11,0x2a,0x5e,0x4c]
         psq_lux 9, 10, 11, 1, 4
# CHECK: psq_st 2, 8(3), 0, 1             # encoding: [0xf0,0x43,0x10,0x08]
         psq_st 2, 8(3), 0, 1
# CHECK: psq_stu 4, -16(5), 1, 2          # encoding: [0xf4,0x85,0xaf,0xf0]
         psq_stu 4, -16(5), 1, 2
# CHECK: psq_stx 6, 7, 8, 0, 3            # encoding: [0x10,0xc7,0x41,0x8e]
         psq_stx 6, 7, 8, 0, 3
# CHECK: psq_stux 9, 10, 11, 1, 4         # encoding: [0x11,0x2a,0x5e,0x4e]
         psq_stux 9, 10, 11, 1, 4
# CHECK: ps_div 2, 3, 4                   # encoding: [0x10,0x43,0x20,0x24]
         ps_div 2, 3, 4
# CHECK: ps_div. 2, 3, 4                  # encoding: [0x10,0x43,0x20,0x25]
         ps_div. 2, 3, 4
# CHECK: ps_sub 2, 3, 4                   # encoding: [0x10,0x43,0x20,0x28]
         ps_sub 2, 3, 4
# CHECK: ps_sub. 2, 3, 4                  # encoding: [0x10,0x43,0x20,0x29]
         ps_sub. 2, 3, 4
# CHECK: ps_add 2, 3, 4                   # encoding: [0x10,0x43,0x20,0x2a]
         ps_add 2, 3, 4
# CHECK: ps_add. 2, 3, 4                  # encoding: [0x10,0x43,0x20,0x2b]
         ps_add. 2, 3, 4
# CHECK: ps_mul 2, 3, 4                   # encoding: [0x10,0x43,0x01,0x32]
         ps_mul 2, 3, 4
# CHECK: ps_mul. 2, 3, 4                  # encoding: [0x10,0x43,0x01,0x33]
         ps_mul. 2, 3, 4
# CHECK: ps_muls0 2, 3, 4                 # encoding: [0x10,0x43,0x01,0x18]
         ps_muls0 2, 3, 4
# CHECK: ps_muls0. 2, 3, 4                # encoding: [0x10,0x43,0x01,0x19]
         ps_muls0. 2, 3, 4
# CHECK: ps_muls1 2, 3, 4                 # encoding: [0x10,0x43,0x01,0x1a]
         ps_muls1 2, 3, 4
# CHECK: ps_muls1. 2, 3, 4                # encoding: [0x10,0x43,0x01,0x1b]
         ps_muls1. 2, 3, 4
# CHECK: ps_sum0 2, 3, 4, 5               # encoding: [0x10,0x43,0x29,0x14]
         ps_sum0 2, 3, 4, 5
# CHECK: ps_sum0. 2, 3, 4, 5              # encoding: [0x10,0x43,0x29,0x15]
         ps_sum0. 2, 3, 4, 5
# CHECK: ps_sum1 2, 3, 4, 5               # encoding: [0x10,0x43,0x29,0x16]
         ps_sum1 2, 3, 4, 5
# CHECK: ps_sum1. 2, 3, 4, 5              # encoding: [0x10,0x43,0x29,0x17]
         ps_sum1. 2, 3, 4, 5
# CHECK: ps_madds0 2, 3, 4, 5             # encoding: [0x10,0x43,0x29,0x1c]
         ps_madds0 2, 3, 4, 5
# CHECK: ps_madds0. 2, 3, 4, 5            # encoding: [0x10,0x43,0x29,0x1d]
         ps_madds0. 2, 3, 4, 5
# CHECK: ps_madds1 2, 3, 4, 5             # encoding: [0x10,0x43,0x29,0x1e]
         ps_madds1 2, 3, 4, 5
# CHECK: ps_madds1. 2, 3, 4, 5            # encoding: [0x10,0x43,0x29,0x1f]
         ps_madds1. 2, 3, 4, 5
# CHECK: ps_sel 2, 3, 4, 5                # encoding: [0x10,0x43,0x29,0x2e]
         ps_sel 2, 3, 4, 5
# CHECK: ps_sel. 2, 3, 4, 5               # encoding: [0x10,0x43,0x29,0x2f]
         ps_sel. 2, 3, 4, 5
# CHECK: ps_msub 2, 3, 4, 5               # encoding: [0x10,0x43,0x29,0x38]
         ps_msub 2, 3, 4, 5
# CHECK: ps_msub. 2, 3, 4, 5              # encoding: [0x10,0x43,0x29,0x39]
         ps_msub. 2, 3, 4, 5
# CHECK: ps_madd 2, 3, 4, 5               # encoding: [0x10,0x43,0x29,0x3a]
         ps_madd 2, 3, 4, 5
# CHECK: ps_madd. 2, 3, 4, 5              # encoding: [0x10,0x43,0x29,0x3b]
         ps_madd. 2, 3, 4, 5
# CHECK: ps_nmsub 2, 3, 4, 5              # encoding: [0x10,0x43,0x29,0x3c]
         ps_nmsub 2, 3, 4, 5
# CHECK: ps_nmsub. 2, 3, 4, 5             # encoding: [0x10,0x43,0x29,0x3d]
         ps_nmsub. 2, 3, 4, 5
# CHECK: ps_nmadd 2, 3, 4, 5              # encoding: [0x10,0x43,0x29,0x3e]
         ps_nmadd 2, 3, 4, 5
# CHECK: ps_nmadd. 2, 3, 4, 5             # encoding: [0x10,0x43,0x29,0x3f]
         ps_nmadd. 2, 3, 4, 5
# CHECK: ps_res 2, 3                      # encoding: [0x10,0x40,0x18,0x30]
         ps_res 2, 3
# CHECK: ps_res. 2, 3                     # encoding: [0x10,0x40,0x18,0x31]
         ps_res. 2, 3
# CHECK: ps_rsqrte 2, 3                   # encoding: [0x10,0x40,0x18,0x34]
         ps_rsqrte 2, 3
# CHECK: ps_rsqrte. 2, 3                  # encoding: [0x10,0x40,0x18,0x35]
         ps_rsqrte. 2, 3
# CHECK: ps_neg 2, 3                      # encoding: [0x10,0x40,0x18,0x50]
         ps_neg 2, 3
# CHECK: ps_neg. 2, 3                     # encoding: [0x10,0x40,0x18,0x51]
         ps_neg. 2, 3
# CHECK: ps_mr 2, 3                       # encoding: [0x10,0x40,0x18,0x90]
         ps_mr 2, 3
# CHECK: ps_mr. 2, 3                      # encoding: [0x10,0x40,0x18,0x91]
         ps_mr. 2, 3
# CHECK: ps_nabs 2, 3                     # encoding: [0x10,0x40,0x19,0x10]
         ps_nabs 2, 3
# CHECK: ps_nabs. 2, 3                    # encoding: [0x10,0x40,0x19,0x11]
         ps_nabs. 2, 3
# CHECK: ps_abs 2, 3                      # encoding: [0x10,0x40,0x1a,0x10]
         ps_abs 2, 3
# CHECK: ps_abs. 2, 3                     # encoding: [0x10,0x40,0x1a,0x11]
         ps_abs. 2, 3
# CHECK: ps_merge00 2, 3, 4               # encoding: [0x10,0x43,0x24,0x20]
         ps_merge00 2, 3, 4
# CHECK: ps_merge00. 2, 3, 4              # encoding: [0x10,0x43,0x24,0x21]
         ps_merge00. 2, 3, 4
# CHECK: ps_merge01 2, 3, 4               # encoding: [0x10,0x43,0x24,0x60]
         ps_merge01 2, 3, 4
# CHECK: ps_merge01. 2, 3, 4              # encoding: [0x10,0x43,0x24,0x61]
         ps_merge01. 2, 3, 4
# CHECK: ps_merge10 2, 3, 4               # encoding: [0x10,0x43,0x24,0xa0]
         ps_merge10 2, 3, 4
# CHECK: ps_merge10. 2, 3, 4              # encoding: [0x10,0x43,0x24,0xa1]
         ps_merge10. 2, 3, 4
# CHECK: ps_merge11 2, 3, 4               # encoding: [0x10,0x43,0x24,0xe0]
         ps_merge11 2, 3, 4
# CHECK: ps_merge11. 2, 3, 4              # encoding: [0x10,0x43,0x24,0xe1]
         ps_merge11. 2, 3, 4
# CHECK: ps_cmpu0 2, 3, 4                 # encoding: [0x11,0x03,0x20,0x00]
         ps_cmpu0 2, 3, 4
# CHECK: ps_cmpo0 2, 3, 4                 # encoding: [0x11,0x03,0x20,0x40]
         ps_cmpo0 2, 3, 4
# CHECK: ps_cmpu1 2, 3, 4                 # encoding: [0x11,0x03,0x20,0x80]
         ps_cmpu1 2, 3, 4
# CHECK: ps_cmpo1 2, 3, 4                 # encoding: [0x11,0x03,0x20,0xc0]
         ps_cmpo1 2, 3, 4
# CHECK: dcbz_l 3, 4                      # encoding: [0x10,0x03,0x27,0xec]
         dcbz_l 3, 4
