# RUN: not llvm-mc -triple=mips64 -mcpu=mips3 -show-encoding %s > %t.out 2> %t.err
# RUN: FileCheck %s --check-prefix=ERR < %t.err
# RUN: FileCheck %s --check-prefix=ENC < %t.out

# Named registers carry the same selector restrictions as numeric REG/SEL pairs.
        mfc0    $2, $config                 # ENC: mfc0 $2, $16, 0            # encoding: [0x40,0x02,0x80,0x00]
        mtc0    $2, $16                     # ENC: mtc0 $2, $16, 0            # encoding: [0x40,0x82,0x80,0x00]
        mfc0    $2, $config1                # ERR: :[[@LINE]]:21: error: selector must be zero for pre-MIPS32 ISAs
        mtc0    $2, $16, 1                  # ERR: :[[@LINE]]:21: error: selector must be zero for pre-MIPS32 ISAs
        dmfc0   $2, $config1                # ERR: :[[@LINE]]:21: error: selector must be zero for pre-MIPS32 ISAs
        dmtc0   $2, $16, 1                  # ERR: :[[@LINE]]:21: error: selector must be zero for pre-MIPS32 ISAs

        .set mips64
        mfc0    $2, $config1                # ENC: mfc0 $2, $16, 1            # encoding: [0x40,0x02,0x80,0x01]
        mtc0    $2, $16, 1                  # ENC: mtc0 $2, $16, 1            # encoding: [0x40,0x82,0x80,0x01]
        dmfc0   $2, $config1                # ENC: dmfc0 $2, $16, 1           # encoding: [0x40,0x22,0x80,0x01]
        dmtc0   $2, $16, 1                  # ENC: dmtc0 $2, $16, 1           # encoding: [0x40,0xa2,0x80,0x01]
