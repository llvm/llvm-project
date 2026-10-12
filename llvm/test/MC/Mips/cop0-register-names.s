# RUN: llvm-mc -triple=mips -mcpu=mips32r2 -show-encoding %s | FileCheck %s
# RUN: llvm-mc -triple=mips -mcpu=mips32r2 -filetype=obj %s | llvm-objdump -d - | FileCheck %s --check-prefix=DIS

        mfc0    $2, $status                 # CHECK: mfc0 $2, $12, 0          # encoding: [0x40,0x02,0x60,0x00]
                                            # DIS: mfc0 $2, $12, 0
        mfc0    $2, $intctl                 # CHECK: mfc0 $2, $12, 1          # encoding: [0x40,0x02,0x60,0x01]
                                            # DIS: mfc0 $2, $12, 1
        mfc0    $2, $config                 # CHECK: mfc0 $2, $16, 0          # encoding: [0x40,0x02,0x80,0x00]
                                            # DIS: mfc0 $2, $16, 0
        mfc0    $2, $config1                # CHECK: mfc0 $2, $16, 1          # encoding: [0x40,0x02,0x80,0x01]
                                            # DIS: mfc0 $2, $16, 1
        mfc0    $2, $config1, 1             # CHECK: mfc0 $2, $16, 1          # encoding: [0x40,0x02,0x80,0x01]
                                            # DIS: mfc0 $2, $16, 1
        mtc0    $2, $ebase                  # CHECK: mtc0 $2, $15, 1          # encoding: [0x40,0x82,0x78,0x01]
                                            # DIS: mtc0 $2, $15, 1
        mtc0    $2, $kscratch6              # CHECK: mtc0 $2, $31, 7          # encoding: [0x40,0x82,0xf8,0x07]
                                            # DIS: mtc0 $2, $31, 7

# Numeric register aliases and absolute selector expressions are also accepted.
        .set cp0reg, $16
        .set cp0sel, 1
        mfc0    $2, cp0reg, cp0sel          # CHECK: mfc0 $2, $16, 1          # encoding: [0x40,0x02,0x80,0x01]
                                            # DIS: mfc0 $2, $16, 1
        .set cp0name, $config1
        mfc0    $2, cp0name                 # CHECK: mfc0 $2, $16, 1          # encoding: [0x40,0x02,0x80,0x01]
                                            # DIS: mfc0 $2, $16, 1
