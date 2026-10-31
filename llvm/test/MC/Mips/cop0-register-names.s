# RUN: llvm-mc -triple=mips -mcpu=mips32r2 -show-encoding %s | FileCheck %s
# RUN: llvm-mc -triple=mips -mcpu=mips32r2 -filetype=obj %s | llvm-objdump -d - | FileCheck %s --check-prefix=DIS
# RUN: llvm-mc -triple=mips -mcpu=mips32r2 -mattr=+micromips -filetype=obj %s | llvm-objdump -d - | FileCheck %s --check-prefix=DIS
# RUN: llvm-mc -triple=mips -mcpu=mips32r6 -mattr=+micromips -filetype=obj %s | llvm-objdump -d - | FileCheck %s --check-prefix=DIS

        mfc0    $2, $status                 # CHECK: mfc0 $2, $12, 0          # encoding: [0x40,0x02,0x60,0x00]
                                            # DIS: mfc0 $2, $12, 0
        mfc0    $2, $intctl                 # CHECK: mfc0 $2, $12, 1          # encoding: [0x40,0x02,0x60,0x01]
                                            # DIS: mfc0 $2, $12, 1
        mfc0    $2, $config                 # CHECK: mfc0 $2, $16, 0          # encoding: [0x40,0x02,0x80,0x00]
                                            # DIS: mfc0 $2, $16, 0
        mfc0    $2, $config1                # CHECK: mfc0 $2, $16, 1          # encoding: [0x40,0x02,0x80,0x01]
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

# COP0 register names from PRA revision 6.09.
        mfc0    $2, $vpcontrol              # CHECK: mfc0 $2, $0, 4           # encoding: [0x40,0x02,0x00,0x04]
                                            # DIS: mfc0 $2, $0, 4
        mfc0    $2, $vpeopt                 # CHECK: mfc0 $2, $1, 7           # encoding: [0x40,0x02,0x08,0x07]
                                            # DIS: mfc0 $2, $1, 7
        mfc0    $2, $globalnumber           # CHECK: mfc0 $2, $3, 1           # encoding: [0x40,0x02,0x18,0x01]
                                            # DIS: mfc0 $2, $3, 1
        mfc0    $2, $tcopt                  # CHECK: mfc0 $2, $3, 7           # encoding: [0x40,0x02,0x18,0x07]
                                            # DIS: mfc0 $2, $3, 7
        mfc0    $2, $xcontext_config        # CHECK: mfc0 $2, $4, 3           # encoding: [0x40,0x02,0x20,0x03]
                                            # DIS: mfc0 $2, $4, 3
        mfc0    $2, $debugcontextid         # CHECK: mfc0 $2, $4, 4           # encoding: [0x40,0x02,0x20,0x04]
                                            # DIS: mfc0 $2, $4, 4
        mfc0    $2, $memorymapid            # CHECK: mfc0 $2, $4, 5           # encoding: [0x40,0x02,0x20,0x05]
                                            # DIS: mfc0 $2, $4, 5
        mfc0    $2, $view_ipl               # CHECK: mfc0 $2, $12, 4          # encoding: [0x40,0x02,0x60,0x04]
                                            # DIS: mfc0 $2, $12, 4
        mfc0    $2, $srsmap2                # CHECK: mfc0 $2, $12, 5          # encoding: [0x40,0x02,0x60,0x05]
                                            # DIS: mfc0 $2, $12, 5
        mfc0    $2, $view_ripl              # CHECK: mfc0 $2, $13, 4          # encoding: [0x40,0x02,0x68,0x04]
                                            # DIS: mfc0 $2, $13, 4
        mfc0    $2, $nestedexc              # CHECK: mfc0 $2, $13, 5          # encoding: [0x40,0x02,0x68,0x05]
                                            # DIS: mfc0 $2, $13, 5
        mfc0    $2, $nestedepc              # CHECK: mfc0 $2, $14, 2          # encoding: [0x40,0x02,0x70,0x02]
                                            # DIS: mfc0 $2, $14, 2
        mfc0    $2, $bevva                  # CHECK: mfc0 $2, $15, 4          # encoding: [0x40,0x02,0x78,0x04]
                                            # DIS: mfc0 $2, $15, 4
