# RUN: llvm-mc -triple=powerpc64 -mattr=+ppcas -show-encoding < %s | FileCheck %s
# RUN: llvm-mc -triple=powerpc64 -mattr=+ppcas -filetype=obj < %s \
# RUN:   | llvm-objdump -d --mattr=+ppcas - | FileCheck %s --check-prefix=DIS
# RUN: not llvm-mc -triple=powerpc64 -mattr=-ppcas < %s 2>&1 \
# RUN:   | FileCheck %s --check-prefix=NO-PPCAS

        lq 4, 0(6)
        lq 20, 0(22), 3
        stq 26, 0(11)
        settag
        scv 7
        lmd 16, -288(1)
        stmd 16, -288(1)
        txer 1, 0, 7
        lsdx 8, 27, 26
        lsdi 8, 27, 16
        stsdx 8, 30, 26
        stsdi 8, 25, 16
        selii 24, -1, 0, 18
        selir 26, 0, 26, 6
        selri 24, 23, 0, 18
        selrr 24, 26, 24, 12

# CHECK: lq 4, 0(6){{.*}}encoding: [0xe0,0x86,0x00,0x02]
# CHECK: lq 20, 0(22), 3{{.*}}encoding: [0xe2,0x96,0x00,0x0f]
# CHECK: stq 26, 0(11){{.*}}encoding: [0xfb,0x4b,0x00,0x02]
# CHECK: settag{{.*}}encoding: [0x7c,0x01,0x03,0xe6]
# CHECK: scv 7{{.*}}encoding: [0x44,0x00,0x00,0xe1]
# CHECK: lmd 16, -288(1){{.*}}encoding: [0xea,0x01,0xfe,0xe3]
# CHECK: stmd 16, -288(1){{.*}}encoding: [0xfa,0x01,0xfe,0xe3]
# CHECK: txer 1, 0, 7{{.*}}encoding: [0x7c,0x20,0x01,0xc8]
# CHECK: lsdx 8, 27, 26{{.*}}encoding: [0x7d,0x1b,0xd4,0x6a]
# CHECK: lsdi 8, 27, 16{{.*}}encoding: [0x7d,0x1b,0x84,0xea]
# CHECK: stsdx 8, 30, 26{{.*}}encoding: [0x7d,0x1e,0xd5,0x6a]
# CHECK: stsdi 8, 25, 16{{.*}}encoding: [0x7d,0x19,0x85,0xea]
# CHECK: selii 24, -1, 0, 18{{.*}}encoding: [0x7b,0xf8,0x04,0x98]
# CHECK: selir 26, 0, 26, 6{{.*}}encoding: [0x78,0x1a,0xd1,0x9a]
# CHECK: selri 24, 23, 0, 18{{.*}}encoding: [0x7a,0xf8,0x04,0x9c]
# CHECK: selrr 24, 26, 24, 12{{.*}}encoding: [0x7b,0x58,0xc3,0x1e]

# DIS: lq 4, 0(6)
# DIS: lq 20, 0(22), 3
# DIS: stq 26, 0(11)
# DIS: settag
# DIS: scv 7
# DIS: lmd 16, -288(1)
# DIS: stmd 16, -288(1)
# DIS: txer 1, 0, 7
# DIS: lsdx 8, 27, 26
# DIS: lsdi 8, 27, 16
# DIS: stsdx 8, 30, 26
# DIS: stsdi 8, 25, 16
# DIS: selii 24, -1, 0, 18
# DIS: selir 26, 0, 26, 6
# DIS: selri 24, 23, 0, 18
# DIS: selrr 24, 26, 24, 12

# NO-PPCAS: error: instruction use requires an option to be enabled
