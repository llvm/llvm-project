# RUN: llvm-mc -triple=powerpc64 -mattr=+ppcas -show-encoding < %s | FileCheck %s
# RUN: llvm-mc -triple=powerpc64 -mattr=+ppcas -filetype=obj < %s \
# RUN:   | llvm-objdump -d --mattr=+ppcas - | FileCheck %s --check-prefix=DIS
# RUN: not llvm-mc -triple=powerpc64 -mattr=-ppcas < %s 2>&1 \
# RUN:   | FileCheck %s --check-prefix=NO-PPCAS

        lq 4, 0(6)
        stq 26, 0(11)
        settag
        scv 7

# CHECK: lq 4, 0(6){{.*}}encoding: [0xe0,0x86,0x00,0x02]
# CHECK: stq 26, 0(11){{.*}}encoding: [0xfb,0x4b,0x00,0x02]
# CHECK: settag{{.*}}encoding: [0x7c,0x01,0x03,0xe6]
# CHECK: scv 7{{.*}}encoding: [0x44,0x00,0x00,0xe1]

# DIS: lq 4, 0(6)
# DIS: stq 26, 0(11)
# DIS: settag
# DIS: scv 7

# NO-PPCAS: error: instruction use requires an option to be enabled
