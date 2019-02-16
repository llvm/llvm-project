# RUN: not llvm-mc %s -triple mips-unknown-linux-gnu -mattr=+fp64 2> %t0 | \
# RUN:   FileCheck %s -check-prefix=CHECK-ASM
# RUN: FileCheck %s -check-prefix=CHECK-ERROR < %t0
#
        .module nooddspreg
# CHECK-ASM: .module nooddspreg

        add.s $f1, $f2, $f5
# CHECK-ERROR: :[[@LINE-1]]:15: error: -mno-odd-spreg prohibits the use of odd FPU registers
# CHECK-ERROR: :[[@LINE-2]]:25: error: -mno-odd-spreg prohibits the use of odd FPU registers

# FIXME: Test should include gnu_attributes directive when implemented.
#        An explicit .gnu_attribute must be checked against the effective
#        command line options and any inconsistencies reported via a warning.
