# RUN: llvm-mc -triple i386-apple-darwin10 %s | FileCheck %s
# RUN: llvm-mc -triple i386-pc-mingw32 %s | FileCheck %s
# RUN: not llvm-mc -triple i386-linux-gnu %s 2>&1 | FileCheck %s -check-prefix=ERROR

# CHECK: TEST0:
# CHECK: .lcomm a,7,4
# CHECK: .lcomm b,8
# CHECK: .lcomm c,0

# ELF doesn't like alignment on .lcomm.
# ERROR: alignment not supported on this target
TEST0:  
        .lcomm a, 8-1, 4
        .lcomm b,8
        .lcomm  c,  0
