# RUN: not llvm-mc %s -triple mips-unknown-linux -mcpu=mips64r2 2>&1 | FileCheck %s
        .set fp=xx
# CHECK: error: '.set fp=xx' requires the O32 ABI
# CHECK: .set fp=xx
# CHECK:          ^
