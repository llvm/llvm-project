// RUN: llvm-mc -triple i386-unknown-unknown %s

        .data
        t0_v0 = 1
        t0_v1 = t0_v0
        .if t0_v1 != 1
        .abort "invalid value"
        .endif

        t1_v0 = 1
        t1_v1 = t0_v0
        t1_v0 = 2
        .if t0_v1 != 1
        .abort "invalid value"
        .endif
