# RUN: llvm-mc -triple i386-unknown-unknown %s
# FIXME: Actually test the output.

        .file 1 "hello"
        .loc 1
        .loc 1 2
        .loc 1 2 3
        .loc 1 2 discriminator 1
