# RUN: llvm-mc -triple i386-unknown-unknown %s | FileCheck %s
# RUN: llvm-mc -triple i386-unknown-unknown %s -filetype=null

        .file 1 "hello"
# CHECK: .file 1 "hello"
        .loc 1
        .loc 1 2
# CHECK: .loc 1 2 0
        .loc 1 2 3
# CHECK: .loc 1 2 3
        .loc 1 2 discriminator 1
# CHECK: 1 2 0 discriminator 1
        .loc 1 2 0 isa 3
# CHECK: 1 2 0 isa 3
        .loc 1 0
