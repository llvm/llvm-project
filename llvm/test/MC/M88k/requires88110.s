# RUN: llvm-mc %s -triple=m88k-unknown-openbsd -filetype=obj -o - \
# RUN: | llvm-readelf --file-header - | FileCheck %s
    .requires_88110
# CHECK: Machine:                           MC88000
# CHECK:   Flags:                             0x4