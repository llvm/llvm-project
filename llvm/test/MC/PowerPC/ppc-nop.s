# RUN: llvm-mc -filetype=obj -triple=powerpc-unknown-linux-gnu %s | llvm-readobj -S --sd - | FileCheck -check-prefix=CHECK-BE %s
# RUN: llvm-mc -filetype=obj -triple=powerpc64-unknown-linux-gnu %s | llvm-readobj -S --sd - | FileCheck -check-prefix=CHECK-BE %s
# RUN: llvm-mc -filetype=obj -triple=powerpc64le-unknown-linux-gnu %s | llvm-readobj -S --sd - | FileCheck -check-prefix=CHECK-LE %s

blr
.p2align 3
blr

.byte 0x42
.p2align 2

# CHECK-BE:  0000: 4E800020 60000000 4E800020 42000000
# CHECK-LE:  0000: 2000804E 00000060 2000804E 42000000

