# RUN: llvm-mc -filetype=obj -triple=powerpc-unknown-linux-gnu %s | llvm-readobj -x .text - | FileCheck -check-prefix=CHECK-BE %s
# RUN: llvm-mc -filetype=obj -triple=ppc32le-unknown-linux-gnu %s | llvm-readobj -x .text - | FileCheck -check-prefix=CHECK-LE %s
add %r0, %r1, %r2
.big
add %r0, %r1, %r2
.little
add %r0, %r1, %r2

# CHECK-BE: 0x00000000 7c011214 7c011214 1412017c          |...|......|
# CHECK-LE: 0x00000000 1412017c 7c011214 1412017c          ...||......|
