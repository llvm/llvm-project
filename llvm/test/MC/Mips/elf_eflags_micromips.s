# RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux -mcpu=mips32 \
# RUN:     -mattr=+micromips < /dev/null -o -| llvm-readobj -h | FileCheck \
# RUN:     -check-prefix=NO-MM %s
# RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux -mcpu=mips32 %s -o - \
# RUN:     | llvm-readobj -h | FileCheck %s

# This *MUST* match the output of 'gcc -c' compiled with the same triple.
# CHECK: Flags [ (0x52001004)
# NO-MM: Flags [ (0x50001004)

        .set micromips
f:
        nop
