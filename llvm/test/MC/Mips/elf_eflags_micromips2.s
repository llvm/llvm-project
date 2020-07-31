# RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux -mcpu=mips32 %s -o - \
# RUN:     | llvm-readobj -h - | FileCheck %s

# This *MUST* match the output of 'gcc -c' compiled with the same triple.
# The microMIPS flag is not set if no microMIPS code was emitted.
# CHECK: Flags [ (0x50001004)

        nop
        .set micromips
        .set nomicromips
        nop
