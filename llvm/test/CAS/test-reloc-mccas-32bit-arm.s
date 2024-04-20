# This test tests to make sure mccas can handle scattered relocations properly on 32 bit arm
# REQUIRES: arm-registered-target
# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc --cas=%t/cas --cas-backend --mccas-verify -triple=armv7-apple-darwin10  -filetype=obj -o %t/reloc.o %s

    movw  r0, :lower16:(fn2-L1)
L1:
fn2:

