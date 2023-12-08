# Instructions that should be valid but currently fail for known reasons (e.g.
# they aren't implemented yet).
# This test is set up to XPASS if any instruction generates an encoding.
#
# RUN: not llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips1 | not FileCheck %s
# CHECK-NOT: encoding
# XFAIL: *

        .set noat
        lwc0    c0_entrylo,-7321($s2)
        swc0    c0_prid,18904($s3)
