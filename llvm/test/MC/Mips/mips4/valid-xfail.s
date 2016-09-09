# Instructions that should be valid but currently fail for known reasons (e.g.
# they aren't implemented yet).
# This test is set up to XPASS if any instruction generates an encoding.
#
# RUN: not llvm-mc %s -triple=mips64-unknown-linux -show-encoding -mcpu=mips4 | not FileCheck %s
# CHECK-NOT: encoding
# XFAIL: *

        .set noat
        c.eq.d          $fcc1,$f15,$f15
        c.eq.s          $fcc5,$f24,$f17
        c.f.d           $fcc4,$f11,$f21
        c.f.s           $fcc4,$f30,$f7
        c.le.d          $fcc4,$f18,$f1
        c.le.s          $fcc6,$f24,$f4
        c.lt.d          $fcc3,$f9,$f3
        c.lt.s          $fcc2,$f17,$f14
        c.nge.d         $fcc5,$f21,$f16
        c.nge.s         $fcc3,$f11,$f8
        c.ngl.s         $fcc2,$f31,$f23
        c.ngle.s        $fcc2,$f18,$f23
        c.ngt.d         $fcc4,$f24,$f7
        c.ngt.s         $fcc5,$f8,$f13
        c.ole.d         $fcc2,$f16,$f31
        c.ole.s         $fcc3,$f7,$f20
        c.olt.d         $fcc4,$f19,$f28
        c.olt.s         $fcc6,$f20,$f7
        c.seq.d         $fcc4,$f31,$f7
        c.seq.s         $fcc7,$f1,$f25
        c.ueq.d         $fcc4,$f13,$f25
        c.ueq.s         $fcc6,$f3,$f30
        c.ule.d         $fcc7,$f25,$f18
        c.ule.s         $fcc7,$f21,$f30
        c.ult.d         $fcc6,$f6,$f17
        c.ult.s         $fcc7,$f24,$f10
        c.un.d          $fcc6,$f23,$f24
        c.un.s          $fcc1,$f30,$f4
        recip.d         $f19,$f6
        recip.s         $f3,$f30
        rsqrt.d         $f3,$f28
        rsqrt.s         $f4,$f8
