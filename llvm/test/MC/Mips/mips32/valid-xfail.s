# Instructions that should be valid but currently fail for known reasons (e.g.
# they aren't implemented yet).
# This test is set up to XPASS if any instruction generates an encoding.
#
# RUN: not llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips32 | not FileCheck %s
# CHECK-NOT: encoding
# XFAIL: *

        .set noat
        c.eq.ps         $fcc5,$f0,$f9
        c.f.ps          $fcc6,$f11,$f11
        c.le.ps         $fcc1,$f7,$f20
        c.lt.ps         $f19,$f5
        c.nge.ps        $f1,$f26
        c.ngl.ps        $f21,$f30
        c.ngle.ps       $fcc7,$f12,$f20
        c.ngt.ps        $fcc5,$f30,$f6
        c.ole.ps        $fcc7,$f21,$f8
        c.olt.ps        $fcc3,$f7,$f16
        c.seq.ps        $fcc6,$f31,$f14
        c.sf.ps         $fcc6,$f4,$f6
        c.ueq.ps        $fcc1,$f5,$f29
        c.ule.ps        $fcc6,$f17,$f3
        c.ult.ps        $fcc7,$f14,$f0
        c.un.ps         $fcc4,$f2,$f26
        rorv            $13,$a3,$s5
