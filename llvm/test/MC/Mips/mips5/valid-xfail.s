# Instructions that should be valid but currently fail for known reasons (e.g.
# they aren't implemented yet).
# This test is set up to XPASS if any instruction generates an encoding.
#
# RUN: not llvm-mc %s -triple=mips64-unknown-linux -show-encoding -mcpu=mips5 | not FileCheck %s
# CHECK-NOT: encoding
# XFAIL: *

        .set noat
        abs.ps          $f22,$f8
        add.ps          $f25,$f27,$f13
        alnv.ps         $f12,$f18,$f30,$12
        c.eq.d          $fcc1,$f15,$f15
        c.eq.ps         $fcc5,$f0,$f9
        c.eq.s          $fcc5,$f24,$f17
        c.f.d           $fcc4,$f11,$f21
        c.f.ps          $fcc6,$f11,$f11
        c.f.s           $fcc4,$f30,$f7
        c.le.d          $fcc4,$f18,$f1
        c.le.ps         $fcc1,$f7,$f20
        c.le.s          $fcc6,$f24,$f4
        c.lt.d          $fcc3,$f9,$f3
        c.lt.ps         $f19,$f5
        c.lt.s          $fcc2,$f17,$f14
        c.nge.d         $fcc5,$f21,$f16
        c.nge.ps        $f1,$f26
        c.nge.s         $fcc3,$f11,$f8
        c.ngl.ps        $f21,$f30
        c.ngl.s         $fcc2,$f31,$f23
        c.ngle.ps       $fcc7,$f12,$f20
        c.ngle.s        $fcc2,$f18,$f23
        c.ngt.d         $fcc4,$f24,$f7
        c.ngt.ps        $fcc5,$f30,$f6
        c.ngt.s         $fcc5,$f8,$f13
        c.ole.d         $fcc2,$f16,$f31
        c.ole.ps        $fcc7,$f21,$f8
        c.ole.s         $fcc3,$f7,$f20
        c.olt.d         $fcc4,$f19,$f28
        c.olt.ps        $fcc3,$f7,$f16
        c.olt.s         $fcc6,$f20,$f7
        c.seq.d         $fcc4,$f31,$f7
        c.seq.ps        $fcc6,$f31,$f14
        c.seq.s         $fcc7,$f1,$f25
        c.sf.ps         $fcc6,$f4,$f6
        c.ueq.d         $fcc4,$f13,$f25
        c.ueq.ps        $fcc1,$f5,$f29
        c.ueq.s         $fcc6,$f3,$f30
        c.ule.d         $fcc7,$f25,$f18
        c.ule.ps        $fcc6,$f17,$f3
        c.ule.s         $fcc7,$f21,$f30
        c.ult.d         $fcc6,$f6,$f17
        c.ult.ps        $fcc7,$f14,$f0
        c.ult.s         $fcc7,$f24,$f10
        c.un.d          $fcc6,$f23,$f24
        c.un.ps         $fcc4,$f2,$f26
        c.un.s          $fcc1,$f30,$f4
        cvt.ps.s        $f3,$f18,$f19
        cvt.s.pl        $f30,$f1
        cvt.s.pu        $f14,$f25
        madd.ps         $f22,$f3,$f14,$f3
        mov.ps          $f22,$f17
        movf.ps         $f10,$f28,$fcc6
        movn.ps         $f31,$f31,$s3
        movt.ps         $f20,$f25,$fcc2
        movz.ps         $f18,$f17,$ra
        msub.ps         $f12,$f14,$f29,$f17
        mul.ps          $f14,$f0,$f16
        neg.ps          $f19,$f13
        nmadd.ps        $f27,$f4,$f9,$f25
        nmsub.ps        $f6,$f12,$f14,$f17
        pll.ps          $f25,$f9,$f30
        plu.ps          $f1,$f26,$f29
        pul.ps          $f9,$f30,$f26
        puu.ps          $f24,$f9,$f2
        recip.d         $f19,$f6
        recip.s         $f3,$f30
        rsqrt.d         $f3,$f28
        rsqrt.s         $f4,$f8
        sub.ps          $f5,$f14,$f26
