# RUN: llvm-mc -show-encoding -triple=mips-unknown-unknown -mcpu=mips64r2 -mattr=mips3d %s | FileCheck %s
#
        .set noat
        addr.ps $f7, $f11, $f3  # CHECK: addr.ps $f7, $f11, $f3  # encoding: [0x46,0xc3,0x59,0xd8]
        cvt.ps.pw $f3, $f18     # CHECK: cvt.ps.pw $f3, $f18     # encoding: [0x46,0x80,0x90,0xe6]
        cvt.pw.ps $f5, $f20     # CHECK: cvt.pw.ps $f5, $f20     # encoding: [0x46,0xc0,0xa1,0x64]
        mulr.ps $f23, $f5, $f1  # CHECK: mulr.ps $f23, $f5, $f1  # encoding: [0x46,0xc1,0x2d,0xda]
