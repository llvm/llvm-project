// RUN: llvm-mc -triple i386-apple-darwin9 %s -filetype=obj -o - | llvm-readobj -S --sd | FileCheck %s

// FIXME: This is a horrible way of checking the output, we need an llvm-mc
// based 'otool'. Use:
//   (f=relax-jumps;
//    llvm-mc -filetype=obj -o $f.mc.o $f.s &&
//    as -arch i386 -o $f.as.o $f.s &&
//    otool -tvr $f.mc.o | tail +2 > $f.mc.dump &&
//    otool -tvr $f.as.o | tail +2 > $f.as.dump &&
//    diff $f.{as,mc}.dump)
// to examine the results in a more sensible fashion.

L1:
        .space 200, 0x90

        je L1
        jb L2
        ja L1
        jg L2
        js L1
        jae L2
        jns L1

        .space 200, 0x90
L2:

        xorl %eax, %eax

// CHECK: SectionData (
// CHECK:   00C0: 90909090 90909090 0F8432FF FFFF0F82  |..........2.....|
// CHECK:   00D0: E6000000 0F8726FF FFFF0F8F DA000000  |......&.........|
// CHECK:   00E0: 0F881AFF FFFF0F83 CE000000 0F890EFF  |................|
// CHECK:   00F0: FFFF9090 90909090 90909090 90909090  |................|
// CHECK:   01A0: 90909090 90909090 90909090 90909090  |................|
// CHECK:   01B0: 90909090 90909090 909031C0           |..........1.|
// CHECK: )
