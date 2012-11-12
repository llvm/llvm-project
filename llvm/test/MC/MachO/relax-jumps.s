// RUN: llvm-mc -triple i386-apple-darwin9 %s -filetype=obj -o - | macho-dump --dump-section-data | FileCheck %s

// FIXME: This is a horrible way of checking the output, we need an llvm-mc
// based 'otool'. Use:
//   (f=relax-jumps;
//    llvm-mc -filetype=obj -o $f.mc.o $f.s &&
//    as -arch i386 -o $f.as.o $f.s &&
//    otool -tvr $f.mc.o | tail +2 > $f.mc.dump &&
//    otool -tvr $f.as.o | tail +2 > $f.as.dump &&
//    diff $f.{as,mc}.dump)
// to examine the results in a more sensible fashion.

// CHECK: ('_section_data', '90
// CHECK: 0f8432ff ffff0f82 e6000000 0f8726ff ffff0f8f da000000 0f881aff ffff0f83 ce000000 0f890eff ffff90
// CHECK: 9031c0')

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
