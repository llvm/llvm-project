# RUN: llvm-mc -filetype=obj -triple x86_64 %s -o %t
# RUN: llvm-readelf -SsX %t | FileCheck %s
# RUN: llvm-readelf -g %t | FileCheck %s --check-prefix=GROUP

## Test that we produce the group sections and that they are before the members

# CHECK:       [ 3] .group            GROUP           0000000000000000 {{.*}} 00000c 04     14   1  4
# CHECK-NEXT:  [ 4] .foo              PROGBITS        0000000000000000 {{.*}} 000001 00 AXG  0   0  1
# CHECK-NEXT:  [ 5] .bar              PROGBITS        0000000000000000 {{.*}} 000001 00 AXG  0   0  1
# CHECK-NEXT:  [ 6] .group            GROUP           0000000000000000 {{.*}} 000008 04     14   2  4
# CHECK-NEXT:  [ 7] .zed              PROGBITS        0000000000000000 {{.*}} 000001 00 AXG  0   0  1
# CHECK-NEXT:  [ 8] .group            GROUP           0000000000000000 {{.*}} 00000c 04     14   4  4
# CHECK-NEXT:  [ 9] .baz              PROGBITS        0000000000000000 {{.*}} 000004 00 AXG  0   0  1
# CHECK-NEXT:  [10] .rela.baz         RELA            0000000000000000 {{.*}} 000018 18   G 14   9  8
# CHECK-NEXT:  [11] sec               PROGBITS        0000000000000000 {{.*}} 000000 00      0   0  1
# CHECK-NEXT:  [12] .group            GROUP           0000000000000000 {{.*}} 000008 04     14   3  4
# CHECK-NEXT:  [13] .qux              PROGBITS        0000000000000000 {{.*}} 000000 00 AXG  0   0  1
# CHECK-NEXT:  [14] .symtab           SYMTAB

## Test that g1 and g2 are local, but g3 is an undefined global.

# CHECK:      0000000000000000     0 NOTYPE  LOCAL  DEFAULT    4 (.foo)    g1
# CHECK-NEXT: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT    6 (.group)  g2
# CHECK-NEXT: 0000000000000000     0 SECTION LOCAL  DEFAULT   11 (sec)     sec
# CHECK-NEXT: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT  UND           g3

# GROUP:       COMDAT group section [    3] `.group' [g1] contains 2 sections:
# GROUP-NEXT:     [Index]    Name
# GROUP-NEXT:     [    4]   .foo
# GROUP-NEXT:     [    5]   .bar
# GROUP-EMPTY:
# GROUP-NEXT:  COMDAT group section [    6] `.group' [g2] contains 1 sections:
# GROUP-NEXT:     [Index]    Name
# GROUP-NEXT:     [    7]   .zed
# GROUP-EMPTY:
# GROUP-NEXT:  COMDAT group section [    8] `.group' [g3] contains 2 sections:
# GROUP-NEXT:     [Index]    Name
# GROUP-NEXT:     [    9]   .baz
# GROUP-NEXT:     [   10]   .rela.baz
# GROUP-EMPTY:
# GROUP-NEXT:  group section [   12] `.group' [] contains 1 sections:
# GROUP-NEXT:     [Index]    Name
# GROUP-NEXT:     [   13]   .qux

	.section	.foo,"axG",@progbits,g1,comdat
g1:
        nop

        .section	.bar,"ax?",@progbits
        nop

        .section	.zed,"axG",@progbits,g2,comdat
        nop

        .section	.baz,"axG",@progbits,g3,comdat
        .long g3

.section sec
.section .qux,"axG",@progbits,sec
