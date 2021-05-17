# RUN: llvm-mc -filetype=obj -triple=x86_64 %s | llvm-readelf -s - | FileCheck %s

// This is a long test that checks that the aliases created by weakref are
// never in the symbol table and that the only case it causes a symbol to
// be output as a weak undefined symbol is if that variable is not defined
// in this file and all the references to it are done via the alias.

# CHECK:      Num:    Value          Size Type    Bind   Vis       Ndx Name
# CHECK-NEXT:   0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT   UND 
# CHECK-NEXT:   1: 0000000000000000     0 SECTION LOCAL  DEFAULT     2 .text
# CHECK-NEXT:   2: 0000000000000018     0 NOTYPE  LOCAL  DEFAULT     2 bar6
# CHECK-NEXT:   3: 0000000000000018     0 NOTYPE  LOCAL  DEFAULT     2 bar7
# CHECK-NEXT:   4: 000000000000001c     0 NOTYPE  LOCAL  DEFAULT     2 bar8
# CHECK-NEXT:   5: 0000000000000020     0 NOTYPE  LOCAL  DEFAULT     2 bar9
# CHECK-NEXT:   6: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT   UND bar1
# CHECK-NEXT:   7: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT   UND bar2
# CHECK-NEXT:   8: 0000000000000000     0 NOTYPE  WEAK   DEFAULT   UND bar3
# CHECK-NEXT:   9: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT   UND bar4
# CHECK-NEXT:  10: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT   UND bar5
# CHECK-NEXT:  11: 0000000000000028     0 NOTYPE  GLOBAL DEFAULT     2 bar10
# CHECK-NEXT:  12: 0000000000000030     0 NOTYPE  GLOBAL DEFAULT     2 bar11
# CHECK-NEXT:  13: 0000000000000030     0 NOTYPE  GLOBAL DEFAULT     2 bar12
# CHECK-NEXT:  14: 0000000000000034     0 NOTYPE  GLOBAL DEFAULT     2 bar13
# CHECK-NEXT:  15: 0000000000000038     0 NOTYPE  GLOBAL DEFAULT     2 bar14
# CHECK-NEXT:  16: 0000000000000040     0 NOTYPE  GLOBAL DEFAULT     2 bar15

        .weakref foo1, bar1

        .weakref foo2, bar2
        .long bar2

        .weakref foo3, bar3
        .long foo3

        .weakref foo4, bar4
        .long foo4
        .long bar4

        .weakref foo5, bar5
        .long bar5
        .long foo5

bar6:
        .weakref foo6, bar6

bar7:
        .weakref foo7, bar7
        .long bar7

bar8:
        .weakref foo8, bar8
        .long foo8

bar9:
        .weakref foo9, bar9
        .long foo9
        .long bar9

bar10:
        .global bar10
        .weakref foo10, bar10
        .long bar10
        .long foo10

bar11:
        .global bar11
        .weakref foo11, bar11

bar12:
        .global bar12
        .weakref foo12, bar12
        .long bar12

bar13:
        .global bar13
        .weakref foo13, bar13
        .long foo13

bar14:
        .global bar14
        .weakref foo14, bar14
        .long foo14
        .long bar14

bar15:
        .global bar15
        .weakref foo15, bar15
        .long bar15
        .long foo15
