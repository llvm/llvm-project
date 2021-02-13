# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t
# RUN: llvm-readobj -r %t | FileCheck %s
# RUN: llvm-readelf -s %t | FileCheck %s --check-prefix=SYM

defined1:
defined2:
defined3:
        .symver defined1, bar1@zed
        .symver undefined1, bar2@zed

        .symver defined2, bar3@@zed

        .symver defined3, bar5@@@zed
        .symver undefined3, bar6@@@zed

        .long defined1
        .long undefined1
        .long defined2
        .long defined3
        .long undefined3

        .global global1
        .symver global1, g1@@zed
global1:

// CHECK:      Relocations [
// CHECK-NEXT:   Section {{.*}} .rela.text {
// CHECK-NEXT:     0x0 R_X86_64_32 .text 0x0
// CHECK-NEXT:     0x4 R_X86_64_32 bar2@zed 0x0
// CHECK-NEXT:     0x8 R_X86_64_32 .text 0x0
// CHECK-NEXT:     0xC R_X86_64_32 .text 0x0
// CHECK-NEXT:     0x10 R_X86_64_32 bar6@zed 0x0
// CHECK-NEXT:   }
// CHECK-NEXT: ]

# SYM:      Symbol table '.symtab' contains 11 entries:
# SYM-NEXT: Num:    Value          Size Type    Bind   Vis       Ndx Name
# SYM-NEXT:   0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT   UND 
# SYM-NEXT:   1: 0000000000000000     0 SECTION LOCAL  DEFAULT     2 .text
# SYM-NEXT:   2: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT     2 defined1
# SYM-NEXT:   3: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT     2 defined2
# SYM-NEXT:   4: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT     2 bar1@zed
# SYM-NEXT:   5: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT     2 bar3@@zed
# SYM-NEXT:   6: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT     2 bar5@@zed
# SYM-NEXT:   7: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT   UND bar2@zed
# SYM-NEXT:   8: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT   UND bar6@zed
# SYM-NEXT:   9: 0000000000000014     0 NOTYPE  GLOBAL DEFAULT     2 g1@@zed
# SYM-NEXT:  10: 0000000000000014     0 NOTYPE  GLOBAL DEFAULT     2 global1
