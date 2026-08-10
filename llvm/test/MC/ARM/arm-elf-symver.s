@ RUN: llvm-mc -filetype=obj -triple arm-none-linux-gnueabi %s -o - | llvm-readobj -r --symbols - | FileCheck %s
@ RUN: llvm-mc -filetype=obj -triple thumb-none-linux-gnueabi %s -o - | llvm-readobj -r --symbols - | FileCheck %s

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

@ CHECK: Relocations [
@ CHECK-NEXT:   Section {{.*}} .rel.text {
@ CHECK-NEXT:     0x0 R_ARM_ABS32 .text
@ CHECK-NEXT:     0x4 R_ARM_ABS32 bar2@zed
@ CHECK-NEXT:     0x8 R_ARM_ABS32 .text
@ CHECK-NEXT:     0xC R_ARM_ABS32 .text
@ CHECK-NEXT:     0x10 R_ARM_ABS32 bar6@zed
@ CHECK-NEXT:   }
@ CHECK-NEXT: ]

@ CHECK:        Symbol {
@ CHECK:        Symbol {
@ CHECK-NEXT:     Name: .text (0)
@ CHECK-NEXT:     Value: 0x0
@ CHECK-NEXT:     Size: 0
@ CHECK-NEXT:     Binding: Local (0x0)
@ CHECK-NEXT:     Type: Section (0x3)
@ CHECK-NEXT:     Other: 0
@ CHECK-NEXT:     Section: .text
@ CHECK-NEXT:   }
@ CHECK-NEXT:   Symbol {
@ CHECK-NEXT:     Name: defined1
@ CHECK-NEXT:     Value: 0x0
@ CHECK-NEXT:     Size: 0
@ CHECK-NEXT:     Binding: Local (0x0)
@ CHECK-NEXT:     Type: None (0x0)
@ CHECK-NEXT:     Other: 0
@ CHECK-NEXT:     Section: .text
@ CHECK-NEXT:   }
@ CHECK-NEXT:   Symbol {
@ CHECK-NEXT:     Name: defined2
@ CHECK-NEXT:     Value: 0x0
@ CHECK-NEXT:     Size: 0
@ CHECK-NEXT:     Binding: Local (0x0)
@ CHECK-NEXT:     Type: None (0x0)
@ CHECK-NEXT:     Other: 0
@ CHECK-NEXT:     Section: .text
@ CHECK-NEXT:   }
@ CHECK:  Symbol {
@ CHECK:    Name: bar1@zed
@ CHECK-NEXT:     Value: 0x0
@ CHECK-NEXT:     Size: 0
@ CHECK-NEXT:     Binding: Local (0x0)
@ CHECK-NEXT:     Type: None (0x0)
@ CHECK-NEXT:     Other: 0
@ CHECK-NEXT:     Section: .text
@ CHECK-NEXT:   }
@ CHECK-NEXT:   Symbol {
@ CHECK-NEXT:     Name: bar3@@zed
@ CHECK-NEXT:     Value: 0x0
@ CHECK-NEXT:     Size: 0
@ CHECK-NEXT:     Binding: Local (0x0)
@ CHECK-NEXT:     Type: None (0x0)
@ CHECK-NEXT:     Other: 0
@ CHECK-NEXT:     Section: .text
@ CHECK-NEXT:   }
@ CHECK-NEXT:   Symbol {
@ CHECK-NEXT:     Name: bar5@@zed
@ CHECK-NEXT:     Value: 0x0
@ CHECK-NEXT:     Size: 0
@ CHECK-NEXT:     Binding: Local (0x0)
@ CHECK-NEXT:     Type: None (0x0)
@ CHECK-NEXT:     Other: 0
@ CHECK-NEXT:     Section: .text
@ CHECK-NEXT:   }
@ CHECK-NEXT:   Symbol {
@ CHECK-NEXT:     Name: global1
@ CHECK-NEXT:     Value: 0x14
@ CHECK-NEXT:     Size: 0
@ CHECK-NEXT:     Binding: Global (0x1)
@ CHECK-NEXT:     Type: None (0x0)
@ CHECK-NEXT:     Other: 0
@ CHECK-NEXT:     Section: .text
@ CHECK-NEXT:   }
@ CHECK-NEXT:   Symbol {
@ CHECK-NEXT:     Name: bar2@zed
@ CHECK-NEXT:     Value: 0x0
@ CHECK-NEXT:     Size: 0
@ CHECK-NEXT:     Binding: Global (0x1)
@ CHECK-NEXT:     Type: None (0x0)
@ CHECK-NEXT:     Other: 0
@ CHECK-NEXT:     Section: Undefined (0x0)
@ CHECK-NEXT:   }
@ CHECK-NEXT:   Symbol {
@ CHECK-NEXT:     Name: bar6@zed
@ CHECK-NEXT:     Value: 0x0
@ CHECK-NEXT:     Size: 0
@ CHECK-NEXT:     Binding: Global (0x1)
@ CHECK-NEXT:     Type: None (0x0)
@ CHECK-NEXT:     Other: 0
@ CHECK-NEXT:     Section: Undefined (0x0)
@ CHECK-NEXT:   }
@ CHECK-NEXT:   Symbol {
@ CHECK-NEXT:     Name: g1@@zed
@ CHECK-NEXT:     Value: 0x14
@ CHECK-NEXT:     Size: 0
@ CHECK-NEXT:     Binding: Global (0x1)
@ CHECK-NEXT:     Type: None (0x0)
@ CHECK-NEXT:     Other: 0
@ CHECK-NEXT:     Section: .text
@ CHECK-NEXT:   }
@ CHECK-NEXT: ]
