# RUN: llvm-mc -filetype=obj -assemble \
# RUN: -triple=aarch64- %s -o - \
# RUN: | llvm-readobj -S --symbols - | FileCheck %s
# CHECK:     Name: $d.1 ({{[1-9][0-9]+}})
# CHECK-NEXT:     Value: 0x4
# CHECK-NEXT:     Size: 0
# CHECK-NEXT:     Binding: Local (0x0)
# CHECK-NEXT:     Type: None (0x0)
# CHECK-NEXT:     Other: 0
# CHECK-NEXT:     Section: .text (0x2)
# CHECK-NEXT:   }

.text
nop
.zero 4
