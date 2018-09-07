// RUN: llvm-mc -filetype=obj -triple thumbv7m-arm-linux-gnu %s -o - \
// RUN: | llvm-readobj -s -t | FileCheck %s

        .text
        bx      lr

        .section        .text.foo,"axy"
        bx      lr

// CHECK:      Section {
// CHECK:        Name: .text
// CHECK-NEXT:   Type: SHT_PROGBITS (0x1)
// CHECK-NEXT:   Flags [ (0x6)
// CHECK-NEXT:     SHF_ALLOC (0x2)
// CHECK-NEXT:     SHF_EXECINSTR (0x4)
// CHECK-NEXT:   ]
// CHECK:      }

// CHECK:      Section {
// CHECK:        Name: .text.foo
// CHECK-NEXT:   Type: SHT_PROGBITS (0x1)
// CHECK-NEXT:   Flags [ (0x20000006)
// CHECK-NEXT:     SHF_ALLOC (0x2)
// CHECK-NEXT:     SHF_ARM_PURECODE (0x20000000)
// CHECK-NEXT:     SHF_EXECINSTR (0x4)
// CHECK-NEXT:   ]
// CHECK:      }
