// RUN: llvm-mc -filetype=obj -triple x86_64-pc-windows-elf %s -o - | llvm-readobj -r --symbols - | FileCheck %s

// Verify that MSVC C++ mangled symbols are not affected by the ELF
// GNU-style symbol versioning. The ELF format is used on Windows by
// the MCJIT execution engine.

        .long "??_R0?AVexception@std@@@8"
        .long "@??_R0?AVinvalid_argument@std@@@8"
        .long "__imp_??_R0?AVlogic_error@std@@@8"
        .long "__imp_@??_R0PAVexception@std@@@8"


// CHECK:       Relocations [
// CHECK-NEXT:    Section {{.*}} .rela.text {
// CHECK-NEXT:      0x0 R_X86_64_32 ??_R0?AVexception@std@@@8 0x0
// CHECK-NEXT:      0x4 R_X86_64_32 @??_R0?AVinvalid_argument@std@@@8 0x0
// CHECK-NEXT:      0x8 R_X86_64_32 __imp_??_R0?AVlogic_error@std@@@8 0x0
// CHECK-NEXT:      0xC R_X86_64_32 __imp_@??_R0PAVexception@std@@@8 0x0
// CHECK-NEXT:    }
// CHECK-NEXT:  ]

// CHECK:       Symbols [
// CHECK:         Symbol {
// CHECK:           Name: ??_R0?AVexception@std@@@8
// CHECK-NEXT:      Value: 0x0
// CHECK-NEXT:      Size: 0
// CHECK-NEXT:      Binding: Global (0x1)
// CHECK-NEXT:      Type: None (0x0)
// CHECK-NEXT:      Other: 0
// CHECK-NEXT:      Section: Undefined (0x0)
// CHECK-NEXT:    }
// CHECK-NEXT:    Symbol {
// CHECK-NEXT:      Name: @??_R0?AVinvalid_argument@std@@@8
// CHECK-NEXT:      Value: 0x0
// CHECK-NEXT:      Size: 0
// CHECK-NEXT:      Binding: Global (0x1)
// CHECK-NEXT:      Type: None (0x0)
// CHECK-NEXT:      Other: 0
// CHECK-NEXT:      Section: Undefined (0x0)
// CHECK-NEXT:    }
// CHECK-NEXT:    Symbol {
// CHECK-NEXT:      Name: __imp_??_R0?AVlogic_error@std@@@8
// CHECK-NEXT:      Value: 0x0
// CHECK-NEXT:      Size: 0
// CHECK-NEXT:      Binding: Global (0x1)
// CHECK-NEXT:      Type: None (0x0)
// CHECK-NEXT:      Other: 0
// CHECK-NEXT:      Section: Undefined (0x0)
// CHECK-NEXT:    }
// CHECK-NEXT:    Symbol {
// CHECK-NEXT:      Name: __imp_@??_R0PAVexception@std@@@8
// CHECK-NEXT:      Value: 0x0
// CHECK-NEXT:      Size: 0
// CHECK-NEXT:      Binding: Global (0x1)
// CHECK-NEXT:      Type: None (0x0)
// CHECK-NEXT:      Other: 0
// CHECK-NEXT:      Section: Undefined (0x0)
// CHECK-NEXT:    }
// CHECK-NEXT:  ]
