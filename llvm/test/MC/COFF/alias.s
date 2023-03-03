// RUN: llvm-mc -filetype=obj -triple i686-pc-win32 %s -o - \
// RUN:   | llvm-readobj --symbols -r - | FileCheck %s

local1:
external_aliased_to_local = local1

        .globl  global_aliased_to_external
global_aliased_to_external = external1

        .globl  global_aliased_to_local
local2:
global_aliased_to_local = local2

        .weak   weak_aliased_to_external
weak_aliased_to_external = external2

// Generate relocs against the above aliases.
        .long external_aliased_to_local
        .long global_aliased_to_external
        .long global_aliased_to_local
        .long weak_aliased_to_external

// CHECK:      Relocations [
// CHECK:        0x0 IMAGE_REL_I386_DIR32 external_aliased_to_local
// CHECK:        0x4 IMAGE_REL_I386_DIR32 external1
// CHECK:        0x8 IMAGE_REL_I386_DIR32 global_aliased_to_local
// CHECK:        0xC IMAGE_REL_I386_DIR32 weak_aliased_to_external
// CHECK:      ]
// CHECK:      Symbols [
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: .text
// CHECK-NEXT:     Value: 0
// CHECK-NEXT:     Section: .text (1)
// CHECK-NEXT:     BaseType: Null (0x0)
// CHECK-NEXT:     ComplexType: Null (0x0)
// CHECK-NEXT:     StorageClass: Static (0x3)
// CHECK-NEXT:     AuxSymbolCount: 1
// CHECK:        }
// CHECK:        Symbol {
// CHECK:          Name: local1
// CHECK-NEXT:     Value: 0
// CHECK-NEXT:     Section: .text (1)
// CHECK-NEXT:     BaseType: Null (0x0)
// CHECK-NEXT:     ComplexType: Null (0x0)
// CHECK-NEXT:     StorageClass: Static (0x3)
// CHECK-NEXT:     AuxSymbolCount: 0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK:          Name: global_aliased_to_external
// CHECK-NEXT:     Value: 0
// CHECK-NEXT:     Section: IMAGE_SYM_UNDEFINED (0)
// CHECK-NEXT:     BaseType: Null (0x0)
// CHECK-NEXT:     ComplexType: Null (0x0)
// CHECK-NEXT:     StorageClass: External (0x2)
// CHECK-NEXT:     AuxSymbolCount: 0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: external1
// CHECK-NEXT:     Value: 0
// CHECK-NEXT:     Section: IMAGE_SYM_UNDEFINED (0)
// CHECK-NEXT:     BaseType: Null (0x0)
// CHECK-NEXT:     ComplexType: Null (0x0)
// CHECK-NEXT:     StorageClass: External (0x2)
// CHECK-NEXT:     AuxSymbolCount: 0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: global_aliased_to_local
// CHECK-NEXT:     Value: 0
// CHECK-NEXT:     Section: .text (1)
// CHECK-NEXT:     BaseType: Null (0x0)
// CHECK-NEXT:     ComplexType: Null (0x0)
// CHECK-NEXT:     StorageClass: External (0x2)
// CHECK-NEXT:     AuxSymbolCount: 0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: local2
// CHECK-NEXT:     Value: 0
// CHECK-NEXT:     Section: .text (1)
// CHECK-NEXT:     BaseType: Null (0x0)
// CHECK-NEXT:     ComplexType: Null (0x0)
// CHECK-NEXT:     StorageClass: Static (0x3)
// CHECK-NEXT:     AuxSymbolCount: 0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: weak_aliased_to_external
// CHECK-NEXT:     Value: 0
// CHECK-NEXT:     Section: IMAGE_SYM_UNDEFINED (0)
// CHECK-NEXT:     BaseType: Null (0x0)
// CHECK-NEXT:     ComplexType: Null (0x0)
// CHECK-NEXT:     StorageClass: WeakExternal (0x69)
// CHECK-NEXT:     AuxSymbolCount: 1
// CHECK-NEXT:     AuxWeakExternal {
// CHECK-NEXT:       Linked: external2
// CHECK-NEXT:       Search: Alias (0x3)
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: external2
// CHECK-NEXT:     Value: 0
// CHECK-NEXT:     Section: IMAGE_SYM_UNDEFINED (0)
// CHECK-NEXT:     BaseType: Null (0x0)
// CHECK-NEXT:     ComplexType: Null (0x0)
// CHECK-NEXT:     StorageClass: External (0x2)
// CHECK-NEXT:     AuxSymbolCount: 0
// CHECK-NEXT:   }
// CHECK-NEXT: ]
