# REQUIRES: x86-registered-target

## Show that llvm-objcopy copies the indirect symbol table properly.
# RUN: llvm-mc -assemble -triple x86_64-apple-darwin9 -filetype=obj %s -o %t
# RUN: llvm-objcopy %t %t.copy
# RUN: llvm-readobj --symbols --macho-indirect-symbols %t.copy \
# RUN:   | FileCheck %s

# __DATA,__nl_symbol_ptr
.non_lazy_symbol_pointer
bar:
        .long 0
baz:
        .long 0

.indirect_symbol bar

# __DATA,__la_symbol_ptr
.lazy_symbol_pointer
foo:
        .long 0

.indirect_symbol foo

# CHECK:      Symbols [
# CHECK-NEXT:   Symbol {
# CHECK-NEXT:     Name: bar (5)
# CHECK-NEXT:     Type: Section (0xE)
# CHECK-NEXT:     Section: __nl_symbol_ptr (0x2)
# CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
# CHECK-NEXT:     Flags [ (0x0)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Value: 0x0
# CHECK-NEXT:   }
# CHECK-NEXT:   Symbol {
# CHECK-NEXT:     Name: baz (1)
# CHECK-NEXT:     Type: Section (0xE)
# CHECK-NEXT:     Section: __nl_symbol_ptr (0x2)
# CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
# CHECK-NEXT:     Flags [ (0x0)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Value: 0x4
# CHECK-NEXT:   }
# CHECK-NEXT:   Symbol {
# CHECK-NEXT:     Name: foo (9)
# CHECK-NEXT:     Type: Section (0xE)
# CHECK-NEXT:     Section: __la_symbol_ptr (0x3)
# CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
# CHECK-NEXT:     Flags [ (0x0)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Value: 0x8
# CHECK-NEXT:   }
# CHECK-NEXT: ]
# CHECK-NEXT: Indirect Symbols {
# CHECK-NEXT:   Number: 2
# CHECK-NEXT:   Symbols [
# CHECK-NEXT:     Entry {
# CHECK-NEXT:       Entry Index: 0
# CHECK-NEXT:       Symbol Index: 0x80000000
# CHECK-NEXT:     }
# CHECK-NEXT:     Entry {
# CHECK-NEXT:       Entry Index: 1
# CHECK-NEXT:       Symbol Index: 0x2
# CHECK-NEXT:     }
# CHECK-NEXT:   ]
# CHECK-NEXT: }
