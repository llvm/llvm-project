// RUN: llvm-mc -triple thumbv7m-apple-darwin-eabi %s -filetype=obj -o %t
// RUN:     llvm-readobj --symbols %t | FileCheck %s

        .data
        var1 = var2
        .long var1
        .long var2
        .long var2 + 4
defined_early:
        .long 0

        alias_to_early = defined_early
        alias_to_late = defined_late

defined_late:
        .long 0

        .global extern_test
        extern_test = var2

        alias_to_local = Ltmp0
Ltmp0:

// CHECK: Symbols [

        // defined_early was defined. Actually has value 0xc.
// CHECK: Symbol {
// CHECK-NEXT:   Name: defined_early
// CHECK-NEXT:   Type: Section (0xE)
// CHECK-NEXT:   Section: __data (0x2)
// CHECK-NEXT:   RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:   Flags [ (0x0)
// CHECK-NEXT:   ]
// CHECK-NEXT:   Value: 0x[[DEFINED_EARLY:[0-9A-F]+]]
// CHECK-NEXT: }

        // alias_to_early was an alias to defined_early. But we can resolve it.
// CHECK: Symbol {
// CHECK-NEXT:   Name: alias_to_early
// CHECK-NEXT:   Type: Section (0xE)
// CHECK-NEXT:   Section: __data (0x2)
// CHECK-NEXT:   RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:   Flags [ (0x0)
// CHECK-NEXT:   ]
// CHECK-NEXT:   Value: 0x[[DEFINED_EARLY]]
// CHECK-NEXT: }

        // alias_to_late was an alias to defined_late. But we can resolve it.
// CHECK: Symbol {
// CHECK-NEXT:   Name: alias_to_late
// CHECK-NEXT:   Type: Section (0xE)
// CHECK-NEXT:   Section: __data (0x2)
// CHECK-NEXT:   RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:   Flags [ (0x0)
// CHECK-NEXT:   ]
// CHECK-NEXT:   Value: 0x[[DEFINED_LATE:[0-9A-F]+]]
// CHECK-NEXT: }

        // defined_late was defined. Just after defined_early.
// CHECK: Symbol {
// CHECK-NEXT:   Name: defined_late
// CHECK-NEXT:   Type: Section (0xE)
// CHECK-NEXT:   Section: __data (0x2)
// CHECK-NEXT:   RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:   Flags [ (0x0)
// CHECK-NEXT:   ]
// CHECK-NEXT:   Value: 0x[[DEFINED_LATE]]
// CHECK-NEXT: }

        // alias_to_local is an alias, but what it points to has no
        // MachO representation. We must resolve it.
// CHECK: Symbol {
// CHECK-NEXT:   Name: alias_to_local (42)
// CHECK-NEXT:   Type: Section (0xE)
// CHECK-NEXT:   Section: __data (0x2)
// CHECK-NEXT:   RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:   Flags [ (0x0)
// CHECK-NEXT:   ]
// CHECK-NEXT:   Value: 0x14
// CHECK-NEXT: }

        // extern_test was a pure alias to the unknown "var2".
        // N_INDR and Extern.
// CHECK:   Name: extern_test
// CHECK-NEXT:   Extern
// CHECK-NEXT:   Type: Indirect (0xA)
// CHECK-NEXT:   Section:  (0x0)
// CHECK-NEXT:   RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:   Flags [ (0x0)
// CHECK-NEXT:   ]
// CHECK-NEXT:   Value: 0x[[VAR2_STRINGINDEX:[0-9a-f]+]]
// CHECK-NEXT: }

        // var1 was another alias to an unknown variable. Not extern this time.
// CHECK: Symbol {
// CHECK-NEXT:   Name: var1 (89)
// CHECK-NEXT:   Type: Indirect (0xA)
// CHECK-NEXT:   Section:  (0x0)
// CHECK-NEXT:   RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:   Flags [ (0x0)
// CHECK-NEXT:   ]
// CHECK-NEXT:   Value: 0x[[VAR2_STRINGINDEX]]
// CHECK-NEXT: }

        // var2 was a normal undefined (extern) symbol.
// CHECK: Symbol {
// CHECK-NEXT:   Name: var2
// CHECK-NEXT:   Extern
// CHECK-NEXT:   Type: Undef (0x0)
// CHECK-NEXT:   Section:  (0x0)
// CHECK-NEXT:   RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:   Flags [ (0x0)
// CHECK-NEXT:   ]
// CHECK-NEXT:   Value: 0x0
// CHECK-NEXT: }
