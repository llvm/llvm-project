// RUN: llvm-mc -triple x86_64-apple-macosx13.0.0 -filetype=obj %s -o - | llvm-readobj --symbols - | FileCheck %s

// Aliases inherit these n_desc fields from their aliasees. Of these flags,
// only N_ALT_ENTRY is also preserved when set on the alias itself.
.text
_plain:
  nop

.alt_entry _alt_entry_target
_alt_entry_target:
  nop

.cold _cold_target
_cold_target:
  nop

.no_dead_strip _no_dead_strip_target
_no_dead_strip_target:
  nop

_reference_type_target:
  nop
// Set the reference type after the label, which would otherwise clear it.
.desc _reference_type_target, 2

.symbol_resolver _resolver_target
_resolver_target:
  nop

.globl _alt_entry_on_alias
.alt_entry _alt_entry_on_alias
_alt_entry_on_alias = _plain

.globl _alt_entry_on_target
_alt_entry_on_target = _alt_entry_target

.globl _cold_on_alias
.cold _cold_on_alias
_cold_on_alias = _plain

.globl _cold_on_target
_cold_on_target = _cold_target

.globl _no_dead_strip_on_alias
.no_dead_strip _no_dead_strip_on_alias
_no_dead_strip_on_alias = _plain

.globl _no_dead_strip_on_target
_no_dead_strip_on_target = _no_dead_strip_target

// Different reference types must not be ORed together: the aliasee wins.
.globl _reference_type_both
.desc _reference_type_both, 1
_reference_type_both = _reference_type_target

.globl _reference_type_on_alias
.desc _reference_type_on_alias, 1
_reference_type_on_alias = _plain

.globl _reference_type_on_target
_reference_type_on_target = _reference_type_target

.globl _resolver_on_alias
.symbol_resolver _resolver_on_alias
_resolver_on_alias = _plain

.globl _resolver_on_target
_resolver_on_target = _resolver_target

// Preserve the aliasee's alt-entry bit along with both weak bits on the alias.
.globl _auto_to_alt_entry
.weak_def_can_be_hidden _auto_to_alt_entry
_auto_to_alt_entry = _alt_entry_target

// Only the weak bits are taken from the alias, not its no-dead-strip bit.
.globl _weak_no_dead_strip
.weak_definition _weak_no_dead_strip
.no_dead_strip _weak_no_dead_strip
_weak_no_dead_strip = _plain

// Preserve the aliasee's no-dead-strip bit when adding the alias's weak bit.
.globl _weak_to_no_dead_strip
.weak_definition _weak_to_no_dead_strip
_weak_to_no_dead_strip = _no_dead_strip_target

// CHECK-LABEL: Name: _alt_entry_on_alias
// CHECK:       RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:  Flags [ (0x200)
// CHECK-NEXT:    AltEntry (0x200)
// CHECK-NEXT:  ]

// CHECK-LABEL: Name: _alt_entry_on_target
// CHECK:       RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:  Flags [ (0x200)
// CHECK-NEXT:    AltEntry (0x200)
// CHECK-NEXT:  ]

// CHECK-LABEL: Name: _auto_to_alt_entry
// CHECK-NEXT:  Extern
// CHECK-NEXT:  Type: Section (0xE)
// CHECK-NEXT:  Section: __text (0x1)
// CHECK-NEXT:  RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:  Flags [ (0x2C0)
// CHECK-NEXT:    AltEntry (0x200)
// CHECK-NEXT:    WeakDef (0x80)
// CHECK-NEXT:    WeakRef (0x40)
// CHECK-NEXT:  ]
// CHECK-NEXT:  Value: 0x1

// CHECK-LABEL: Name: _cold_on_alias
// CHECK:       RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:  Flags [ (0x0)
// CHECK-NEXT:  ]

// CHECK-LABEL: Name: _cold_on_target
// CHECK:       RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:  Flags [ (0x400)
// CHECK-NEXT:    ColdFunc (0x400)
// CHECK-NEXT:  ]

// CHECK-LABEL: Name: _no_dead_strip_on_alias
// CHECK:       RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:  Flags [ (0x0)
// CHECK-NEXT:  ]

// CHECK-LABEL: Name: _no_dead_strip_on_target
// CHECK:       RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:  Flags [ (0x20)
// CHECK-NEXT:    NoDeadStrip (0x20)
// CHECK-NEXT:  ]

// CHECK-LABEL: Name: _reference_type_both
// CHECK:       RefType: ReferenceFlagDefined (0x2)
// CHECK-NEXT:  Flags [ (0x0)
// CHECK-NEXT:  ]

// CHECK-LABEL: Name: _reference_type_on_alias
// CHECK:       RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:  Flags [ (0x0)
// CHECK-NEXT:  ]

// CHECK-LABEL: Name: _reference_type_on_target
// CHECK:       RefType: ReferenceFlagDefined (0x2)
// CHECK-NEXT:  Flags [ (0x0)
// CHECK-NEXT:  ]

// CHECK-LABEL: Name: _resolver_on_alias
// CHECK:       RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:  Flags [ (0x0)
// CHECK-NEXT:  ]

// CHECK-LABEL: Name: _resolver_on_target
// CHECK:       RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:  Flags [ (0x100)
// CHECK-NEXT:    SymbolResolver (0x100)
// CHECK-NEXT:  ]

// CHECK-LABEL: Name: _weak_no_dead_strip
// CHECK-NEXT:  Extern
// CHECK-NEXT:  Type: Section (0xE)
// CHECK-NEXT:  Section: __text (0x1)
// CHECK-NEXT:  RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:  Flags [ (0x80)
// CHECK-NEXT:    WeakDef (0x80)
// CHECK-NEXT:  ]
// CHECK-NEXT:  Value: 0x0

// CHECK-LABEL: Name: _weak_to_no_dead_strip
// CHECK-NEXT:  Extern
// CHECK-NEXT:  Type: Section (0xE)
// CHECK-NEXT:  Section: __text (0x1)
// CHECK-NEXT:  RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:  Flags [ (0xA0)
// CHECK-NEXT:    NoDeadStrip (0x20)
// CHECK-NEXT:    WeakDef (0x80)
// CHECK-NEXT:  ]
// CHECK-NEXT:  Value: 0x3
