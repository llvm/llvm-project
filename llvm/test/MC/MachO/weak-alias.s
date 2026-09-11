// RUN: llvm-mc -triple x86_64-apple-macosx13.0.0 -filetype=obj %s -o - | llvm-readobj --symbols - | FileCheck %s

.text
_foo:
  nop

.globl _external
_external = _foo

.globl _weak
.weak_definition _weak
_weak = _foo

.globl _auto
.weak_def_can_be_hidden _auto
_auto = _foo

_local = _foo

.globl _weak_target
.weak_definition _weak_target
_weak_target:
  nop

.globl _strong_to_weak
_strong_to_weak = _weak_target

// CHECK:      Name: _local
// CHECK-NEXT: Type: Section (0xE)
// CHECK-NEXT: Section: __text (0x1)
// CHECK-NEXT: RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT: Flags [ (0x0)
// CHECK-NEXT: ]
// CHECK-NEXT: Value: 0x0

// CHECK:      Name: _auto
// CHECK-NEXT: Extern
// CHECK-NEXT: Type: Section (0xE)
// CHECK-NEXT: Section: __text (0x1)
// CHECK-NEXT: RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT: Flags [ (0xC0)
// CHECK-NEXT:   WeakDef (0x80)
// CHECK-NEXT:   WeakRef (0x40)
// CHECK-NEXT: ]
// CHECK-NEXT: Value: 0x0

// CHECK:      Name: _external
// CHECK-NEXT: Extern
// CHECK-NEXT: Type: Section (0xE)
// CHECK-NEXT: Section: __text (0x1)
// CHECK-NEXT: RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT: Flags [ (0x0)
// CHECK-NEXT: ]
// CHECK-NEXT: Value: 0x0

// CHECK:      Name: _strong_to_weak
// CHECK-NEXT: Extern
// CHECK-NEXT: Type: Section (0xE)
// CHECK-NEXT: Section: __text (0x1)
// CHECK-NEXT: RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT: Flags [ (0x80)
// CHECK-NEXT:   WeakDef (0x80)
// CHECK-NEXT: ]
// CHECK-NEXT: Value: 0x1

// CHECK:      Name: _weak
// CHECK-NEXT: Extern
// CHECK-NEXT: Type: Section (0xE)
// CHECK-NEXT: Section: __text (0x1)
// CHECK-NEXT: RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT: Flags [ (0x80)
// CHECK-NEXT:   WeakDef (0x80)
// CHECK-NEXT: ]
// CHECK-NEXT: Value: 0x0

// CHECK:      Name: _weak_target
// CHECK-NEXT: Extern
// CHECK-NEXT: Type: Section (0xE)
// CHECK-NEXT: Section: __text (0x1)
// CHECK-NEXT: RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT: Flags [ (0x80)
// CHECK-NEXT:   WeakDef (0x80)
// CHECK-NEXT: ]
// CHECK-NEXT: Value: 0x1
