# REQUIRES: x86-registered-target

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos %s -o %t

# RUN: llvm-objcopy --weaken %t %t2
# RUN: llvm-readobj --symbols %t2 --sort-symbols=name | FileCheck %s

# CHECK:      Symbols [
# CHECK-NEXT: Symbol {
# CHECK-NEXT:   Name: _global ({{[0-9]+}})
# CHECK-NEXT:   Extern
# CHECK-NEXT:   Type: Section (0xE)
# CHECK-NEXT:   Section: __text (0x1)
# CHECK-NEXT:   RefType: UndefinedNonLazy (0x0)
# CHECK-NEXT:   Flags [ (0x80)
# CHECK-NEXT:     WeakDef (0x80)
# CHECK-NEXT:   ]
# CHECK-NEXT:   Value: 0x0
# CHECK-NEXT: }
# CHECK-NEXT: Symbol {
# CHECK-NEXT:   Name: _global_data ({{[0-9]+}})
# CHECK-NEXT:   Extern
# CHECK-NEXT:   Type: Section (0xE)
# CHECK-NEXT:   Section: __const (0x2)
# CHECK-NEXT:   RefType: UndefinedNonLazy (0x0)
# CHECK-NEXT:   Flags [ (0x80)
# CHECK-NEXT:     WeakDef (0x80)
# CHECK-NEXT:   ]
# CHECK-NEXT:   Value: 0x0
# CHECK-NEXT: }
# CHECK-NEXT: Symbol {
# CHECK-NEXT:   Name: _local ({{[0-9]+}})
# CHECK-NEXT:   Type: Section (0xE)
# CHECK-NEXT:   Section: __text (0x1)
# CHECK-NEXT:   RefType: UndefinedNonLazy (0x0)
# CHECK-NEXT:   Flags [ (0x0)
# CHECK-NEXT:   ]
# CHECK-NEXT:   Value: 0x0
# CHECK-NEXT: }
# CHECK-NEXT: Symbol {
# CHECK-NEXT:   Name: _local_data ({{[0-9]+}})
# CHECK-NEXT:   Type: Section (0xE)
# CHECK-NEXT:   Section: __const (0x2)
# CHECK-NEXT:   RefType: UndefinedNonLazy (0x0)
# CHECK-NEXT:   Flags [ (0x0)
# CHECK-NEXT:   ]
# CHECK-NEXT:   Value: 0x0
# CHECK-NEXT: }
# CHECK-NEXT: Symbol {
# CHECK-NEXT:   Name: _weak ({{[0-9]+}})
# CHECK-NEXT:   Type: Section (0xE)
# CHECK-NEXT:   Section: __text (0x1)
# CHECK-NEXT:   RefType: UndefinedNonLazy (0x0)
# CHECK-NEXT:   Flags [ (0x80)
# CHECK-NEXT:     WeakDef (0x80)
# CHECK-NEXT:   ]
# CHECK-NEXT:   Value: 0x0
# CHECK-NEXT: }
# CHECK-NEXT: Symbol {
# CHECK-NEXT:   Name: _weak_data ({{[0-9]+}})
# CHECK-NEXT:   Type: Section (0xE)
# CHECK-NEXT:   Section: __const (0x2)
# CHECK-NEXT:   RefType: UndefinedNonLazy (0x0)
# CHECK-NEXT:   Flags [ (0x80)
# CHECK-NEXT:     WeakDef (0x80)
# CHECK-NEXT:   ]
# CHECK-NEXT:   Value: 0x0
# CHECK-NEXT: }
# CHECK-NEXT: Symbol {
# CHECK-NEXT:   Name: _weak_global ({{[0-9]+}})
# CHECK-NEXT:   Extern
# CHECK-NEXT:   Type: Section (0xE)
# CHECK-NEXT:   Section: __text (0x1)
# CHECK-NEXT:   RefType: UndefinedNonLazy (0x0)
# CHECK-NEXT:   Flags [ (0x80)
# CHECK-NEXT:     WeakDef (0x80)
# CHECK-NEXT:   ]
# CHECK-NEXT:   Value: 0x0
# CHECK-NEXT: }
# CHECK-NEXT: ]

.globl _global
_global:

_local:

.weak_definition _weak
_weak:

.weak_definition _weak_global
.globl _weak_global
_weak_global:

.section __TEXT,__const
.globl _global_data
_global_data:
_local_data:

.weak_definition _weak_data
_weak_data:
