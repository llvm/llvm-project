# REQUIRES: x86-registered-target

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos %s -o %t

# RUN: llvm-objcopy -W _func %t %t2
# RUN: llvm-readobj --symbols %t2 | FileCheck %s -check-prefix=CHECK-1

# CHECK-1:      Symbol {
# CHECK-1-NEXT:   Name: _foo (1)
# CHECK-1-NEXT:   Extern
# CHECK-1-NEXT:   Type: Section (0xE)
# CHECK-1-NEXT:   Section: __const (0x2)
# CHECK-1-NEXT:   RefType: UndefinedNonLazy (0x0)
# CHECK-1-NEXT:   Flags [ (0x0)
# CHECK-1-NEXT:   ]
# CHECK-1-NEXT:   Value: 0x0
# CHECK-1-NEXT: }
# CHECK-1-NEXT: Symbol {
# CHECK-1-NEXT:   Name: _func (6)
# CHECK-1-NEXT:   Extern
# CHECK-1-NEXT:   Type: Section (0xE)
# CHECK-1-NEXT:   Section: __text (0x1)
# CHECK-1-NEXT:   RefType: UndefinedNonLazy (0x0)
# CHECK-1-NEXT:   Flags [ (0x80)
# CHECK-1-NEXT:     WeakDef (0x80)
# CHECK-1-NEXT:   ]
# CHECK-1-NEXT:   Value: 0x0
# CHECK-1-NEXT: }

# RUN: echo _foo > %t.weaken.txt
# RUN: echo _func >> %t.weaken.txt
# RUN: llvm-objcopy --weaken-symbols %t.weaken.txt %t %t3
# RUN: llvm-readobj --symbols %t3 | FileCheck %s -check-prefix=CHECK-2

# CHECK-2:      Symbol {
# CHECK-2-NEXT:   Name: _foo (1)
# CHECK-2-NEXT:   Extern
# CHECK-2-NEXT:   Type: Section (0xE)
# CHECK-2-NEXT:   Section: __const (0x2)
# CHECK-2-NEXT:   RefType: UndefinedNonLazy (0x0)
# CHECK-2-NEXT:   Flags [ (0x80)
# CHECK-2-NEXT:     WeakDef (0x80)
# CHECK-2-NEXT:   ]
# CHECK-2-NEXT:   Value: 0x0
# CHECK-2-NEXT: }
# CHECK-2-NEXT: Symbol {
# CHECK-2-NEXT:   Name: _func (6)
# CHECK-2-NEXT:   Extern
# CHECK-2-NEXT:   Type: Section (0xE)
# CHECK-2-NEXT:   Section: __text (0x1)
# CHECK-2-NEXT:   RefType: UndefinedNonLazy (0x0)
# CHECK-2-NEXT:   Flags [ (0x80)
# CHECK-2-NEXT:     WeakDef (0x80)
# CHECK-2-NEXT:   ]
# CHECK-2-NEXT:   Value: 0x0
# CHECK-2-NEXT: }

## Verify --weaken-symbol plays nice with --redefine-sym.
# RUN: llvm-objcopy -W _foo --redefine-sym _foo=_bar %t %t4
# RUN: llvm-readobj --symbols %t4 | FileCheck %s -check-prefix=CHECK-3

# CHECK-3:      Symbol {
# CHECK-3-NEXT:   Name: _bar (1)
# CHECK-3-NEXT:   Extern
# CHECK-3-NEXT:   Type: Section (0xE)
# CHECK-3-NEXT:   Section: __const (0x2)
# CHECK-3-NEXT:   RefType: UndefinedNonLazy (0x0)
# CHECK-3-NEXT:   Flags [ (0x80)
# CHECK-3-NEXT:     WeakDef (0x80)
# CHECK-3-NEXT:   ]
# CHECK-3-NEXT:   Value: 0x0
# CHECK-3-NEXT: }
# CHECK-3-NEXT: Symbol {
# CHECK-3-NEXT:   Name: _func (6)
# CHECK-3-NEXT:   Extern
# CHECK-3-NEXT:   Type: Section (0xE)
# CHECK-3-NEXT:   Section: __text (0x1)
# CHECK-3-NEXT:   RefType: UndefinedNonLazy (0x0)
# CHECK-3-NEXT:   Flags [ (0x0)
# CHECK-3-NEXT:   ]
# CHECK-3-NEXT:   Value: 0x0
# CHECK-3-NEXT: }

.globl _func
_func:

.section __TEXT,__const
.globl _foo
_foo:
