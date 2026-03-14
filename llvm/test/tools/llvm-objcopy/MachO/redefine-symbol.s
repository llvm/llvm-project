# REQUIRES: x86-registered-target

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos %s -o %t

## A non-existent symbol does not error.
# RUN: llvm-objcopy --redefine-sym _func=_cnuf1234 --redefine-sym _foo=_ba --redefine-sym=_notexist= %t %t2 2>&1 | count 0
# RUN: llvm-readobj --symbols %t2 | FileCheck %s

# RUN: echo '_func _cnuf1234 #rename func' > %t.rename.txt
# RUN: echo '  _foo _ba ' >> %t.rename.txt
# RUN: echo '_notexist _notexist' >> %t.rename.txt
# RUN: llvm-objcopy --redefine-syms %t.rename.txt %t %t3 2>&1 | count 0
# RUN: cmp %t2 %t3

# CHECK:      Symbol {
# CHECK-NEXT:   Name: _ba (1)
# CHECK-NEXT:   Extern
# CHECK-NEXT:   Type: Section (0xE)
# CHECK-NEXT:   Section: __const (0x2)
# CHECK-NEXT:   RefType: UndefinedNonLazy (0x0)
# CHECK-NEXT:   Flags [ (0x0)
# CHECK-NEXT:   ]
# CHECK-NEXT:   Value: 0x0
# CHECK-NEXT: }
# CHECK-NEXT: Symbol {
# CHECK-NEXT:   Name: _cnuf1234 (5)
# CHECK-NEXT:   Extern
# CHECK-NEXT:   Type: Section (0xE)
# CHECK-NEXT:   Section: __text (0x1)
# CHECK-NEXT:   RefType: UndefinedNonLazy (0x0)
# CHECK-NEXT:   Flags [ (0x0)
# CHECK-NEXT:   ]
# CHECK-NEXT:   Value: 0x0
# CHECK-NEXT: }

## Check that --redefine-sym executes before --strip-symbol.
# RUN: llvm-objcopy --strip-symbol _foo --redefine-sym _foo=_ba %t %t.notstripped
# RUN: llvm-readobj --symbols %t.notstripped | FileCheck %s --check-prefix=NOTSTRIPPED
# NOTSTRIPPED: Name: _ba
# NOTSTRIPPED: Name: _func

## FIXME: _ba should be stripped after --strip-symbol is implemented.
# RUN: llvm-objcopy --strip-symbol _ba --redefine-sym _foo=_ba %t %t.noba
# RUN: llvm-readobj --symbols %t.noba | FileCheck %s --check-prefix=NOTSTRIPPED

.globl _func
_func:

.section __TEXT,__const
.globl _foo
_foo:
