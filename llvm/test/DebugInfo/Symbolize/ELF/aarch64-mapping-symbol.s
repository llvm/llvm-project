# REQUIRES: aarch64-registered-target
## Ignore AArch64 mapping symbols (with a prefix of $d or $x).

# RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o %t

## Verify that mapping symbols are actually present in the object at expected
## addresses.
# RUN: llvm-nm --special-syms %t | FileCheck %s -check-prefix MAPPING_SYM

# MAPPING_SYM:      0000000000000000 t $d.0
# MAPPING_SYM-NEXT: 000000000000000c t $d.2
# MAPPING_SYM-NEXT: 0000000000000004 t $x.1
# MAPPING_SYM-NEXT: 0000000000000000 T foo

# RUN: llvm-symbolizer --obj=%t 0 4 0xc | FileCheck %s -check-prefix SYMBOL

# SYMBOL:      foo
# SYMBOL-NEXT: ??:0:0
# SYMBOL-EMPTY:
# SYMBOL:      foo
# SYMBOL-NEXT: ??:0:0
# SYMBOL-EMPTY:
# SYMBOL:      foo
# SYMBOL-NEXT: ??:0:0

    .global foo
foo:
    .word 32
    nop
    nop
    .word 42
