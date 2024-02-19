# REQUIRES: csky-registered-target
## Ignore CSKY mapping symbols (with a prefix of $d or $t).

# RUN: llvm-mc -filetype=obj -triple=csky %s -o %t

## Verify that mapping symbols are actually present in the object at expected
## addresses.
# RUN: llvm-nm --special-syms %t | FileCheck %s -check-prefix MAPPING_SYM

# MAPPING_SYM:      00000000 t $d.0
# MAPPING_SYM-NEXT: 00000008 t $d.2
# MAPPING_SYM-NEXT: 00000004 t $t.1
# MAPPING_SYM-NEXT: 00000000 T foo

# RUN: llvm-symbolizer --obj=%t 0 4 0xc | FileCheck %s -check-prefix SYMBOL

# SYMBOL:      foo
# SYMBOL-NEXT: ??:0:0
# SYMBOL-EMPTY:
# SYMBOL:      foo
# SYMBOL-NEXT: ??:0:0
# SYMBOL-EMPTY:
# SYMBOL:      foo
# SYMBOL-NEXT: ??:0:0

.globl foo
foo:
  .long 32
  nop
  nop
  .long 42
