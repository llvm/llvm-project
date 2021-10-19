# REQUIRES: arm-registered-target
## Ignore ARM mapping symbols (with a prefix of $a, $d or $t).

# RUN: llvm-mc -filetype=obj -triple=armv7-none-linux %s -o %t

## Verify that mapping symbols are actually present in the object at expected
## addresses.
# RUN: llvm-nm --special-syms %t | FileCheck %s -check-prefix MAPPING_A

# MAPPING_A:      00000004 t $a.1
# MAPPING_A-NEXT: 00000000 t $d.0
# MAPPING_A-NEXT: 00000008 t $d.2
# MAPPING_A-NEXT: 00000000 T foo

# RUN: llvm-mc -filetype=obj -triple=thumbv7-none-linux %s -o %tthumb
# RUN: llvm-nm --special-syms %tthumb | FileCheck %s -check-prefix MAPPING_T

# MAPPING_T:      00000000 t $d.0
# MAPPING_T-NEXT: 00000006 t $d.2
# MAPPING_T-NEXT: 00000004 t $t.1
# MAPPING_T-NEXT: 00000000 T foo

# RUN: llvm-symbolizer --obj=%t 4 8 | FileCheck %s -check-prefix SYMBOL
# RUN: llvm-symbolizer --obj=%tthumb 4 8 | FileCheck %s -check-prefix SYMBOL

# SYMBOL:       foo
# SYMBOL-NEXT:  ??:0:0
# SYMBOL-EMPTY:
# SYMBOL-NEXT:  foo
# SYMBOL-NEXT:  ??:0:0

.globl foo
foo:
  .word 32
  nop
  .word 32
