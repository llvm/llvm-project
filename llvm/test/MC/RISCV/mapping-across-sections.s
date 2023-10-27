# RUN: llvm-mc -triple=riscv32 -filetype=obj < %s | llvm-readelf -Ss - | FileCheck %s
# RUN: llvm-mc -triple=riscv64 -filetype=obj < %s | llvm-readelf -Ss - | FileCheck %s

        .text
        nop

# .wibble should *not* inherit .text's mapping symbol. It's a completely
# different section.
        .section .wibble
        nop

# A section should be able to start with a $d.
        .section .starts_data
        .word 42

# Changing back to .text should not emit a redundant $x.
        .text
        nop

# With all those constraints, we want:
#   + .text to have $x at 0 and no others
#   + .wibble to have $x at 0
#   + .starts_data to have $d at 0

## Capture section indices.
# CHECK: [[#TEXT:]]] .text
# CHECK: [[#WIBBLE:]]] .wibble
# CHECK: [[#STARTS_DATA:]]] .starts_data

# CHECK:    Value  Size Type    Bind   Vis     Ndx              Name
# CHECK: 00000000     0 NOTYPE  LOCAL  DEFAULT [[#TEXT]]        $x
# CHECK: 00000000     0 NOTYPE  LOCAL  DEFAULT [[#WIBBLE]]      $x
# CHECK: 00000000     0 NOTYPE  LOCAL  DEFAULT [[#STARTS_DATA]] $d
